import os
import hydra
from omegaconf import OmegaConf
import torch
import utils
import pandas as pd
from torch.utils import benchmark

FIGURES_DIR = "../thesis/figures/"


def main():
    results = []
    for model_name in ["vaes", "unet", "fpn"]:
        result = benchmark_single_model(model_name)
        results.append(result)
    df = pd.DataFrame(results)
    df.set_index("Name", inplace=True)
    df.columns = pd.MultiIndex.from_tuples(
        [c.split(",") for c in df.columns]
    )
    print(df)
    df = df.transpose()
    tex = (
        df.style.highlight_min(props="textbf:--rwrap;", axis=1)
        .format(precision=2)
        .to_latex(
            caption="Characteristics of the various architectures with ResNet50 as backbone. Inference measurements were done on a NVIDIA GTX 1070 with an image shape of 128x128.",
            label="tab:model_characteristics",
            position="ht",
            position_float="centering",
            # environment="longtable",
            clines="skip-last;data",
            multirow_align="t",
            hrules=True,
        )
        .replace("Name", "Batch Size")
    )
    with open(
        os.path.join(FIGURES_DIR, "model_characteristics.tex"), mode="w"
    ) as f:
        f.write(tex)


def benchmark_single_model(model_name):
    with hydra.initialize("../src/conf/model"):
        cfg = hydra.compose(config_name=model_name)
        cfg = OmegaConf.to_container(cfg)
        cfg["label_channels"] = 25
        results = utils.benchmark_subprocess(cfg, model_name)
        summary = utils.get_model_summary_in_subprocess(cfg)
        # model = utils.instantiate_dict(cfg, label_channels=25)
        # results = benchmark_inference(model, model_name)

        # summary = utils.get_model_summary(model)
        # del model

        return {
            "Name": model_name,
            "Parameters (x$1e^6$), ": summary["total_params"] * 1e-6,
            "Total MAC (x$1e^9$), ": summary["total_mult_adds"] * 1e-9,
            **{
                f"Inference Speed (ms),{r['batch_size']}": r[
                    "timer"
                ].median
                * 1000
                for r in results
            },
            **{
                f"Memory Usage (mb),{r['batch_size']}": r["peak memory"]
                * 1e-6
                for r in results
            },
        }


def benchmark_inference(
    model, model_name: str, batch_size=1, input_shape=(3, 128, 128)
):
    model = model.to("cuda")
    results = []
    for batch_size in [1, 2, 8, 32]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=None)
        input = torch.rand(batch_size, *input_shape, device="cuda")
        t1 = benchmark.Timer(
            stmt="model(input)",
            globals={"model": model, "input": input},
            num_threads=torch.get_num_threads(),
            label=model_name,
            description=f"{batch_size=}",
        )

        with torch.inference_mode():
            result = t1.adaptive_autorange(0.00001)
        results.append(
            {
                "batch_size": batch_size,
                "timer": result,
                "peak memory": torch.cuda.max_memory_allocated(
                    device=None
                ),
            }
        )
    del model
    return results


if __name__ == "__main__":
    main()
