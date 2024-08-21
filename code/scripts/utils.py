import multiprocessing
import multiprocessing.managers
import sys
from pathlib import Path

import pandas as pd
import torchinfo
import hydra

try:
    multiprocessing.set_start_method("spawn")
except Exception:
    pass


def fix_path():
    # Nasty hack to add src directory
    sys.path.insert(1, str(Path(__file__).parents[1] / "src"))
    # I am sorry


fix_path()  # noqa


def instantiate_dict(config, *args, **kwargs) -> object:
    return hydra.utils.instantiate(config, *args, **kwargs)


def get_model_summary(model, input_shape=(3, 128, 128)):
    if isinstance(model, dict):
        model = instantiate_dict(model)
    return torchinfo.summary(model, (1, *input_shape))


def _get_model_summary_from_process(model, return_dict):
    fix_path()
    val = get_model_summary(model)
    return_dict["total_params"] = float(val.total_params)
    return_dict["total_mult_adds"] = float(val.total_mult_adds)
    return return_dict


def get_model_summary_in_subprocess(model_dict):
    # return_dict = multiprocessing.managers.M()
    man = multiprocessing.Manager()
    return_dict = man.dict()
    p = multiprocessing.Process(
        target=_get_model_summary_from_process,
        args=(model_dict, return_dict),
    )
    p.start()
    p.join()
    return return_dict


def benchmark_subprocess(model_dict, name):
    man = multiprocessing.Manager()
    return_list = man.list()
    p = multiprocessing.Process(
        target=benchmark_inference,
        args=(model_dict, name),
        kwargs={"return_value": return_list},
    )

    p.start()
    p.join()
    return return_list


def format_anova_table(
    df_anova: pd.DataFrame, caption: str, label: str
) -> str:
    return (
        df_anova.style.format(na_rep="n.a.", precision=2)
        .highlight_between(
            subset="PR(>F)",
            axis=1,
            left=-1,
            right=0.05,
            props="textbf:--rwrap;",
        )
        .format_index(escape="latex", axis=1)
        .format_index(escape="latex", axis=0)
        .to_latex(
            caption=caption,
            label=label,
            position="ht",
            position_float="centering",
            hrules=True,
        )
    )


def format_effect_size(effect_size, caption: str, label: str):
    return (
        effect_size.style.format(na_rep="n.a.", precision=2)
        .highlight_between(
            subset="P>|t|",
            axis=1,
            left=-1,
            right=0.05,
            props="textbf:--rwrap;",
        )
        .format_index(escape="latex", axis=1)
        .format_index(escape="latex", axis=0)
        .to_latex(
            caption=caption
            + "\\\\Where:\\\\\\hphantom{tabb}Coef. is the effectsize.\\\\\\hphantom{tabb}P> |t| is the $p$-value. Bolded if significant ($\\alpha\le0.05$).",
            label=label,
            position="ht",
            position_float="centering",
            hrules=True,
        )
    )


def benchmark_inference(
    model_config,
    model_name: str,
    batch_size=1,
    input_shape=(3, 128, 128),
    return_value=None,
):
    import torch
    from torch.utils import benchmark

    model = instantiate_dict(model_config)
    model = model.to("cuda")
    if return_value is None:
        return_value = []
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
            result = t1.adaptive_autorange(
                1e-5, min_run_time=30, max_run_time=60
            )
        return_value.append(
            {
                "batch_size": batch_size,
                "timer": result,
                "peak memory": torch.cuda.max_memory_allocated(
                    device=None
                ),
            }
        )
    del model
    return return_value
