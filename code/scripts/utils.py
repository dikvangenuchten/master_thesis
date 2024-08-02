import sys
from pathlib import Path

import pandas as pd
import torchinfo
import hydra


def fix_path():
    # Nasty hack to add src directory
    sys.path.insert(1, str(Path(__file__).parents[1] / "src"))
    # I am sorry


fix_path()  # noqa


def instantiate_dict(config, *args, **kwargs) -> object:
    return hydra.utils.instantiate(config, *args, **kwargs)


def get_model_summary(model, input_shape=(3, 128, 128)):
    return torchinfo.summary(model, (1, *input_shape))


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
        .to_latex(
            caption=caption,
            label=label,
            position="ht",
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
        .to_latex(
            caption=caption
            + "\\\\Where:\\\\\\hphantom{tabb}Coef. the effectsize.\\\\\\hphantom{tabb}P> |t| the p-value. Bolded if significant ($\\alpha\le0.05$).",
            label=label,
            position="ht",
            hrules=True,
        )
    )
