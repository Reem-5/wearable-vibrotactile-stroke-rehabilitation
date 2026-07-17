"""
Figure 5: Baseline FMA-UE Motor Scores

This script:
1. Loads the de-identified clinical trial dataset.
2. Compares baseline FMA-UE motor scores between the treatment and control groups
   using a two-sided Mann–Whitney U test.
3. Calculates an effect size based on the standardized U statistic.
4. Generates and saves Figure 5 in multiple formats.

Expected repository structure:

wearable-vibrotactile-stroke-rehabilitation/
│
├── Code/
│   └── Fig5_Baseline_FMA_UE_Motor.py
├── Data/
│   └── Stroke_Vibrotactile_Rehabilitation_Data.xlsx
└── Figures/
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


# =====================================================
# Figure formatting
# =====================================================
FONT_FAMILY = "Arial"

TITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 13
STATS_SIZE = 13
LEGEND_SIZE = 13
LINE_WIDTH = 1.2

sns.set_theme(style="white", context="paper")

mpl.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": TICK_LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": LINE_WIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# =====================================================
# Statistical helper function
# =====================================================
def mannwhitney_with_effect(
    group_1: pd.Series,
    group_2: pd.Series,
) -> tuple[float, float, float, float]:
    """
    Perform a two-sided Mann–Whitney U test and calculate an r effect size.

    Parameters
    ----------
    group_1 : pandas.Series
        Numeric observations from the first group.
    group_2 : pandas.Series
        Numeric observations from the second group.

    Returns
    -------
    u_statistic : float
        Mann–Whitney U statistic for group 1.
    p_value : float
        Two-sided p-value.
    z_value : float
        Standardized U statistic.
    r_effect : float
        Absolute effect-size estimate, calculated as |Z| / sqrt(N).
    """
    group_1 = pd.to_numeric(group_1, errors="coerce").dropna()
    group_2 = pd.to_numeric(group_2, errors="coerce").dropna()

    if group_1.empty or group_2.empty:
        raise ValueError(
            "Both comparison groups must contain at least one valid observation."
        )

    u_statistic, p_value = mannwhitneyu(
        group_1,
        group_2,
        alternative="two-sided",
    )

    n_1 = len(group_1)
    n_2 = len(group_2)

    mean_u = n_1 * n_2 / 2
    standard_deviation_u = np.sqrt(
        n_1 * n_2 * (n_1 + n_2 + 1) / 12
    )

    z_value = (u_statistic - mean_u) / standard_deviation_u
    r_effect = abs(z_value) / np.sqrt(n_1 + n_2)

    return u_statistic, p_value, z_value, r_effect


def format_axis(axis: plt.Axes) -> None:
    """Apply consistent formatting to the figure axis."""
    axis.set_xlabel("Group", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("FMA-UE motor score", fontsize=AXIS_LABEL_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.grid(False)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for spine in axis.spines.values():
        spine.set_linewidth(LINE_WIDTH)


# =====================================================
# Repository paths
# =====================================================
script_path = Path(__file__).resolve()
project_root = script_path.parents[1]

data_file = (
    project_root
    / "Data"
    / "Stroke_Vibrotactile_Rehabilitation_Data.xlsx"
)

output_directory = project_root / "Figures"
output_directory.mkdir(parents=True, exist_ok=True)

if not data_file.exists():
    raise FileNotFoundError(
        "The dataset was not found.\n"
        f"Expected location: {data_file}\n"
        "Place the de-identified Excel file inside the repository's Data folder."
    )


# =====================================================
# Load and prepare data
# =====================================================
data = pd.read_excel(
    data_file,
    sheet_name="FMA_UE",
)

data.columns = data.columns.str.strip()

required_columns = {
    "Group",
    "Baseline_motor Function",
}

missing_columns = required_columns.difference(data.columns)

if missing_columns:
    raise KeyError(
        "The following required columns are missing from the FMA_UE sheet: "
        + ", ".join(sorted(missing_columns))
    )

group_mapping = {
    "T": "Treatment",
    "C": "Control",
    "Treatment": "Treatment",
    "Control": "Control",
}

data["Group"] = (
    data["Group"]
    .astype(str)
    .str.strip()
    .map(group_mapping)
)

baseline_column = "Baseline_motor Function"

data[baseline_column] = pd.to_numeric(
    data[baseline_column],
    errors="coerce",
)

plot_data = data[
    ["Group", baseline_column]
].dropna(
    subset=["Group", baseline_column]
)

plot_data = plot_data[
    plot_data["Group"].isin(["Treatment", "Control"])
].copy()

if plot_data.empty:
    raise ValueError(
        "No valid treatment or control observations were found."
    )


# =====================================================
# Statistical analysis
# =====================================================
treatment_scores = plot_data.loc[
    plot_data["Group"] == "Treatment",
    baseline_column,
]

control_scores = plot_data.loc[
    plot_data["Group"] == "Control",
    baseline_column,
]

u_statistic, p_value, z_value, r_effect = mannwhitney_with_effect(
    treatment_scores,
    control_scores,
)

print("\nFigure 5: Baseline FMA-UE Motor Scores")
print("---------------------------------------")
print(f"Treatment n = {len(treatment_scores)}")
print(f"Control n = {len(control_scores)}")
print(f"U = {u_statistic:.2f}")
print(f"p = {p_value:.3f}")
print(f"Z = {z_value:.3f}")
print(f"r = {r_effect:.3f}")


# =====================================================
# Generate Figure 5
# =====================================================
group_order = ["Treatment", "Control"]

palette = {
    "Treatment": "#5DADE2",
    "Control": "#EC7063",
}

figure, axis = plt.subplots(figsize=(6, 6))

sns.boxplot(
    data=plot_data,
    x="Group",
    y=baseline_column,
    order=group_order,
    hue="Group",
    hue_order=group_order,
    palette=palette,
    width=0.8,
    linewidth=LINE_WIDTH,
    legend=False,
    ax=axis,
)

sns.stripplot(
    data=plot_data,
    x="Group",
    y=baseline_column,
    order=group_order,
    color="black",
    size=6,
    jitter=0.18,
    alpha=0.8,
    ax=axis,
)

axis.set_title(
    "Baseline FMA-UE Motor Scores",
    fontsize=TITLE_SIZE,
    fontweight="normal",
    pad=30,
)

format_axis(axis)

statistics_label = (
    f"U = {u_statistic:.2f}, "
    f"p = {p_value:.3f}, "
    f"r = {r_effect:.2f}"
)

axis.text(
    0.5,
    1.00,
    statistics_label,
    transform=axis.transAxes,
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=STATS_SIZE,
)

figure.tight_layout()


# =====================================================
# Save figure files
# =====================================================
output_stem = "Figure5_Baseline_FMA_UE_Motor_Scores"

figure.savefig(
    output_directory / f"{output_stem}.pdf",
    bbox_inches="tight",
)

figure.savefig(
    output_directory / f"{output_stem}.svg",
    bbox_inches="tight",
)

figure.savefig(
    output_directory / f"{output_stem}.eps",
    bbox_inches="tight",
)

figure.savefig(
    output_directory / f"{output_stem}.png",
    dpi=300,
    bbox_inches="tight",
)

figure.savefig(
    output_directory / f"{output_stem}.tiff",
    dpi=300,
    bbox_inches="tight",
)

print(f"\nFigure files saved to:\n{output_directory}")

plt.show()