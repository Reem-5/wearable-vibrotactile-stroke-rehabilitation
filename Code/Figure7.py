"""
Figure 7: FMA-UE Change Scores — Treatment versus Control

This script:
1. Loads the de-identified clinical trial dataset.
2. Compares FMA-UE change scores between the treatment and control groups.
3. Uses two-sided Mann–Whitney U tests.
4. Reports rank-based effect-size estimates.
5. Applies Benjamini-Hochberg false discovery rate correction across the
   three exploratory FMA-UE domains.
6. Calculates an independent-samples Hodges–Lehmann estimate and bootstrap
   95% confidence interval for each domain.
7. Generates Figure 7 and saves it in TIFF, EPS, SVG, PDF, and PNG formats.

Expected repository structure:

wearable-vibrotactile-stroke-rehabilitation/
│
├── Code/
│   └── Fig7_FMA_UE_Change_Scores.py
├── Data/
│   └── Stroke_Vibrotactile_Rehabilitation_Data.xlsx
└── Figures/
"""

from math import sqrt
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# =====================================================
# Reproducibility
# =====================================================
RANDOM_SEED = 42
N_BOOTSTRAP = 10_000

np.random.seed(RANDOM_SEED)


# =====================================================
# Repository paths
# =====================================================
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "Data"
    / "Stroke_Vibrotactile_Rehabilitation_Data.xlsx"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "Figures"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "The dataset could not be found.\n"
        f"Expected location: {DATA_FILE}\n"
        "Place the de-identified Excel dataset inside the Data folder."
    )


# =====================================================
# Dataset settings
# =====================================================
SHEET_NAME = "FMA_UE"

GROUP_ORDER = ["Treatment", "Control"]

DOMAINS = [
    {
        "domain": "Motor",
        "column": "Motor Change",
        "panel": "A",
        "title": "FMA-UE Motor — Change",
    },
    {
        "domain": "Sensation",
        "column": "Sensation Change",
        "panel": "B",
        "title": "FMA-UE Sensation — Change",
    },
    {
        "domain": "Joint Motion",
        "column": "Joint motion Change",
        "panel": "C",
        "title": "FMA-UE Joint Motion — Change",
    },
    {
        "domain": "Joint Pain",
        "column": "Joint Pain Change",
        "panel": "D",
        "title": "FMA-UE Joint Pain — Change",
    },
]

EXPLORATORY_DOMAINS = [
    "Sensation",
    "Joint Motion",
    "Joint Pain",
]


# =====================================================
# Global figure formatting
# =====================================================
FONT_FAMILY = "Arial"

TITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 13
STATS_SIZE = 13
PANEL_LABEL_SIZE = 22
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
        "axes.linewidth": LINE_WIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# =====================================================
# Statistical helper functions
# =====================================================
def prepare_independent_groups(
    data: pd.DataFrame,
    change_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract valid treatment and control observations.

    Parameters
    ----------
    data
        DataFrame containing Group and change-score columns.
    change_column
        Name of the change-score column.

    Returns
    -------
    treatment, control
        Numeric NumPy arrays for the two study groups.
    """
    treatment = pd.to_numeric(
        data.loc[data["Group"] == "Treatment", change_column],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    control = pd.to_numeric(
        data.loc[data["Group"] == "Control", change_column],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(treatment) == 0 or len(control) == 0:
        raise ValueError(
            f"Treatment or control observations are missing for: {change_column}"
        )

    return treatment, control


def mannwhitney_with_effect(
    treatment: np.ndarray,
    control: np.ndarray,
) -> dict[str, float | int]:
    """
    Perform a two-sided Mann–Whitney U test and calculate effect size r.

    The reported effect size is calculated as:

        r = |Z| / sqrt(n1 + n2)
    """
    test_result = mannwhitneyu(
        treatment,
        control,
        alternative="two-sided",
    )

    u_statistic = float(test_result.statistic)
    p_value = float(test_result.pvalue)

    n_treatment = len(treatment)
    n_control = len(control)

    expected_u = n_treatment * n_control / 2

    standard_deviation_u = sqrt(
        n_treatment
        * n_control
        * (n_treatment + n_control + 1)
        / 12
    )

    z_value = (
        (u_statistic - expected_u) / standard_deviation_u
        if standard_deviation_u > 0
        else 0.0
    )

    effect_size = abs(z_value) / np.sqrt(
        n_treatment + n_control
    )

    return {
        "U": u_statistic,
        "p_raw": p_value,
        "Z": z_value,
        "r": effect_size,
        "n_treatment": n_treatment,
        "n_control": n_control,
    }


def hodges_lehmann_independent_ci(
    treatment: np.ndarray,
    control: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = 95.0,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """
    Calculate the independent-samples Hodges–Lehmann location-shift estimate.

    The point estimate is the median of all pairwise differences:

        treatment score - control score

    A percentile bootstrap confidence interval is generated by independently
    resampling observations within each study group.
    """
    pairwise_differences = (
        treatment[:, np.newaxis] - control[np.newaxis, :]
    ).ravel()

    hl_estimate = float(
        np.median(pairwise_differences)
    )

    random_generator = np.random.default_rng(seed)

    bootstrap_estimates = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for index in range(n_bootstrap):
        treatment_sample = random_generator.choice(
            treatment,
            size=len(treatment),
            replace=True,
        )

        control_sample = random_generator.choice(
            control,
            size=len(control),
            replace=True,
        )

        bootstrap_pairwise_differences = (
            treatment_sample[:, np.newaxis]
            - control_sample[np.newaxis, :]
        ).ravel()

        bootstrap_estimates[index] = np.median(
            bootstrap_pairwise_differences
        )

    alpha = 100.0 - confidence_level

    confidence_interval_low = float(
        np.percentile(
            bootstrap_estimates,
            alpha / 2,
        )
    )

    confidence_interval_high = float(
        np.percentile(
            bootstrap_estimates,
            100.0 - alpha / 2,
        )
    )

    return {
        "HL": hl_estimate,
        "CI_low": confidence_interval_low,
        "CI_high": confidence_interval_high,
    }


def format_p_value(p_value: float) -> str:
    """Format p-values for figures and terminal output."""
    if p_value < 0.001:
        return "< 0.001"

    return f"= {p_value:.3f}"


# =====================================================
# Load and validate data
# =====================================================
data = pd.read_excel(
    DATA_FILE,
    sheet_name=SHEET_NAME,
)

data.columns = data.columns.str.strip()

required_columns = {
    "Group",
    *[domain["column"] for domain in DOMAINS],
}

missing_columns = required_columns.difference(data.columns)

if missing_columns:
    raise KeyError(
        "The following required columns are missing:\n"
        + "\n".join(
            f"- {column}"
            for column in sorted(missing_columns)
        )
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

data = data[
    data["Group"].isin(GROUP_ORDER)
].copy()

if data.empty:
    raise ValueError(
        "No valid Treatment or Control observations were found."
    )

for domain in DOMAINS:
    data[domain["column"]] = pd.to_numeric(
        data[domain["column"]],
        errors="coerce",
    )


# =====================================================
# Compute statistics
# =====================================================
results: dict[str, dict[str, Any]] = {}

for domain_index, domain in enumerate(DOMAINS):
    domain_name = domain["domain"]
    change_column = domain["column"]

    treatment_values, control_values = prepare_independent_groups(
        data,
        change_column,
    )

    test_results = mannwhitney_with_effect(
        treatment_values,
        control_values,
    )

    hl_results = hodges_lehmann_independent_ci(
        treatment_values,
        control_values,
        seed=RANDOM_SEED + domain_index,
    )

    results[domain_name] = {
        **test_results,
        **hl_results,
        "p_adj": np.nan,
    }


# =====================================================
# Apply FDR correction to exploratory domains
# =====================================================
raw_secondary_p_values = [
    results[domain]["p_raw"]
    for domain in EXPLORATORY_DOMAINS
]

adjusted_secondary_p_values = multipletests(
    raw_secondary_p_values,
    alpha=0.05,
    method="fdr_bh",
)[1]

for domain, adjusted_p_value in zip(
    EXPLORATORY_DOMAINS,
    adjusted_secondary_p_values,
):
    results[domain]["p_adj"] = float(
        adjusted_p_value
    )


# =====================================================
# Print statistical results
# =====================================================
print("\nFigure 7: FMA-UE Between-Group Change Scores")
print("Difference direction: Treatment minus Control")
print("=" * 62)

for domain in DOMAINS:
    domain_name = domain["domain"]
    result = results[domain_name]

    print(f"\n{domain_name}")
    print("-" * 40)
    print(
        f"Treatment n = {result['n_treatment']}, "
        f"Control n = {result['n_control']}"
    )
    print(f"Mann–Whitney U = {result['U']:.2f}")
    print(f"Z = {result['Z']:.3f}")
    print(f"Raw p-value = {result['p_raw']:.3f}")

    if domain_name in EXPLORATORY_DOMAINS:
        print(
            f"FDR-adjusted p-value = "
            f"{result['p_adj']:.3f}"
        )

    print(f"Effect size r = {result['r']:.2f}")
    print(
        f"Hodges–Lehmann estimate = "
        f"{result['HL']:.2f}"
    )
    print(
        "Bootstrap 95% CI = "
        f"[{result['CI_low']:.2f}, "
        f"{result['CI_high']:.2f}]"
    )


# =====================================================
# Figure settings
# =====================================================
GROUP_PALETTE = {
    "Treatment": "#5DADE2",
    "Control": "#EC7063",
}


def format_axis(axis: plt.Axes) -> None:
    """Apply consistent axis formatting."""
    axis.set_xlabel(
        "Group",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_ylabel(
        "Change score",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
    )

    axis.grid(False)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for spine in axis.spines.values():
        spine.set_linewidth(LINE_WIDTH)


def add_panel(
    axis: plt.Axes,
    domain_name: str,
    change_column: str,
    title: str,
    panel_label: str,
) -> None:
    """Add one FMA-UE domain panel to Figure 7."""
    plotting_data = data[
        ["Group", change_column]
    ].dropna().copy()

    sns.violinplot(
        data=plotting_data,
        x="Group",
        y=change_column,
        order=GROUP_ORDER,
        hue="Group",
        hue_order=GROUP_ORDER,
        palette=GROUP_PALETTE,
        legend=False,
        inner=None,
        cut=0,
        linewidth=LINE_WIDTH,
        width=0.45,
        ax=axis,
    )

    sns.boxplot(
        data=plotting_data,
        x="Group",
        y=change_column,
        order=GROUP_ORDER,
        width=0.18,
        showcaps=True,
        showfliers=False,
        boxprops={
            "facecolor": "black",
            "alpha": 0.70,
            "linewidth": LINE_WIDTH,
        },
        medianprops={
            "color": "white",
            "linewidth": 1.4,
        },
        whiskerprops={
            "linewidth": LINE_WIDTH,
            "color": "black",
        },
        capprops={
            "linewidth": LINE_WIDTH,
            "color": "black",
        },
        ax=axis,
    )

    sns.stripplot(
        data=plotting_data,
        x="Group",
        y=change_column,
        order=GROUP_ORDER,
        jitter=0.10,
        size=6.25,
        color="red",
        edgecolor="black",
        linewidth=0.4,
        zorder=4,
        ax=axis,
    )

    axis.set_title(
        title,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        pad=12,
    )

    format_axis(axis)

    result = results[domain_name]

    if domain_name == "Motor":
        statistics_text = (
            f"U = {result['U']:.2f}   "
            f"Z = {result['Z']:.2f}\n"
            f"p {format_p_value(result['p_raw'])}   "
            f"r = {result['r']:.2f}"
        )
    else:
        statistics_text = (
            f"U = {result['U']:.2f}   "
            f"Z = {result['Z']:.2f}\n"
            f"p {format_p_value(result['p_raw'])}\n"
            f"$p_{{adj}}$ "
            f"{format_p_value(result['p_adj'])}   "
            f"r = {result['r']:.2f}"
        )

    y_minimum, y_maximum = axis.get_ylim()
    y_range = y_maximum - y_minimum

    if y_range == 0:
        y_range = 1.0

    axis.set_ylim(
        y_minimum,
        y_maximum + 0.40 * y_range,
    )

    statistics_y_position = (
        y_maximum + 0.20 * y_range
    )

    axis.text(
        0.5,
        statistics_y_position,
        statistics_text,
        transform=axis.get_yaxis_transform(),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=STATS_SIZE,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.90,
        },
    )

    axis.text(
        -0.12,
        1.05,
        panel_label,
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="bottom",
    )


# =====================================================
# Create Figure 7
# =====================================================
figure, axes = plt.subplots(
    2,
    2,
    figsize=(11, 8),
)

axes = axes.flatten()

for axis, domain in zip(axes, DOMAINS):
    add_panel(
        axis=axis,
        domain_name=domain["domain"],
        change_column=domain["column"],
        title=domain["title"],
        panel_label=domain["panel"],
    )

figure.tight_layout()


# =====================================================
# Save Figure 7
# =====================================================
output_files = {
    "TIFF": OUTPUT_DIRECTORY / "Fig7.tif",
    "EPS": OUTPUT_DIRECTORY / "Fig7.eps",
    "SVG": OUTPUT_DIRECTORY / "Fig7.svg",
    "PDF": OUTPUT_DIRECTORY / "Fig7.pdf",
    "PNG": OUTPUT_DIRECTORY / "Fig7.png",
}

figure.savefig(
    output_files["TIFF"],
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
    pil_kwargs={"compression": "tiff_lzw"},
)

figure.savefig(
    output_files["EPS"],
    bbox_inches="tight",
    pad_inches=0.02,
)

figure.savefig(
    output_files["SVG"],
    bbox_inches="tight",
    pad_inches=0.02,
)

figure.savefig(
    output_files["PDF"],
    bbox_inches="tight",
    pad_inches=0.02,
)

figure.savefig(
    output_files["PNG"],
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02,
)

print("\nFigure 7 files saved successfully:")

for file_format, file_path in output_files.items():
    print(f"{file_format}: {file_path}")

plt.show()