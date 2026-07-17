"""
Figure 6: FMA-UE Scores at Baseline and Post-Intervention

This script:
1. Loads the de-identified clinical trial dataset.
2. Compares baseline and post-intervention FMA-UE scores within each group.
3. Uses two-sided paired Wilcoxon signed-rank tests.
4. Reports effect-size estimates.
5. Applies Benjamini-Hochberg false discovery rate correction across the
   three exploratory FMA-UE domains separately within each study group.
6. Reports paired change estimates with bootstrap 95% confidence intervals.
7. Generates Figure 6 and saves it in TIFF, EPS, SVG, and PDF formats.

Expected repository structure:

wearable-vibrotactile-stroke-rehabilitation/
│
├── Code/
│   └── Fig6_FMA_UE_Baseline_vs_Post.py
├── Data/
│   └── Stroke_Vibrotactile_Rehabilitation_Data.xlsx
└── Figures/
"""

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42
N_BOOTSTRAP = 10_000

np.random.seed(RANDOM_SEED)


# ============================================================
# Repository paths
# ============================================================
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


# ============================================================
# Global figure formatting
# ============================================================
FONT_FAMILY = "Arial"

TITLE_SIZE = 12
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
STATS_SIZE = 10
PANEL_LABEL_SIZE = 16
LEGEND_SIZE = 10
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


# ============================================================
# Analysis configuration
# ============================================================
GROUP_ORDER = ["Treatment", "Control"]
TIME_ORDER = ["Baseline", "Post"]

OUTCOMES = [
    {
        "domain": "Motor",
        "baseline": "Baseline_motor Function",
        "post": "Post_motor Function",
        "panel": "A",
    },
    {
        "domain": "Sensation",
        "baseline": "Baseline_Sensation",
        "post": "Post_Sensation",
        "panel": "B",
    },
    {
        "domain": "Joint Motion",
        "baseline": "Baseline_Joint motion",
        "post": "Post_Joint motion",
        "panel": "C",
    },
    {
        "domain": "Joint Pain",
        "baseline": "Baseline_Joint Pain",
        "post": "Post_Joint Pain",
        "panel": "D",
    },
]

EXPLORATORY_DOMAINS = [
    "Sensation",
    "Joint Motion",
    "Joint Pain",
]


# ============================================================
# Statistical functions
# ============================================================
def prepare_paired_values(
    baseline: pd.Series,
    post: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return complete numeric baseline-post pairs.

    Parameters
    ----------
    baseline
        Baseline observations.
    post
        Post-intervention observations.

    Returns
    -------
    baseline_values, post_values
        Numeric NumPy arrays containing complete pairs.
    """
    paired_data = pd.DataFrame(
        {
            "baseline": pd.to_numeric(baseline, errors="coerce"),
            "post": pd.to_numeric(post, errors="coerce"),
        }
    ).dropna()

    if paired_data.empty:
        raise ValueError(
            "No complete baseline-post pairs were available for analysis."
        )

    return (
        paired_data["baseline"].to_numpy(dtype=float),
        paired_data["post"].to_numpy(dtype=float),
    )


def wilcoxon_with_effect(
    baseline: pd.Series,
    post: pd.Series,
) -> dict[str, float | int]:
    """
    Perform a paired two-sided Wilcoxon signed-rank test.

    The reported effect size is calculated as |Z| / sqrt(n), where n is the
    number of complete baseline-post pairs.

    Notes
    -----
    The Z value is approximated from the Wilcoxon statistic using its
    null mean and standard deviation. The signed-rank test is conducted on
    post-intervention minus baseline scores.
    """
    baseline_values, post_values = prepare_paired_values(
        baseline,
        post,
    )

    differences = post_values - baseline_values
    nonzero_differences = differences[differences != 0]

    n_pairs = len(differences)
    n_nonzero = len(nonzero_differences)

    if n_nonzero == 0:
        return {
            "n": n_pairs,
            "n_nonzero": 0,
            "W": 0.0,
            "p_raw": 1.0,
            "Z": 0.0,
            "r": 0.0,
        }

    test_result = wilcoxon(
        post_values,
        baseline_values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    w_statistic = float(test_result.statistic)
    p_value = float(test_result.pvalue)

    expected_w = n_nonzero * (n_nonzero + 1) / 4
    standard_deviation_w = np.sqrt(
        n_nonzero
        * (n_nonzero + 1)
        * (2 * n_nonzero + 1)
        / 24
    )

    z_value = (
        (w_statistic - expected_w) / standard_deviation_w
        if standard_deviation_w > 0
        else 0.0
    )

    effect_size = abs(z_value) / np.sqrt(n_pairs)

    return {
        "n": n_pairs,
        "n_nonzero": n_nonzero,
        "W": w_statistic,
        "p_raw": p_value,
        "Z": z_value,
        "r": effect_size,
    }


def paired_change_ci(
    baseline: pd.Series,
    post: pd.Series,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = 95.0,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """
    Estimate the median paired change and its bootstrap confidence interval.

    Change is defined as:

        post-intervention score - baseline score

    The point estimate is the median of the participant-level paired
    differences. The confidence interval is obtained by bootstrap resampling
    participants with replacement.
    """
    baseline_values, post_values = prepare_paired_values(
        baseline,
        post,
    )

    differences = post_values - baseline_values
    median_change = float(np.median(differences))

    random_generator = np.random.default_rng(seed)

    bootstrap_estimates = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for index in range(n_bootstrap):
        bootstrap_sample = random_generator.choice(
            differences,
            size=len(differences),
            replace=True,
        )
        bootstrap_estimates[index] = np.median(
            bootstrap_sample
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
        "median_change": median_change,
        "ci_low": confidence_interval_low,
        "ci_high": confidence_interval_high,
    }


def format_p_value(p_value: float) -> str:
    """Format a p-value for figure presentation."""
    if p_value < 0.001:
        return "< 0.001"

    return f"= {p_value:.3f}"


def format_axis(axis: plt.Axes) -> None:
    """Apply consistent formatting to one panel."""
    axis.set_xlabel("Group")
    axis.set_ylabel("Score")
    axis.tick_params(axis="both")
    axis.grid(False)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for spine in axis.spines.values():
        spine.set_linewidth(LINE_WIDTH)


# ============================================================
# Load and validate data
# ============================================================
data = pd.read_excel(
    DATA_FILE,
    sheet_name="FMA_UE",
)

data.columns = data.columns.str.strip()

required_columns = {"Group"}

for outcome in OUTCOMES:
    required_columns.add(outcome["baseline"])
    required_columns.add(outcome["post"])

missing_columns = required_columns.difference(data.columns)

if missing_columns:
    raise KeyError(
        "The following required columns are missing from the FMA_UE sheet:\n"
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

for outcome in OUTCOMES:
    data[outcome["baseline"]] = pd.to_numeric(
        data[outcome["baseline"]],
        errors="coerce",
    )

    data[outcome["post"]] = pd.to_numeric(
        data[outcome["post"]],
        errors="coerce",
    )


# ============================================================
# Calculate statistics
# ============================================================
results: dict[str, dict[str, dict[str, Any]]] = {}

for outcome in OUTCOMES:
    domain = outcome["domain"]
    baseline_column = outcome["baseline"]
    post_column = outcome["post"]

    results[domain] = {}

    for group_index, group in enumerate(GROUP_ORDER):
        group_data = data.loc[data["Group"] == group]

        test_results = wilcoxon_with_effect(
            group_data[baseline_column],
            group_data[post_column],
        )

        change_results = paired_change_ci(
            group_data[baseline_column],
            group_data[post_column],
            seed=RANDOM_SEED + group_index,
        )

        results[domain][group] = {
            **test_results,
            **change_results,
            "p_fdr": np.nan,
        }


# ============================================================
# Apply FDR correction to exploratory domains
# ============================================================
for group in GROUP_ORDER:
    raw_p_values = [
        results[domain][group]["p_raw"]
        for domain in EXPLORATORY_DOMAINS
    ]

    adjusted_p_values = multipletests(
        raw_p_values,
        alpha=0.05,
        method="fdr_bh",
    )[1]

    for domain, adjusted_p_value in zip(
        EXPLORATORY_DOMAINS,
        adjusted_p_values,
    ):
        results[domain][group]["p_fdr"] = float(
            adjusted_p_value
        )


# ============================================================
# Print statistical results
# ============================================================
print("\nFigure 6: FMA-UE Within-Group Results")
print("Comparison direction: Post-intervention minus Baseline")
print("=" * 60)

for outcome in OUTCOMES:
    domain = outcome["domain"]

    for group in GROUP_ORDER:
        result = results[domain][group]

        print(f"\n{domain} — {group}")
        print("-" * 40)
        print(f"Complete pairs, n = {result['n']}")
        print(
            "Nonzero differences used by Wilcoxon test, "
            f"n = {result['n_nonzero']}"
        )
        print(f"Wilcoxon W = {result['W']:.2f}")
        print(f"Raw p-value = {result['p_raw']:.3f}")

        if domain in EXPLORATORY_DOMAINS:
            print(
                "FDR-adjusted p-value = "
                f"{result['p_fdr']:.3f}"
            )

        print(f"Approximate Z = {result['Z']:.3f}")
        print(f"Effect size r = {result['r']:.2f}")
        print(
            "Median paired change = "
            f"{result['median_change']:.2f}"
        )
        print(
            "Bootstrap 95% CI = "
            f"[{result['ci_low']:.2f}, "
            f"{result['ci_high']:.2f}]"
        )


# ============================================================
# Reshape data for plotting
# ============================================================
plotting_frames = {}

for outcome in OUTCOMES:
    domain = outcome["domain"]

    baseline_data = data[
        ["Group", outcome["baseline"]]
    ].rename(
        columns={outcome["baseline"]: "Score"}
    )
    baseline_data["Time"] = "Baseline"

    post_data = data[
        ["Group", outcome["post"]]
    ].rename(
        columns={outcome["post"]: "Score"}
    )
    post_data["Time"] = "Post"

    plotting_data = pd.concat(
        [baseline_data, post_data],
        ignore_index=True,
    ).dropna(
        subset=["Group", "Time", "Score"]
    )

    plotting_frames[domain] = plotting_data


# ============================================================
# Figure colors
# ============================================================
TREATMENT_BASELINE = "#5DADE2"
TREATMENT_POST = "#AED6F1"
CONTROL_BASELINE = "#EC7063"
CONTROL_POST = "#F5B7B1"

BOX_COLORS = [
    TREATMENT_BASELINE,
    CONTROL_BASELINE,
    TREATMENT_POST,
    CONTROL_POST,
]

LEGEND_HANDLES = [
    mpl.patches.Patch(
        facecolor=TREATMENT_BASELINE,
        edgecolor="black",
        label="Treatment Baseline",
    ),
    mpl.patches.Patch(
        facecolor=TREATMENT_POST,
        edgecolor="black",
        label="Treatment Post",
    ),
    mpl.patches.Patch(
        facecolor=CONTROL_BASELINE,
        edgecolor="black",
        label="Control Baseline",
    ),
    mpl.patches.Patch(
        facecolor=CONTROL_POST,
        edgecolor="black",
        label="Control Post",
    ),
]


# ============================================================
# Generate Figure 6
# ============================================================
figure, axes = plt.subplots(
    2,
    2,
    figsize=(8.7, 5.6),
)

axes = axes.flatten()

for axis, outcome in zip(axes, OUTCOMES):
    domain = outcome["domain"]
    panel_letter = outcome["panel"]

    plotting_data = plotting_frames[domain]

    sns.boxplot(
        data=plotting_data,
        x="Group",
        y="Score",
        hue="Time",
        order=GROUP_ORDER,
        hue_order=TIME_ORDER,
        palette={
            "Baseline": "#FFFFFF",
            "Post": "#FFFFFF",
        },
        width=0.6,
        fliersize=0,
        linewidth=LINE_WIDTH,
        ax=axis,
    )

    for patch, color in zip(
        axis.patches[:4],
        BOX_COLORS,
    ):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(1.0)
        patch.set_linewidth(LINE_WIDTH)

    sns.stripplot(
        data=plotting_data,
        x="Group",
        y="Score",
        hue="Time",
        order=GROUP_ORDER,
        hue_order=TIME_ORDER,
        dodge=True,
        color="black",
        size=5,
        jitter=0.18,
        alpha=0.85,
        zorder=10,
        ax=axis,
    )

    panel_legend = axis.get_legend()

    if panel_legend is not None:
        panel_legend.remove()

    axis.set_title(
        domain,
        fontweight="normal",
        pad=16,
    )

    format_axis(axis)

    axis.text(
        -0.10,
        1.06,
        panel_letter,
        transform=axis.transAxes,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
    )

    y_minimum = plotting_data["Score"].min()
    y_maximum = plotting_data["Score"].max()
    y_range = y_maximum - y_minimum

    if y_range == 0:
        y_range = 1.0

    axis.set_ylim(
        y_minimum - 0.05 * y_range,
        y_maximum + 0.60 * y_range,
    )

    statistics_positions = {
        "Treatment": 0.30,
        "Control": 0.72,
    }

    for group in GROUP_ORDER:
        result = results[domain][group]

        if domain == "Motor":
            statistics_text = (
                f"p {format_p_value(result['p_raw'])}\n"
                f"r = {result['r']:.2f}"
            )
        else:
            statistics_text = (
                f"p {format_p_value(result['p_raw'])}\n"
                f"$p_{{adj}}$ "
                f"{format_p_value(result['p_fdr'])}\n"
                f"r = {result['r']:.2f}"
            )

        axis.text(
            statistics_positions[group],
            0.86,
            statistics_text,
            transform=axis.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=STATS_SIZE,
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "boxstyle": "round,pad=0.25",
                "linewidth": 0.8,
                "alpha": 0.95,
            },
        )


# ============================================================
# Add shared legend
# ============================================================
figure.legend(
    handles=LEGEND_HANDLES,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=True,
    fontsize=LEGEND_SIZE,
    handlelength=1.4,
    handletextpad=0.4,
    borderpad=0.3,
)

figure.tight_layout(
    rect=[0.04, 0.06, 0.98, 0.95],
    h_pad=2.3,
    w_pad=2.3,
)


# ============================================================
# Save Figure 6
# ============================================================
output_files = {
    "TIFF": OUTPUT_DIRECTORY / "Fig6.tif",
    "EPS": OUTPUT_DIRECTORY / "Fig6.eps",
    "SVG": OUTPUT_DIRECTORY / "Fig6.svg",
    "PDF": OUTPUT_DIRECTORY / "Fig6.pdf",
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

print("\nFigure 6 files saved successfully:")

for file_format, file_path in output_files.items():
    print(f"{file_format}: {file_path}")

plt.show()