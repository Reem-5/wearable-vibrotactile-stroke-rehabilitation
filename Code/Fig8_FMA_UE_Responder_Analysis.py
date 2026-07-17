"""
Figure 8: FMA-UE Responder Analysis

This script:
1. Loads the de-identified clinical trial dataset.
2. Calculates FMA-UE change scores as post-intervention minus baseline.
3. Classifies participants as responders using prespecified thresholds.
4. Compares responder proportions between treatment and control groups
   using two-sided Fisher's exact tests.
5. Applies Benjamini-Hochberg false discovery rate correction across the
   three exploratory FMA-UE domains.
6. Generates Figure 8 and saves it in TIFF, EPS, SVG, PDF, and PNG formats.

Expected repository structure:

wearable-vibrotactile-stroke-rehabilitation/
│
├── Code/
│   └── Fig8_FMA_UE_Responder_Analysis.py
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
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


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
# Dataset settings
# ============================================================
SHEET_NAME = "FMA_UE"

GROUP_ORDER = ["Treatment", "Control"]

DOMAINS = {
    "Motor": {
        "baseline": "Baseline_motor Function",
        "post": "Post_motor Function",
        "threshold": 5.0,
        "panel": "A",
    },
    "Sensation": {
        "baseline": "Baseline_Sensation",
        "post": "Post_Sensation",
        "threshold": 1.0,
        "panel": "B",
    },
    "Joint Motion": {
        "baseline": "Baseline_Joint motion",
        "post": "Post_Joint motion",
        "threshold": 1.0,
        "panel": "C",
    },
    "Joint Pain": {
        "baseline": "Baseline_Joint Pain",
        "post": "Post_Joint Pain",
        "threshold": 1.0,
        "panel": "D",
    },
}

EXPLORATORY_DOMAINS = [
    "Sensation",
    "Joint Motion",
    "Joint Pain",
]


# ============================================================
# Global figure formatting
# ============================================================
FONT_FAMILY = "Arial"

TITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 13
STATS_SIZE = 13
PANEL_LABEL_SIZE = 22
LINE_WIDTH = 1.2

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


# ============================================================
# Helper functions
# ============================================================
def format_p_value(p_value: float) -> str:
    """Format p-values for figure presentation."""
    if p_value < 0.001:
        return "< 0.001"

    return f"= {p_value:.3f}"


def format_axis(axis: plt.Axes) -> None:
    """Apply consistent formatting to one figure panel."""
    axis.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
    )

    axis.grid(False)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for spine in axis.spines.values():
        spine.set_linewidth(LINE_WIDTH)


def calculate_responder_statistics(
    data: pd.DataFrame,
    baseline_column: str,
    post_column: str,
    threshold: float,
) -> dict[str, Any]:
    """
    Calculate responder proportions and perform Fisher's exact test.

    A responder is defined as a participant whose change score is greater
    than or equal to the prespecified threshold.

    Change score is calculated as:

        post-intervention score - baseline score
    """
    domain_data = pd.DataFrame(
        {
            "Group": data["Group"],
            "Baseline": pd.to_numeric(
                data[baseline_column],
                errors="coerce",
            ),
            "Post": pd.to_numeric(
                data[post_column],
                errors="coerce",
            ),
        }
    ).dropna(
        subset=["Group", "Baseline", "Post"]
    )

    domain_data["Change"] = (
        domain_data["Post"]
        - domain_data["Baseline"]
    )

    domain_data["Responder"] = (
        domain_data["Change"] >= threshold
    )

    counts: dict[str, dict[str, int]] = {}
    proportions: dict[str, float] = {}

    for group in GROUP_ORDER:
        group_data = domain_data.loc[
            domain_data["Group"] == group
        ]

        total = len(group_data)
        responders = int(
            group_data["Responder"].sum()
        )
        nonresponders = total - responders

        counts[group] = {
            "responders": responders,
            "nonresponders": nonresponders,
            "total": total,
        }

        proportions[group] = (
            responders / total
            if total > 0
            else np.nan
        )

    if any(
        counts[group]["total"] == 0
        for group in GROUP_ORDER
    ):
        raise ValueError(
            "Responder analysis requires valid observations in both groups."
        )

    contingency_table = [
        [
            counts["Treatment"]["responders"],
            counts["Treatment"]["nonresponders"],
        ],
        [
            counts["Control"]["responders"],
            counts["Control"]["nonresponders"],
        ],
    ]

    odds_ratio, fisher_p_value = fisher_exact(
        contingency_table,
        alternative="two-sided",
    )

    return {
        "counts": counts,
        "proportions": proportions,
        "threshold": threshold,
        "odds_ratio": float(odds_ratio),
        "fisher_p": float(fisher_p_value),
        "p_adj": np.nan,
    }


# ============================================================
# Load and validate data
# ============================================================
data = pd.read_excel(
    DATA_FILE,
    sheet_name=SHEET_NAME,
)

data.columns = data.columns.str.strip()

required_columns = {"Group"}

for domain_settings in DOMAINS.values():
    required_columns.add(
        domain_settings["baseline"]
    )
    required_columns.add(
        domain_settings["post"]
    )

missing_columns = required_columns.difference(
    data.columns
)

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


# ============================================================
# Compute responder statistics
# ============================================================
results: dict[str, dict[str, Any]] = {}

for domain_name, domain_settings in DOMAINS.items():
    results[domain_name] = (
        calculate_responder_statistics(
            data=data,
            baseline_column=domain_settings["baseline"],
            post_column=domain_settings["post"],
            threshold=domain_settings["threshold"],
        )
    )


# ============================================================
# Apply FDR correction to exploratory domains
# ============================================================
raw_exploratory_p_values = [
    results[domain]["fisher_p"]
    for domain in EXPLORATORY_DOMAINS
]

adjusted_p_values = multipletests(
    raw_exploratory_p_values,
    alpha=0.05,
    method="fdr_bh",
)[1]

for domain_name, adjusted_p_value in zip(
    EXPLORATORY_DOMAINS,
    adjusted_p_values,
):
    results[domain_name]["p_adj"] = float(
        adjusted_p_value
    )


# ============================================================
# Print results
# ============================================================
print("\nFigure 8: FMA-UE Responder Analysis")
print("=" * 55)

for domain_name, result in results.items():
    print(
        f"\n{domain_name} responders: "
        f"change ≥ {result['threshold']:.2f}"
    )
    print("-" * 45)

    for group in GROUP_ORDER:
        group_counts = result["counts"][group]
        proportion = result["proportions"][group]

        print(
            f"{group}: "
            f"{group_counts['responders']}/"
            f"{group_counts['total']} "
            f"({100 * proportion:.1f}%)"
        )

    print(
        f"Fisher's exact test p-value = "
        f"{result['fisher_p']:.3f}"
    )

    if domain_name in EXPLORATORY_DOMAINS:
        print(
            f"FDR-adjusted p-value = "
            f"{result['p_adj']:.3f}"
        )

    print(
        f"Odds ratio = "
        f"{result['odds_ratio']:.3f}"
    )


# ============================================================
# Figure settings
# ============================================================
BAR_COLORS = {
    "Treatment": "#5DADE2",
    "Control": "#EC7063",
}


# ============================================================
# Create Figure 8
# ============================================================
figure, axes = plt.subplots(
    2,
    2,
    figsize=(10.5, 8.2),
    sharey=True,
)

axes = axes.flatten()

for axis, (domain_name, domain_settings) in zip(
    axes,
    DOMAINS.items(),
):
    result = results[domain_name]

    x_positions = np.arange(
        len(GROUP_ORDER)
    )

    bar_heights = [
        result["proportions"][group]
        for group in GROUP_ORDER
    ]

    axis.bar(
        x_positions,
        bar_heights,
        color=[
            BAR_COLORS[group]
            for group in GROUP_ORDER
        ],
        edgecolor="black",
        linewidth=LINE_WIDTH,
        width=0.65,
    )

    for x_position, height, group in zip(
        x_positions,
        bar_heights,
        GROUP_ORDER,
    ):
        group_counts = result["counts"][group]

        label = (
            f"{100 * height:.1f}%\n"
            f"({group_counts['responders']}/"
            f"{group_counts['total']})"
        )

        axis.text(
            x_position,
            height + 0.025,
            label,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=STATS_SIZE,
        )

    threshold = result["threshold"]

    if domain_name == "Motor":
        title_text = (
            f"{domain_name} Responders "
            f"(Δ ≥ {threshold:.2f})\n"
            f"p {format_p_value(result['fisher_p'])}"
        )
    else:
        title_text = (
            f"{domain_name} Responders "
            f"(Δ ≥ {threshold:.2f})\n"
            f"p {format_p_value(result['fisher_p'])}; "
            f"$p_{{adj}}$ "
            f"{format_p_value(result['p_adj'])}"
        )

    axis.set_title(
        title_text,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        pad=8,
    )

    axis.text(
        -0.08,
        1.04,
        domain_settings["panel"],
        transform=axis.transAxes,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        GROUP_ORDER,
        fontsize=TICK_LABEL_SIZE,
    )

    axis.set_ylim(
        0,
        1.12,
    )

    axis.set_ylabel(
        "Proportion of responders",
        fontsize=AXIS_LABEL_SIZE,
    )

    format_axis(axis)


figure.tight_layout(
    rect=[0.04, 0.04, 1.0, 0.98]
)


# ============================================================
# Save Figure 8
# ============================================================
output_files = {
    "TIFF": OUTPUT_DIRECTORY / "Fig8.tif",
    "EPS": OUTPUT_DIRECTORY / "Fig8.eps",
    "SVG": OUTPUT_DIRECTORY / "Fig8.svg",
    "PDF": OUTPUT_DIRECTORY / "Fig8.pdf",
    "PNG": OUTPUT_DIRECTORY / "Fig8.png",
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

print("\nFigure 8 files saved successfully:")

for file_format, file_path in output_files.items():
    print(f"{file_format}: {file_path}")

plt.show()
