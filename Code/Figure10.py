"""
Figure 10: Modified Ashworth Scale Responder Analysis

This script:
1. Loads the de-identified clinical trial dataset.
2. Calculates MAS scores for the fingers, wrist, and elbow.
3. Defines a responder as a participant with at least a 1-point reduction
   in the corresponding MAS domain score.
4. Compares responder proportions between treatment and control groups
   using two-sided Fisher's exact tests.
5. Applies Benjamini-Hochberg false discovery rate correction across
   the three MAS responder domains.
6. Generates Figure 10 and saves it in TIFF, EPS, SVG, PDF, and PNG formats.

Expected repository structure:

wearable-vibrotactile-stroke-rehabilitation/
│
├── Code/
│   └── Fig10_MAS_Responder_Analysis.py
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
SHEET_NAME = "MAS"

GROUP_ORDER = ["Treatment", "Control"]
RESPONDER_THRESHOLD = 1.0

DOMAIN_SETTINGS = {
    "Fingers": {
        "baseline_flexor": "Baseline_Finger_Flexors",
        "baseline_extensor": "Baseline_Finger_Extensors",
        "post_flexor": "Post_Finger_Flexors",
        "post_extensor": "Post_Finger_Extensors",
        "panel": "A",
    },
    "Wrist": {
        "baseline_flexor": "Baseline_Wrist_Flexors",
        "baseline_extensor": "Baseline_Wrist_Extensors",
        "post_flexor": "Post_Wrist_Flexors",
        "post_extensor": "Post_Wrist_Extensors",
        "panel": "B",
    },
    "Elbow": {
        "baseline_flexor": "Baseline_Elbow_Flexors",
        "baseline_extensor": "Baseline_Elbow_Extensors",
        "post_flexor": "Post_Elbow_Flexors",
        "post_extensor": "Post_Elbow_Extensors",
        "panel": "C",
    },
}


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
# Column-name standardization
# ============================================================
COLUMN_RENAME_MAP = {
    "Baseline_Finger Flexors": "Baseline_Finger_Flexors",
    "Post_Finger Flexors": "Post_Finger_Flexors",
    "Baseline_Finger Extensors": "Baseline_Finger_Extensors",
    "Post_Finger Extensors": "Post_Finger_Extensors",

    "Baseline_Wrist Flexors": "Baseline_Wrist_Flexors",
    "Post_Wrist Flexors": "Post_Wrist_Flexors",
    "Baseline_wrist Extensors": "Baseline_Wrist_Extensors",
    "Post_wrist Extensors": "Post_Wrist_Extensors",

    "Baseline_Elbow Flexors": "Baseline_Elbow_Flexors",
    "Post_Elbow Flexors": "Post_Elbow_Flexors",
    "Baseline_Elbow Extensors": "Baseline_Elbow_Extensors",
    "Post_Elbow Extensors": "Post_Elbow_Extensors",
}


# ============================================================
# Helper functions
# ============================================================
def format_p_value(p_value: float) -> str:
    """Format p-values for figure display."""
    if p_value < 0.001:
        return "< 0.001"

    return f"= {p_value:.3f}"


def format_axis(axis: plt.Axes) -> None:
    """Apply consistent formatting to one panel."""
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
    domain_name: str,
) -> dict[str, Any]:
    """
    Calculate MAS responder counts and Fisher's exact test.

    A responder is defined as a participant whose domain score decreases
    by at least RESPONDER_THRESHOLD points:

        baseline domain score - post-intervention domain score
        >= RESPONDER_THRESHOLD

    A positive reduction therefore represents improvement.
    """
    settings = DOMAIN_SETTINGS[domain_name]

    baseline_values = data[
        [
            settings["baseline_flexor"],
            settings["baseline_extensor"],
        ]
    ].mean(
        axis=1,
        skipna=False,
    )

    post_values = data[
        [
            settings["post_flexor"],
            settings["post_extensor"],
        ]
    ].mean(
        axis=1,
        skipna=False,
    )

    domain_data = pd.DataFrame(
        {
            "Group": data["Group"],
            "Baseline": baseline_values,
            "Post": post_values,
        }
    ).dropna(
        subset=["Group", "Baseline", "Post"]
    )

    domain_data["Reduction"] = (
        domain_data["Baseline"]
        - domain_data["Post"]
    )

    domain_data["Responder"] = (
        domain_data["Reduction"]
        >= RESPONDER_THRESHOLD
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
            f"Responder analysis for {domain_name} requires valid "
            "observations in both study groups."
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

    odds_ratio, p_value = fisher_exact(
        contingency_table,
        alternative="two-sided",
    )

    return {
        "counts": counts,
        "proportions": proportions,
        "odds_ratio": float(odds_ratio),
        "p_raw": float(p_value),
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
data = data.rename(columns=COLUMN_RENAME_MAP)

required_columns = {"Group"}

for settings in DOMAIN_SETTINGS.values():
    required_columns.update(
        {
            settings["baseline_flexor"],
            settings["baseline_extensor"],
            settings["post_flexor"],
            settings["post_extensor"],
        }
    )

missing_columns = required_columns.difference(
    data.columns
)

if missing_columns:
    raise KeyError(
        "The following required columns are missing from "
        f"the {SHEET_NAME} worksheet:\n"
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

numeric_columns = sorted(
    {
        column
        for settings in DOMAIN_SETTINGS.values()
        for column in (
            settings["baseline_flexor"],
            settings["baseline_extensor"],
            settings["post_flexor"],
            settings["post_extensor"],
        )
    }
)

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )


# ============================================================
# Calculate responder statistics
# ============================================================
results: dict[str, dict[str, Any]] = {}

for domain_name in DOMAIN_SETTINGS:
    results[domain_name] = (
        calculate_responder_statistics(
            data=data,
            domain_name=domain_name,
        )
    )


# ============================================================
# Apply FDR correction across the three MAS domains
# ============================================================
raw_p_values = [
    results[domain]["p_raw"]
    for domain in DOMAIN_SETTINGS
]

adjusted_p_values = multipletests(
    raw_p_values,
    alpha=0.05,
    method="fdr_bh",
)[1]

for domain_name, adjusted_p_value in zip(
    DOMAIN_SETTINGS,
    adjusted_p_values,
):
    results[domain_name]["p_adj"] = float(
        adjusted_p_value
    )


# ============================================================
# Print statistical results
# ============================================================
print("\nFigure 10: MAS Responder Analysis")
print(
    "Responder definition: at least a "
    f"{RESPONDER_THRESHOLD:.1f}-point MAS reduction"
)
print("=" * 60)

for domain_name, result in results.items():
    print(f"\n{domain_name}")
    print("-" * 40)

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
        f"{result['p_raw']:.3f}"
    )

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
# Generate Figure 10
# ============================================================
figure, axes = plt.subplots(
    1,
    3,
    figsize=(10.5, 3.8),
    sharey=True,
)

for axis, (domain_name, settings) in zip(
    axes,
    DOMAIN_SETTINGS.items(),
):
    result = results[domain_name]

    x_positions = np.arange(
        len(GROUP_ORDER)
    )

    bar_heights = [
        result["proportions"][group]
        for group in GROUP_ORDER
    ]

    bars = axis.bar(
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

    for bar, height, group in zip(
        bars,
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
            bar.get_x() + bar.get_width() / 2,
            height + 0.03,
            label,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=STATS_SIZE,
        )

    axis.set_ylim(
        0,
        1.12,
    )

    title_text = (
        f"{domain_name} MAS Responders\n"
        f"p {format_p_value(result['p_raw'])}; "
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
        -0.12,
        1.05,
        settings["panel"],
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="left",
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        GROUP_ORDER,
        fontsize=TICK_LABEL_SIZE,
    )

    if domain_name == "Fingers":
        axis.set_ylabel(
            "Proportion of responders\n"
            "(≥1-point MAS reduction)",
            fontsize=AXIS_LABEL_SIZE,
        )

    format_axis(axis)


figure.tight_layout(
    rect=[0.04, 0.04, 1.0, 0.98]
)


# ============================================================
# Save Figure 10
# ============================================================
output_files = {
    "TIFF": OUTPUT_DIRECTORY / "Fig10.tif",
    "EPS": OUTPUT_DIRECTORY / "Fig10.eps",
    "SVG": OUTPUT_DIRECTORY / "Fig10.svg",
    "PDF": OUTPUT_DIRECTORY / "Fig10.pdf",
    "PNG": OUTPUT_DIRECTORY / "Fig10.png",
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

print("\nFigure 10 files saved successfully:")

for file_format, file_path in output_files.items():
    print(f"{file_format}: {file_path}")

plt.show()