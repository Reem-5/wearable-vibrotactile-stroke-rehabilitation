"""
Supplementary Figure S1: Individual FMA-UE Change Scores

This script:
1. Loads the de-identified clinical trial dataset.
2. Calculates participant-level FMA-UE change scores as
   post-intervention minus baseline.
3. Performs within-group Wilcoxon signed-rank tests.
4. Performs between-group Mann–Whitney U tests.
5. Calculates rank-based effect sizes for between-group comparisons.
6. Applies Benjamini–Hochberg false discovery rate correction to the
   secondary FMA-UE domains:
       - Sensation
       - Joint Motion
       - Joint Pain
7. Generates Supplementary Figure S1.
8. Saves the figure in TIFF, EPS, SVG, PDF, and PNG formats.

Motor is treated as the primary outcome and is reported without
multiplicity correction.

Expected repository structure:

wearable-vibrotactile-stroke-rehabilitation/
│
├── Code/
│   └── FigS1_FMA_UE_Individual_Change_Scores.py
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
from scipy.stats import mannwhitneyu, wilcoxon
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

GROUP_ORDER = [
    "Treatment",
    "Control",
]

DOMAIN_SETTINGS = {
    "Motor": {
        "baseline": "Baseline_motor Function",
        "post": "Post_motor Function",
        "change": "Change_Motor_FMA",
    },
    "Sensation": {
        "baseline": "Baseline_Sensation",
        "post": "Post_Sensation",
        "change": "Change_Sensation_FMA",
    },
    "Joint Motion": {
        "baseline": "Baseline_Joint motion",
        "post": "Post_Joint motion",
        "change": "Change_Joint_Motion_FMA",
    },
    "Joint Pain": {
        "baseline": "Baseline_Joint Pain",
        "post": "Post_Joint Pain",
        "change": "Change_Joint_Pain_FMA",
    },
}

DOMAIN_ORDER = [
    "Motor",
    "Sensation",
    "Joint Motion",
    "Joint Pain",
]

SECONDARY_DOMAINS = [
    "Sensation",
    "Joint Motion",
    "Joint Pain",
]


# ============================================================
# Global figure formatting
# ============================================================
FONT_FAMILY = "Arial"

TITLE_SIZE = 14
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 13
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
    """Format a p-value for figure display."""
    if pd.isna(p_value):
        return "NA"

    if p_value < 0.001:
        return "< 0.001"

    return f"= {p_value:.3f}"


def safe_wilcoxon(
    values: np.ndarray,
) -> dict[str, float | int]:
    """
    Perform a two-sided Wilcoxon signed-rank test on change scores.

    Testing change scores against zero is equivalent to testing whether
    paired post-intervention and baseline scores differ.

    All-zero change-score vectors are handled explicitly.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    nonzero_values = values[
        values != 0
    ]

    if len(values) == 0:
        return {
            "W": np.nan,
            "p": np.nan,
            "n": 0,
            "n_nonzero": 0,
        }

    if len(nonzero_values) == 0:
        return {
            "W": 0.0,
            "p": 1.0,
            "n": len(values),
            "n_nonzero": 0,
        }

    test_result = wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
    )

    return {
        "W": float(test_result.statistic),
        "p": float(test_result.pvalue),
        "n": len(values),
        "n_nonzero": len(nonzero_values),
    }


def mannwhitney_with_effect(
    treatment: np.ndarray,
    control: np.ndarray,
) -> dict[str, float | int]:
    """
    Perform a two-sided Mann–Whitney U test.

    The reported effect size is:

        r = |Z| / sqrt(n_treatment + n_control)

    Z is calculated using the large-sample Mann–Whitney variance.
    """
    treatment = np.asarray(
        treatment,
        dtype=float,
    )

    control = np.asarray(
        control,
        dtype=float,
    )

    treatment = treatment[
        np.isfinite(treatment)
    ]

    control = control[
        np.isfinite(control)
    ]

    if len(treatment) == 0 or len(control) == 0:
        return {
            "U": np.nan,
            "p": np.nan,
            "Z": np.nan,
            "r": np.nan,
            "n_treatment": len(treatment),
            "n_control": len(control),
        }

    test_result = mannwhitneyu(
        treatment,
        control,
        alternative="two-sided",
    )

    u_statistic = float(
        test_result.statistic
    )

    p_value = float(
        test_result.pvalue
    )

    n_treatment = len(treatment)
    n_control = len(control)

    expected_u = (
        n_treatment
        * n_control
        / 2
    )

    standard_deviation_u = sqrt(
        n_treatment
        * n_control
        * (n_treatment + n_control + 1)
        / 12
    )

    if standard_deviation_u > 0:
        z_value = (
            u_statistic - expected_u
        ) / standard_deviation_u
    else:
        z_value = np.nan

    if np.isfinite(z_value):
        effect_size = abs(
            z_value
        ) / sqrt(
            n_treatment + n_control
        )
    else:
        effect_size = np.nan

    return {
        "U": u_statistic,
        "p": p_value,
        "Z": z_value,
        "r": effect_size,
        "n_treatment": n_treatment,
        "n_control": n_control,
    }


def apply_fdr_correction(
    results: dict[str, dict[str, Any]],
    p_value_key: str,
    adjusted_key: str,
) -> None:
    """
    Apply Benjamini–Hochberg correction across secondary domains.

    Results are modified in place.
    """
    valid_domains = []
    raw_p_values = []

    for domain in SECONDARY_DOMAINS:
        p_value = results[domain][p_value_key]

        if pd.notna(p_value):
            valid_domains.append(domain)
            raw_p_values.append(p_value)
        else:
            results[domain][adjusted_key] = np.nan

    if not raw_p_values:
        return

    adjusted_p_values = multipletests(
        raw_p_values,
        alpha=0.05,
        method="fdr_bh",
    )[1]

    for domain, adjusted_p_value in zip(
        valid_domains,
        adjusted_p_values,
    ):
        results[domain][adjusted_key] = float(
            adjusted_p_value
        )


def format_axis(
    axis: plt.Axes,
) -> None:
    """Apply consistent axis formatting."""
    axis.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
    )

    axis.grid(False)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for spine in axis.spines.values():
        spine.set_linewidth(
            LINE_WIDTH
        )


def add_y_padding(
    axis: plt.Axes,
    lower_padding: float = 0.14,
    upper_padding: float = 0.20,
) -> None:
    """Add vertical padding around plotted change scores."""
    y_minimum, y_maximum = axis.get_ylim()
    y_range = y_maximum - y_minimum

    if y_range == 0:
        y_range = 1.0

    axis.set_ylim(
        y_minimum - lower_padding * y_range,
        y_maximum + upper_padding * y_range,
    )


def add_individual_change_panel(
    axis: plt.Axes,
    participant_labels: list[str],
    change_values: np.ndarray,
    color: str,
) -> None:
    """Draw one participant-level lollipop plot."""
    x_positions = np.arange(
        len(change_values)
    )

    for x_position, change_value in zip(
        x_positions,
        change_values,
    ):
        axis.vlines(
            x=x_position,
            ymin=0,
            ymax=change_value,
            color=color,
            linewidth=2,
            clip_on=False,
        )

        axis.scatter(
            x_position,
            change_value,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            s=45,
            zorder=3,
            clip_on=False,
        )

    axis.axhline(
        y=0,
        color="black",
        linewidth=LINE_WIDTH,
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        participant_labels,
        rotation=45,
        horizontalalignment="right",
        fontsize=TICK_LABEL_SIZE,
    )

    axis.margins(
        x=0.08
    )

    add_y_padding(
        axis
    )

    format_axis(
        axis
    )


# ============================================================
# Load and validate data
# ============================================================
data = pd.read_excel(
    DATA_FILE,
    sheet_name=SHEET_NAME,
)

data.columns = data.columns.str.strip()

required_columns = {
    "Group",
}

for settings in DOMAIN_SETTINGS.values():
    required_columns.add(
        settings["baseline"]
    )
    required_columns.add(
        settings["post"]
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
        + "\n\nAvailable columns are:\n"
        + "\n".join(
            str(column)
            for column in data.columns
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

data["Group"] = pd.Categorical(
    data["Group"],
    categories=GROUP_ORDER,
    ordered=True,
)


# ============================================================
# Convert outcome columns to numeric
# ============================================================
numeric_columns = sorted(
    {
        column
        for settings in DOMAIN_SETTINGS.values()
        for column in (
            settings["baseline"],
            settings["post"],
        )
    }
)

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )


# ============================================================
# Calculate FMA-UE change scores
# ============================================================
for domain_name, settings in DOMAIN_SETTINGS.items():
    data[settings["change"]] = (
        data[settings["post"]]
        - data[settings["baseline"]]
    )


# ============================================================
# Compute statistical analyses
# ============================================================
results: dict[str, dict[str, Any]] = {}

for domain_name in DOMAIN_ORDER:
    settings = DOMAIN_SETTINGS[
        domain_name
    ]

    change_column = settings[
        "change"
    ]

    treatment_change = (
        data.loc[
            data["Group"] == "Treatment",
            change_column,
        ]
        .dropna()
        .to_numpy(dtype=float)
    )

    control_change = (
        data.loc[
            data["Group"] == "Control",
            change_column,
        ]
        .dropna()
        .to_numpy(dtype=float)
    )

    treatment_wilcoxon = safe_wilcoxon(
        treatment_change
    )

    control_wilcoxon = safe_wilcoxon(
        control_change
    )

    between_group_test = mannwhitney_with_effect(
        treatment_change,
        control_change,
    )

    results[domain_name] = {
        "n_treatment": len(
            treatment_change
        ),
        "n_control": len(
            control_change
        ),
        "W_treatment": treatment_wilcoxon["W"],
        "p_treatment": treatment_wilcoxon["p"],
        "n_nonzero_treatment": treatment_wilcoxon[
            "n_nonzero"
        ],
        "p_treatment_adj": np.nan,
        "W_control": control_wilcoxon["W"],
        "p_control": control_wilcoxon["p"],
        "n_nonzero_control": control_wilcoxon[
            "n_nonzero"
        ],
        "p_control_adj": np.nan,
        "U": between_group_test["U"],
        "Z": between_group_test["Z"],
        "p_between": between_group_test["p"],
        "p_between_adj": np.nan,
        "r": between_group_test["r"],
    }


# ============================================================
# Apply FDR corrections to secondary domains
# ============================================================
apply_fdr_correction(
    results=results,
    p_value_key="p_treatment",
    adjusted_key="p_treatment_adj",
)

apply_fdr_correction(
    results=results,
    p_value_key="p_control",
    adjusted_key="p_control_adj",
)

apply_fdr_correction(
    results=results,
    p_value_key="p_between",
    adjusted_key="p_between_adj",
)


# ============================================================
# Print statistical results
# ============================================================
print(
    "\nSupplementary Figure S1: "
    "FMA-UE Individual Change Statistics"
)
print(
    "Change direction: Post-intervention minus Baseline"
)
print("=" * 72)

for domain_name in DOMAIN_ORDER:
    result = results[
        domain_name
    ]

    print(
        f"\n{domain_name}"
    )
    print(
        "-" * 48
    )

    print(
        "Treatment: "
        f"n = {result['n_treatment']}, "
        f"nonzero pairs = {result['n_nonzero_treatment']}, "
        f"W = {result['W_treatment']:.2f}, "
        f"p = {result['p_treatment']:.3f}"
    )

    if domain_name in SECONDARY_DOMAINS:
        print(
            "Treatment FDR-adjusted p-value = "
            f"{result['p_treatment_adj']:.3f}"
        )

    print(
        "Control: "
        f"n = {result['n_control']}, "
        f"nonzero pairs = {result['n_nonzero_control']}, "
        f"W = {result['W_control']:.2f}, "
        f"p = {result['p_control']:.3f}"
    )

    if domain_name in SECONDARY_DOMAINS:
        print(
            "Control FDR-adjusted p-value = "
            f"{result['p_control_adj']:.3f}"
        )

    print(
        "Between groups: "
        f"U = {result['U']:.2f}, "
        f"Z = {result['Z']:.2f}, "
        f"p = {result['p_between']:.3f}, "
        f"r = {result['r']:.2f}"
    )

    if domain_name in SECONDARY_DOMAINS:
        print(
            "Between-group FDR-adjusted p-value = "
            f"{result['p_between_adj']:.3f}"
        )


# ============================================================
# Figure settings
# ============================================================
TREATMENT_COLOR = "#5DADE2"
CONTROL_COLOR = "#EC7063"

PANEL_LABELS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
]


# ============================================================
# Create Supplementary Figure S1
# ============================================================
figure, axes = plt.subplots(
    nrows=4,
    ncols=2,
    figsize=(14, 16),
    sharey=False,
)

for row_index, domain_name in enumerate(
    DOMAIN_ORDER
):
    settings = DOMAIN_SETTINGS[
        domain_name
    ]

    change_column = settings[
        "change"
    ]

    domain_data = data.loc[
        data[change_column].notna()
    ].copy()

    treatment_data = domain_data.loc[
        domain_data["Group"] == "Treatment"
    ].copy()

    control_data = domain_data.loc[
        domain_data["Group"] == "Control"
    ].copy()

    treatment_data = treatment_data.reset_index(
        drop=True
    )

    control_data = control_data.reset_index(
        drop=True
    )

    treatment_labels = [
        f"T{index}"
        for index in range(
            1,
            len(treatment_data) + 1,
        )
    ]

    control_labels = [
        f"C{index}"
        for index in range(
            1,
            len(control_data) + 1,
        )
    ]

    treatment_values = treatment_data[
        change_column
    ].to_numpy(
        dtype=float
    )

    control_values = control_data[
        change_column
    ].to_numpy(
        dtype=float
    )

    result = results[
        domain_name
    ]

    # --------------------------------------------------------
    # Treatment panel
    # --------------------------------------------------------
    treatment_axis = axes[
        row_index,
        0,
    ]

    add_individual_change_panel(
        axis=treatment_axis,
        participant_labels=treatment_labels,
        change_values=treatment_values,
        color=TREATMENT_COLOR,
    )

    if domain_name == "Motor":
        treatment_title = (
            f"{domain_name} – Treatment "
            f"(n = {result['n_treatment']})\n"
            f"Wilcoxon p "
            f"{format_p_value(result['p_treatment'])}\n"
            f"MWU p "
            f"{format_p_value(result['p_between'])}, "
            f"r = {result['r']:.2f}"
        )
    else:
        treatment_title = (
            f"{domain_name} – Treatment "
            f"(n = {result['n_treatment']})\n"
            f"Wilcoxon p "
            f"{format_p_value(result['p_treatment'])}, "
            f"$p_{{adj}}$ "
            f"{format_p_value(result['p_treatment_adj'])}\n"
            f"MWU p "
            f"{format_p_value(result['p_between'])}, "
            f"$p_{{adj}}$ "
            f"{format_p_value(result['p_between_adj'])}, "
            f"r = {result['r']:.2f}"
        )

    treatment_axis.set_title(
        treatment_title,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        pad=8,
    )

    treatment_axis.text(
        -0.14,
        1.10,
        PANEL_LABELS[
            row_index * 2
        ],
        transform=treatment_axis.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="left",
        clip_on=False,
    )

    # --------------------------------------------------------
    # Control panel
    # --------------------------------------------------------
    control_axis = axes[
        row_index,
        1,
    ]

    add_individual_change_panel(
        axis=control_axis,
        participant_labels=control_labels,
        change_values=control_values,
        color=CONTROL_COLOR,
    )

    if domain_name == "Motor":
        control_title = (
            f"{domain_name} – Control "
            f"(n = {result['n_control']})\n"
            f"Wilcoxon p "
            f"{format_p_value(result['p_control'])}"
        )
    else:
        control_title = (
            f"{domain_name} – Control "
            f"(n = {result['n_control']})\n"
            f"Wilcoxon p "
            f"{format_p_value(result['p_control'])}, "
            f"$p_{{adj}}$ "
            f"{format_p_value(result['p_control_adj'])}"
        )

    control_axis.set_title(
        control_title,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        pad=8,
    )

    control_axis.text(
        -0.14,
        1.10,
        PANEL_LABELS[
            row_index * 2 + 1
        ],
        transform=control_axis.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="left",
        clip_on=False,
    )


# ============================================================
# Shared axis label and layout
# ============================================================
figure.text(
    0.04,
    0.50,
    "Change in FMA-UE Score\n(Post − Baseline)",
    verticalalignment="center",
    horizontalalignment="center",
    rotation="vertical",
    fontsize=AXIS_LABEL_SIZE,
)

figure.tight_layout(
    rect=[0.07, 0.04, 1.0, 0.99],
    h_pad=5.5,
    w_pad=2.5,
)


# ============================================================
# Save Supplementary Figure S1
# ============================================================
output_files = {
    "TIFF": OUTPUT_DIRECTORY / "FigS1.tif",
    "EPS": OUTPUT_DIRECTORY / "FigS1.eps",
    "SVG": OUTPUT_DIRECTORY / "FigS1.svg",
    "PDF": OUTPUT_DIRECTORY / "FigS1.pdf",
    "PNG": OUTPUT_DIRECTORY / "FigS1.png",
}

figure.savefig(
    output_files["TIFF"],
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
    pil_kwargs={
        "compression": "tiff_lzw",
    },
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

print(
    "\nSupplementary Figure S1 files saved successfully:"
)

for file_format, file_path in output_files.items():
    print(
        f"{file_format}: {file_path}"
    )

plt.show()