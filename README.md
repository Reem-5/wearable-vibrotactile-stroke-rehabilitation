# Feasibility of Wearable Vibrotactile Stimulation Glove and Shirt in Upper-Limb Rehabilitation for Stroke Patients

**Authors:** Walaa Ayyad, Yeongdae Kim, Nathan Odom, and Mazen Al Borno

---

## Overview

This repository contains the de-identified clinical trial dataset, Python analysis scripts, and publication-quality figures associated with the manuscript:

> **Feasibility of Wearable Vibrotactile Stimulation Glove and Shirt in Upper-Limb Rehabilitation for Stroke Patients**

The repository provides all materials required to reproduce the statistical analyses and figures presented in the manuscript and is intended to support transparency and reproducible research.

---

## Repository Structure

```text
wearable-vibrotactile-stroke-rehabilitation/
│
├── Code/          # Python scripts for statistical analyses and figure generation
├── Data/          # De-identified clinical trial dataset
├── Figures/       # Publication-quality figures
├── LICENSE
├── README.md
└── .gitignore
```

---

## Data

The `Data` folder contains the de-identified clinical trial dataset:

- **Stroke_Vibrotactile_Rehabilitation_Data.xlsx**

The workbook includes the following worksheets:

| Worksheet | Description |
|-----------|-------------|
| **README** | Description of dataset variables and naming conventions |
| **FMA_UE** | Fugl–Meyer Assessment for Upper Extremity (FMA-UE) data |
| **MAS** | Modified Ashworth Scale (MAS) data |

The dataset has been fully de-identified and contains no personally identifiable participant information.

---

## Code

The `Code` folder contains the Python scripts used to reproduce all statistical analyses and publication-quality figures presented in the manuscript.

| Script | Description |
|---------|-------------|
| Figure5.py | FMA-UE baseline vs. post-intervention analysis |
| Figure6.py | FMA-UE within-group analysis |
| Figure7.py | FMA-UE between-group change-score analysis |
| Figure8.py | FMA-UE responder analysis |
| Figure9.py | MAS change-score analysis |
| Figure10.py | MAS responder analysis |
| FigureS1.py | Individual FMA-UE change-score plots |
| FigureS2.py | Individual MAS change-score plots |

Each script reproduces the corresponding statistical analysis and automatically exports the associated publication-quality figure.

---

## Figures

The `Figures` folder contains the publication-quality figures included in the manuscript.

Figures are provided in PDF format for viewing and reference.

---

## Software Requirements

The analyses were performed using **Python 3**.

Required Python packages include:

- numpy
- pandas
- scipy
- matplotlib
- seaborn
- statsmodels
- openpyxl

---

## Reproducing the Analyses

1. Clone this repository.
2. Install the required Python packages.
3. Ensure that `Stroke_Vibrotactile_Rehabilitation_Data.xlsx` is located in the `Data` folder.
4. Run the desired Python script from the `Code` folder.

Each script reproduces the corresponding statistical analysis and exports the associated figure.

---

## License

This repository is distributed under the **MIT License**.

---

## Citation

If you use the data or code from this repository, please cite the associated publication.

Complete citation information will be added after the manuscript is published.
