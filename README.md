# PVOS: Parallel Variation Operator Strategy for Multi-objective Evolutionary Optimization

Source code, experiment notebooks, post-analysis workbooks, and figures for the manuscript:

**Operator-Level Parallel Crossover Ensembles with Fixed Evaluation Budgets for Multi-objective Evolutionary Optimization**
Meshari Alhazmi, Kyle Camarda
Department of Chemical and Petroleum Engineering, The University of Kansas
Submitted to *Applied Soft Computing* (Elsevier).

> Zenodo archive DOI: *to be added after the v1.0.0 release.*

---

## Overview

This repository contains the implementations, experiment notebooks, consolidated post-analysis workbooks, and rendered figures underlying every table and figure in the manuscript. It is the public artifact referenced in the manuscript's Data availability statement.

The Parallel Variation Operator Strategy (PVOS) is a host-transferable, fixed-budget multi-operator variation stage. PVOS applies several complementary crossover operators concurrently to the same parent pairs, pools the resulting offspring within each generation, and leaves the host MOEA's environmental selection unchanged. The PVOS-4C configuration combines four operators: Simulated Binary Crossover (SBX), Blend Crossover (BLX-α), Single-Point Crossover (SPX), and Uniform Crossover (UX).

The repository covers the five experimental blocks reported in the paper:

1. PVOS-2C operator pairing study (six two-operator combinations).
2. Baseline NSGA-II versus PVOS-4C on 27 benchmark problems, with the population-size sensitivity analysis.
3. PVOS-4C versus four adaptive operator-selection baselines (Borg, HNSGA, EnXEA, OVEA) under a common four-operator pool and equal-budget protocol.
4. PVOS-4C transferred to SPEA2, NSGA-III, and AGE-MOEA2 without retuning.
5. Styrene monomer reactor case study (single-bed, steam-injected, double-bed configurations).

---

## Repository layout

```
pvos/
├── CITATION.cff                       Citation metadata
├── README.md                          This file
├── requirements.txt                   Python dependencies
├── experiments/                       Algorithm implementations and run notebooks
│   ├── PVOS-2C implementation (all 6 operator pairs).ipynb
│   ├── NSGA2-PVOS-4c.ipynb
│   ├── NSGA2-Borg.ipynb
│   ├── NSGA2-HNSGA.ipynb
│   ├── NSGA2-EnXEA.ipynb
│   ├── NSGA2-OVEA.ipynb
│   ├── SPEA2 PVOS-4C.ipynb
│   ├── NSGA3-PVOS-4C.ipynb
│   ├── AGE-MOEA2-PVOS-4C.ipynb
│   └── styrene/                       Styrene reactor case study
│       ├── SB/                        Single-bed configuration
│       │   ├── NSGA-2 for tarafder_styrene.ipynb
│       │   ├── PVOS-4C for tarafder_styrene.ipynb
│       │   ├── objectives/tarafder_styrene.py
│       │   ├── rstar_utils.py
│       │   ├── normalization_minmax.json
│       │   ├── R_star_raw.csv
│       │   └── R_star_normalized.csv
│       ├── SI/                        Steam-injected configuration (same structure as SB)
│       └── DB/                        Double-bed configuration (same structure as SB)
├── Post analysis Result/              Consolidated post-analysis workbooks
│   ├── Tables 1-9.xlsx                Main paper tables
│   └── Supplementary Tables S1-S7.xlsx Supplementary tables
└── figures/                           Manuscript and supplementary figures
    ├── Figure 1.pdf                   PVOS schematic
    ├── Figure 2.pdf                   Styrene reactor configurations
    ├── Figure 3.png                   PVOS-2C pairing median summary
    ├── Figure 4.pdf                   Population-size sensitivity curves
    ├── Figure 5_*.png                 Pareto-front snapshots (ZDT3, ZDT6, DTLZ1, DTLZ6, DC1DTLZ3)
    ├── Figure 6.pdf                   Attainment lift and best-IGD+ boxplots
    ├── Figure 7.pdf                   Per-problem ΔIGD+ across transferred hosts
    ├── Figure 8.pdf                   Crossover-survival shares on WFG8
    ├── Figure 9a.pdf, 9b.pdf, 9c.pdf  Styrene wall-clock behavior
    ├── Figure 10.pdf                  Styrene operator-survival composition
    └── Figure S1 *.pdf                IGD+ trajectories for the Figure 5 problems
```

---

## Software environment

The experiments were executed in the environment reported in Section 4.3 of the manuscript:

| Component | Version |
| --- | --- |
| Python | 3.13 |
| pymoo | 0.6.1 |
| NumPy | 2.1 |
| joblib | 1.4 |
| SciPy | 1.13 or compatible |
| statsmodels | 0.14 |
| pandas | 2.2 or compatible |
| openpyxl | 3.1 or compatible |
| matplotlib | 3.8 or compatible |

The exact pins are listed in `requirements.txt`.

### Hardware used in the paper

| Component | Value |
| --- | --- |
| CPU | Intel Core i9-14900KF |
| Memory | 32 GB RAM |
| Operating system | Windows 11 Home |

Wall-clock numbers reported in Tables 7 and 9 are tied to this configuration. Reproducing solution quality (IGD⁺ and HV) does not require the same hardware; reproducing runtime numbers exactly does.

---

## Installation

```bash
git clone https://github.com/CamardasLabKU/pvos.git
cd pvos

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS or Linux

# Install dependencies
pip install -r requirements.txt
```

Launch the notebooks:

```bash
jupyter lab
```

---

## Reproducing the manuscript tables and figures

All tables and figures are reproducible from the notebooks under `experiments/`. Per-run results, per-seed metrics, and summary statistics are embedded in the executed notebooks' cell outputs, and consolidated values used in the manuscript are reported in the two workbooks under `Post analysis Result/`. The mapping below identifies which notebook produces each artifact in the paper.

| Paper artifact | Source notebook(s) | Summary workbook |
| --- | --- | --- |
| Table 5 (Population-size sensitivity) | `experiments/NSGA2-PVOS-4c.ipynb` | `Post analysis Result/Tables 1-9.xlsx` |
| Table 6 (NSGA-II vs PVOS-4C, 27 benchmarks) | `experiments/NSGA2-PVOS-4c.ipynb` | `Post analysis Result/Tables 1-9.xlsx` |
| Table 7 (PVOS-4C vs AOS baselines, suite-level) | `experiments/NSGA2-PVOS-4c.ipynb`, `NSGA2-Borg.ipynb`, `NSGA2-HNSGA.ipynb`, `NSGA2-EnXEA.ipynb`, `NSGA2-OVEA.ipynb` | `Post analysis Result/Tables 1-9.xlsx` |
| Table 8 (Host-transfer summary) | `experiments/SPEA2 PVOS-4C.ipynb`, `NSGA3-PVOS-4C.ipynb`, `AGE-MOEA2-PVOS-4C.ipynb` | `Post analysis Result/Tables 1-9.xlsx` |
| Table 9 (Styrene case study) | `experiments/styrene/{SB,SI,DB}/NSGA-2 for tarafder_styrene.ipynb`, `experiments/styrene/{SB,SI,DB}/PVOS-4C for tarafder_styrene.ipynb` | `Post analysis Result/Tables 1-9.xlsx` |
| Tables S1 and S2 (PVOS-2C pairing, per-problem) | `experiments/PVOS-2C implementation (all 6 operator pairs).ipynb` | `Post analysis Result/Supplementary Tables S1-S7.xlsx` |
| Tables S3 to S6 (AOS comparison, per-problem and audits) | `experiments/NSGA2-PVOS-4c.ipynb`, `NSGA2-Borg.ipynb`, `NSGA2-HNSGA.ipynb`, `NSGA2-EnXEA.ipynb`, `NSGA2-OVEA.ipynb` | `Post analysis Result/Supplementary Tables S1-S7.xlsx` |
| Table S7 (Host-transfer, per-problem) | `experiments/SPEA2 PVOS-4C.ipynb`, `NSGA3-PVOS-4C.ipynb`, `AGE-MOEA2-PVOS-4C.ipynb` | `Post analysis Result/Supplementary Tables S1-S7.xlsx` |
| Figure 1 (PVOS schematic) | conceptual diagram, not data-driven | `figures/Figure 1.pdf` |
| Figure 2 (Styrene reactor configurations) | conceptual diagram, not data-driven | `figures/Figure 2.pdf` |
| Figure 3 (PVOS-2C pairing median summary) | `experiments/PVOS-2C implementation (all 6 operator pairs).ipynb` | `figures/Figure 3.png` |
| Figure 4 (Population-size sensitivity curves) | `experiments/NSGA2-PVOS-4c.ipynb` | `figures/Figure 4.pdf` |
| Figure 5 (Pareto-front snapshots) | `experiments/NSGA2-PVOS-4c.ipynb` | `figures/Figure 5_*.png` |
| Figure 6 (Attainment lift and best-IGD⁺ boxplots) | `experiments/NSGA2-PVOS-4c.ipynb` and the four AOS notebooks | `figures/Figure 6.pdf` |
| Figure 7 (Per-problem ΔIGD⁺ across transferred hosts) | `experiments/SPEA2 PVOS-4C.ipynb`, `NSGA3-PVOS-4C.ipynb`, `AGE-MOEA2-PVOS-4C.ipynb` | `figures/Figure 7.pdf` |
| Figure 8 (Crossover-survival shares on WFG8) | `experiments/NSGA2-PVOS-4c.ipynb`, `SPEA2 PVOS-4C.ipynb`, `NSGA3-PVOS-4C.ipynb`, `AGE-MOEA2-PVOS-4C.ipynb` | `figures/Figure 8.pdf` |
| Figure 9 (Styrene wall-clock behavior) | `experiments/styrene/{SB,SI,DB}/*.ipynb` | `figures/Figure 9a.pdf`, `9b.pdf`, `9c.pdf` |
| Figure 10 (Styrene operator-survival composition) | `experiments/styrene/{SB,SI,DB}/PVOS-4C for tarafder_styrene.ipynb` | `figures/Figure 10.pdf` |
| Figure S1 (IGD⁺ trajectories) | `experiments/NSGA2-PVOS-4c.ipynb` | `figures/Figure S1 *.pdf` |

Re-running the tables and figures from already-executed notebooks is fast because per-run results and intermediate aggregates are stored in the notebook outputs. Re-running the full experiments from scratch is expensive, particularly the styrene case study.

---

## Benchmark suite

Twenty-seven analytical problems are used:

| Family | Problems |
| --- | --- |
| ZDT | ZDT1, ZDT2, ZDT3, ZDT4, ZDT6 |
| DTLZ | DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7 |
| WFG | WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9 |
| Constrained DTLZ | C1DTLZ1, C2DTLZ2, C3DTLZ4 |
| DC-DTLZ | DC1DTLZ1, DC1DTLZ3, DC3DTLZ1 |

Per-problem dimensionality, objective count, constraint count, and generation budget follow Table 2 of the manuscript. Population size is `N = 100` for the main benchmark experiments. The population-size sensitivity analysis (Table 5) uses `N ∈ {40, 100, 200, 400}` with `N × G` held constant per problem.

Common parameter settings: `pc = 0.9` for all crossovers, `pm = 1/D` polynomial mutation, SBX distribution index `ηc = 15`, polynomial-mutation distribution index `ηm = 20`, BLX-α mixing coefficient `α = 0.5`, and per-variable uniform swap probability `pswap = 0.5`.

---

## Styrene monomer reactor case study

Three reactor configurations are modeled as bi-objective constrained problems, following Tarafder et al. (2005):

| Configuration | Decision variables | Inequality constraints | Folder |
| --- | --- | --- | --- |
| Single-bed (SB) | 8 | 5 | `experiments/styrene/SB/` |
| Steam-injected (SI) | 10 | 7 | `experiments/styrene/SI/` |
| Double-bed (DB) | 11 | 7 | `experiments/styrene/DB/` |

The reactor is modeled as a pseudo-homogeneous plug-flow unit with coupled mass, energy, and pressure-drop balances, the six-reaction Sheel-Crowe styrene system, the intrinsic kinetics of Abdalla et al., and temperature-dependent thermochemical properties. HE1 is modeled as a counter-current sensible-heat exchanger with LMTD-based area calculation.

Each configuration folder contains the process-model implementation (`objectives/tarafder_styrene.py`), the reference-set utilities (`rstar_utils.py`), the empirical reference set in raw and normalized form (`R_star_raw.csv`, `R_star_normalized.csv`), the normalization bounds (`normalization_minmax.json`), and the paired baseline NSGA-II and PVOS-4C notebooks. The reference sets and normalization bounds were constructed by pooling 60 final Pareto fronts (30 baseline NSGA-II and 30 PVOS-4C runs) per configuration and filtering to their non-dominated union, as described in Section 4.4 of the manuscript.

Styrene runs use `N = 80`, `G = 100`, and 30 paired seeds (approximately 8,000 function evaluations per run).

---

## Random seeds and statistical protocol

The ordered list of 30 random seeds used in each paired experiment is defined within the corresponding notebook. Within each problem, the same seed ordering is used across compared algorithms so that two-method comparisons are paired by seed.

Significance testing in the manuscript uses two-sided Wilcoxon signed-rank tests at `α = 0.05` with Holm correction within each metric-specific family of tests. For the suite-level AOS comparison, Friedman tests are followed by Holm-corrected paired Wilcoxon tests using PVOS-4C as the control. Wilcoxon tests are implemented in SciPy with `zero_method='wilcox'` and `method='auto'`; Holm correction uses `statsmodels`.

---

## Data and code availability

In line with the manuscript's Data availability statement, all source code and reproducibility materials supporting this study are publicly available in this repository. No additional restricted or proprietary data are required to reproduce the reported results.

---

## Citation

If you use this code, please cite the paper and this repository.

Paper:

```bibtex
@article{alhazmi2026pvos,
  title   = {Operator-Level Parallel Crossover Ensembles with Fixed Evaluation Budgets for Multi-objective Evolutionary Optimization},
  author  = {Alhazmi, Meshari and Camarda, Kyle},
  journal = {Applied Soft Computing},
  year    = {2026},
  note    = {Under review}
}
```

Repository: see `CITATION.cff`, or click *Cite this repository* on the GitHub page. After the Zenodo archive is created, the DOI will be added to both `CITATION.cff` and the badge at the top of this README.

---

## License

This project is released under the MIT License. License metadata is recorded in `CITATION.cff`.

---

## Contact

**Meshari Alhazmi**, [mesh@ku.edu](mailto:mesh@ku.edu)
**Kyle Camarda** (corresponding author), [camarda@ku.edu](mailto:camarda@ku.edu)
Department of Chemical and Petroleum Engineering, The University of Kansas, 1530 W 15th St, Lawrence, KS 66045, United States of America.
