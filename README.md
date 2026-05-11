# PVOS: Parallel Variation Operator Strategy for Multi-objective Evolutionary Optimization

Source code, configuration files, raw results, and post-processing notebooks for the manuscript:

**Operator-Level Parallel Crossover Ensembles with Fixed Evaluation Budgets for Multi-objective Evolutionary Optimization**
Meshari Alhazmi, Kyle Camarda
Department of Chemical and Petroleum Engineering, The University of Kansas
Submitted to *Applied Soft Computing* (Elsevier).

> Zenodo archive DOI: *to be added after the v1.0.0 release.*

---

## Overview

This repository contains the implementations, experimental configurations, raw per-run results, and analysis notebooks underlying every table and figure in the manuscript.

The Parallel Variation Operator Strategy (PVOS) is a host-transferable, fixed-budget multi-operator variation stage. PVOS applies several complementary crossover operators concurrently to the same parent pairs, pools the resulting offspring within each generation, and leaves the host MOEA's environmental selection unchanged. The PVOS-4C configuration combines four operators: Simulated Binary Crossover (SBX), Blend Crossover (BLX-α), Single-Point Crossover (SPX), and Uniform Crossover (UX).

The repository covers the five experimental blocks reported in the paper:

1. PVOS-2C operator pairing study (six two-operator combinations).
2. Baseline NSGA-II versus PVOS-4C on 27 benchmark problems.
3. PVOS-4C versus four adaptive operator-selection baselines (Borg, HNSGA, EnXEA, OVEA) under a common four-operator pool and equal-budget protocol.
4. PVOS-4C transferred to SPEA2, NSGA-III, and AGE-MOEA2 without retuning.
5. Styrene monomer reactor case study (single-bed, steam-injected, double-bed configurations).

---

## Repository layout

```
pvos/
├── experiments/             Algorithm implementations and run notebooks for the five experimental blocks
├── Post analysis Result/    Post-processing notebook and Excel summary workbooks for all tables
├── figures/                 Figures used in the manuscript and supplementary material
├── CITATION.cff             Citation metadata
├── LICENSE                  MIT License
├── requirements.txt         Python dependencies
└── README.md                This file
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
| SciPy, pandas, openpyxl, matplotlib | latest compatible |

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

All tables and figures are reproducible from the notebooks under `experiments/` and `Post analysis Result/`. The mapping below identifies which notebook produces each artifact in the paper.

| Paper artifact | Source notebook |
| --- | --- |
| Table 5 (Population-size sensitivity) | `experiments/...` |
| Table 6 (NSGA-II vs PVOS-4C, 27 benchmarks) | `experiments/...`, summarized in `Post analysis Result/...` |
| Table 7 (PVOS-4C vs AOS baselines, suite-level) | `experiments/...`, summarized in `Post analysis Result/...` |
| Table 8 (Host-transfer summary) | `experiments/...`, summarized in `Post analysis Result/...` |
| Table 9 (Styrene case study) | `experiments/...`, summarized in `Post analysis Result/...` |
| Tables S1 and S2 (PVOS-2C pairing, per-problem) | `Post analysis Result/...` |
| Tables S3 to S6 (AOS comparison, per-problem and audits) | `Post analysis Result/...` |
| Table S7 (Host-transfer, per-problem) | `Post analysis Result/...` |
| Figure 3 (PVOS-2C pairing median summary) | `figures/...` |
| Figure 4 (Population-size sensitivity curves) | `figures/...` |
| Figure 5 (Pareto-front snapshots) | `figures/...` |
| Figure 6 (Attainment lift and best-IGD⁺ boxplots) | `figures/...` |
| Figure 7 (Per-problem ΔIGD⁺ across transferred hosts) | `figures/...` |
| Figure 8 (Crossover-survival shares on WFG8) | `figures/...` |
| Figure 9 (Styrene wall-clock behavior) | `figures/...` |
| Figure 10 (Styrene operator-survival composition) | `figures/...` |

> The exact notebook filename in each row should be filled in to match the final repository contents before submission.

Re-running tables and figures from the stored raw CSVs is fast. Re-running full experiments from scratch is expensive, particularly the styrene case study.

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

| Configuration | Decision variables | Inequality constraints |
| --- | --- | --- |
| Single-bed (SB) | 8 | 5 |
| Steam-injected (SI) | 10 | 7 |
| Double-bed (DB) | 11 | 7 |

The reactor is modeled as a pseudo-homogeneous plug-flow unit with coupled mass, energy, and pressure-drop balances, the six-reaction Sheel-Crowe styrene system, the intrinsic kinetics of Abdalla et al., and temperature-dependent thermochemical properties. HE1 is modeled as a counter-current sensible-heat exchanger with LMTD-based area calculation.

Empirical reference sets `R*` and normalization bounds `(refmin, refmax)` for each configuration are included with the styrene experiment files. They were constructed by pooling 60 final Pareto fronts (30 baseline NSGA-II and 30 PVOS-4C runs) per configuration and filtering to their non-dominated union, as described in Section 4.4 of the manuscript.

Styrene runs use `N = 80`, `G = 100`, and 30 paired seeds (approximately 8,000 function evaluations per run).

---

## Random seeds and statistical protocol

The ordered list of 30 random seeds used across all paired experiments is stored alongside the experimental notebooks. Within each problem, the same seed ordering is used across compared algorithms so that two-method comparisons are paired by seed.

Significance testing in the manuscript uses two-sided Wilcoxon signed-rank tests at `α = 0.05` with Holm correction within each metric-specific family of tests. For the suite-level AOS comparison, Friedman tests are followed by Holm-corrected paired Wilcoxon tests using PVOS-4C as the control. All tests are implemented in SciPy with `zero_method='wilcox'` and `method='auto'`.

---

## Citation

If you use this code or data, please cite the paper and the repository.

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

This project is released under the MIT License. See `LICENSE` for details.

---

## Contact

**Meshari Alhazmi**, [mesh@ku.edu](mailto:mesh@ku.edu)
**Kyle Camarda** (corresponding author), [camarda@ku.edu](mailto:camarda@ku.edu)
Department of Chemical and Petroleum Engineering, The University of Kansas, 1530 W 15th St, Lawrence, KS 66045, United States of America.
