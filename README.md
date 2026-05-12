# CaDRe: Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis

Reference implementation of **CaDRe** (**Ca**usal **D**iscovery and **Re**presentation learning), the framework introduced in our ICML 2026 paper *Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis*.

CaDRe jointly identifies (i) the causal structure among observed climate variables and (ii) the latent dynamic process that drives them, from time-series data alone, with nonparametric identifiability guarantees.

> **Paper:** [Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis](https://openreview.net/forum?id=MZxhwFT7yF)
> **Authors:** Minghao Fu, Biwei Huang, Zijian Li, Yujia Zheng, Ignavier Ng, Guangyi Chen, Yingyao Hu, Kun Zhang
> **Affiliations:** MBZUAI, CMU, UCSD, JHU

## Method at a glance

```
            x_{t-1}, x_t, x_{t+1}                latent driver z_t with sparse temporal
                  observed                       dynamics (instantaneous + lagged)
                     │                                          │
                     ▼                                          │
              ┌──────────────┐                                  │
              │  z-encoder φ │──► ẑ_t  ─────────┐               │
              ├──────────────┤                  │ ─► flow prior r(ẑ) ─► J_r (latent DAG)
              │  s-encoder η │──► ŝ_t  ─────────┤
              └──────────────┘                  │
                                                ▼
                                          decoder ψ
                                                │
                                                ▼
                                              x̂_t
                                                │
                                                ▼
                  J_g(x_t) = I − D_m · J_m⁻¹  (observed causal graph, Corollary 1)
```

Three theoretical pillars (paper §3):

1. **Identifiability of the latent space** (Theorem 3.2): three contiguous observations suffice to recover `ẑ_t = h_z(z_t)` up to an invertible differentiable transformation.
2. **SEM ⇔ nonlinear ICA equivalence** (Lemma 3.3) + **functional equivalence** (Theorem 3.5): `J_g · J_m = J_m − D_m`. This lets us read the observed causal DAG off the ICA Jacobian without invertibility of the mixing.
3. **Identifiability of the observed causal graph** (Theorem 3.6) under a mild generation-variability condition.

## Repository layout

| Path | Role | Paper section |
|---|---|---|
| `SSM/` | Main CaDRe trainers and evaluators for CESM2 / WeatherBench. State-space VAE with z-encoder, s-encoder, decoder, flow priors. | §4 (Estimation), §5.2 |
| `LiLY/` | Core modules: VAE encoders/decoders, flow priors (`modules/components/spline.py`), latent dynamics models (`modules/CESM2.py`, `modules/nonparam.py`), datasets. | §4 |
| `Caulimate/` | Lightweight library: data simulation, graph utilities, metrics, visualisation. Imported as a package. | §5.1, Appendix C |
| `forecasting/` | Forecasting baselines and CESM2/ERSST/WeatherBench evaluation. | §5.2, Tables 5–6, 18–20 |
| `analyze/` | Climate-data visualisations, mask construction, eud-distance pruning, downstream analysis notebooks. | §5.2, Figs. 5, 10, 11 |
| `dataset/` | Pre-processing scripts for CESM2 / WeatherBench, downscaling. | Appendix C.2 |
| `scripts/` | High-level launchers for CESM2 / CESM3 training and inference. | §5.2 |
| `tests/` | Smoke tests and visualisation utilities. | — |
| `CI_comp/` | Comparison with constraint-based CI-based methods (LPCMCI). | §5.1 |
| `CauDiff/` | Earlier causal-diffusion experiments (kept for reference). | — |
| `LinGau`, `LatLinGau`, `GPDCM`, `General`, `neurips` | Predecessor experiments and ablations from earlier project iterations. Not used in the published results. | legacy |
| `quickstart.ipynb` | End-to-end notebook for downloading data and running CaDRe. | — |

## Installation

```bash
# 1. Clone
git clone https://github.com/MinghaoFu/CaDRe.git
cd CaDRe

# 2. Set up a fresh environment (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Install the lightweight Caulimate utilities
pip install -e package/
```

## Environment variables

Personal cluster paths have been replaced with placeholders. Set the following before running any script:

```bash
export PROJECT_ROOT="$(pwd)"           # repo root
export DATA_DIR=/path/to/your/data     # CESM2, ERSST, WeatherBench, simulated
export CKPT_DIR=/path/to/checkpoints   # model checkpoints
export LOG_DIR=/path/to/logs           # training logs / wandb dir
export OUTPUT_DIR=/path/to/outputs     # artefacts, figures
export DATASET_DIR="$DATA_DIR"         # legacy alias used by some scripts
```

Scripts that still contain `${DATA_DIR}`, `${PROJECT_ROOT}`, etc. as string literals expect these variables to be substituted (typically through `os.path.expandvars` or shell-side expansion). Replace those literals with the appropriate paths before training.

## Data

We use four datasets, all publicly available:

| Dataset | Source | Notes |
|---|---|---|
| **CESM2 Pacific SST** | NCAR CESM2 control run (1850–2014 monthly SST) | Restricted to oceanic Pacific grid; ~28 086 spatial variables; we downsample to 6×14 grid for training. |
| **ERSST v5** | [NOAA NCEI](https://www.ncei.noaa.gov/products/extended-reconstructed-sst) | 1880–present global monthly SST; we use 100-dim spatial downsample. |
| **WeatherBench** | [Rasp et al. 2020](https://github.com/pangeo-data/WeatherBench) | Used for wind-field surrogate evaluation in §5.2. |
| **Weather (Jena)** | [BGC Jena](https://www.bgc-jena.mpg.de/wetter/) | 10-minute rooftop station data for forecasting. |

A working notebook for downloading and inspecting WeatherBench is provided in `quickstart.ipynb`. Pre-processed CESM2 derivatives (`CESM2_pacific_SST.pkl`, `CESM2_pacific_grouped_SST.nc`, downscaled coordinates) are produced by `dataset/CESM2_analysis.ipynb`. We do **not** redistribute the raw datasets in this repository; use the official sources above and place them under `$DATA_DIR/CESM2/`, `$DATA_DIR/ERSST/`, etc.

## Reproducing the paper

### Synthetic identifiability (§5.1, Tables 11–15, Fig. 4)

```bash
# Train CaDRe on simulated data (varying d_x, d_z, sparsity)
python General/train_syn.py --config General/nonparam.yaml

# Constraint-based + score-based comparison (CD-NOD, PCMCI, LPCMCI, BIC, GSF)
jupyter notebook CI_comp/LPCMCI.ipynb
```

### Real-world CESM2 / WeatherBench training (§5.2, Tables 5, 16)

```bash
# Main CESM2 trainer
python SSM/train_CESM2.py

# WeatherBench
python SSM/test_WB.py
```

### Forecasting comparisons (Tables 5–6, 18–20)

```bash
jupyter notebook forecasting/CESM2Forecast.ipynb
jupyter notebook forecasting/DownscaledCESM2Forecast.ipynb
```

### Climate visualisations (Figs. 5, 10, 11)

```bash
jupyter notebook analyze/WeatherBench_wind.ipynb
jupyter notebook analyze/init_downscale_CESM2.ipynb
```

## Repository status

This is the camera-ready code release. The repository keeps the full evolution of the project (older predecessor experiments are still present under `LinGau/`, `LatLinGau/`, `GPDCM/`, `General/`, `neurips/`) for transparency and reproducibility. The primary CaDRe code path used in the paper is `SSM/` + `LiLY/` + `Caulimate/`.

If you find a missing dataset preprocessing step, an outdated checkpoint path, or any sanitizer-related residue (placeholder `${DATA_DIR}` strings that should be replaced), please open an issue.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{fu2026learning,
  title     = {Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis},
  author    = {Fu, Minghao and Huang, Biwei and Li, Zijian and Zheng, Yujia and
               Ng, Ignavier and Chen, Guangyi and Hu, Yingyao and Zhang, Kun},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## Acknowledgements

CaDRe builds on the eigendecomposition technique of [Hu & Schennach (2008)](https://www.cambridge.org/core/journals/econometric-theory/article/abs/instrumental-variable-treatment-of-nonclassical-measurement-error-models/) and on the temporal CRL line of work (TDRL, LEAP, CaRiNG, IDOL, CITRIS). We thank the CESM2, ERSST, and WeatherBench teams for releasing their data, and our reviewers for the feedback that shaped the camera-ready version.

## Contact

For questions, please contact the corresponding authors:

- Minghao Fu — `minghao.fu@mbzuai.ac.ae`
- Biwei Huang — `bih007@ucsd.edu`
