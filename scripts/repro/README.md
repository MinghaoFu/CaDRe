# Reproducing the paper

One-line launchers for the experiments reported in
*Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis* (ICML 2026).

All scripts assume you have set the environment variables described in the
top-level `README.md` (`PROJECT_ROOT`, `DATA_DIR`, `CKPT_DIR`, `LOG_DIR`,
`OUTPUT_DIR`). They wrap the underlying Python entry-points with sensible
defaults so the lagged real-world experiments can be re-run without
re-deriving every flag from the paper appendix.

| Script | Paper section | What it runs |
|---|---|---|
| `run_synthetic.sh` | §5.1, Tables 11–15, Fig. 4 | Train CaDRe on simulated data; sweep `d_x`, `d_z`, and sparsity regime. |
| `run_neural_baselines.sh` | App. D, Fig. 8 | DYNOTEARS / CUTS / Rhino / Jacobian-CD baselines, then CaDRe at the same grid. |
| `run_causalrivers.sh` | App. D, Fig. 9(b) | External validation on the CausalRivers benchmark. |
| `run_higher_order.sh` | App. D, Table 16 | Higher-order ($L=2$) Markov latent dynamics. |
| `run_cesm2.sh` | §5.2, Tables 5, 6 | Train CaDRe on CESM2 Pacific SST; evaluate on the wind-field surrogate. |
| `run_weatherbench.sh` | §5.2, Tables 5, 6 | Train + evaluate on WeatherBench. |
| `run_ersst.sh` | §5.2, Table 6 | Train + evaluate on ERSST v5. |
| `run_forecasting.sh` | §5.2, Tables 5, 18–20 | Forecasting comparison against transformer / Mamba / TimesNet / MICN baselines. |
| `run_visualize.sh` | §5.2, Figs. 5, 10, 11 | Climate maps, ENSO causal chain, latent-variable physical alignment. |
| `run_all.sh` | All of the above | Sequential full reproduction. Long-running. |

## Notes

* These scripts run on a single GPU by default. Multi-GPU is supported by the
  underlying Lightning trainers via the standard `--gpus N` argument; pass it
  through with `EXTRA_ARGS="--gpus 4" ./scripts/repro/run_cesm2.sh`.
* The synthetic scripts use the data generators in `General/gen_data.py` and
  do not need any external data download.
* The real-world scripts assume that you have placed the datasets under
  `$DATA_DIR/CESM2/`, `$DATA_DIR/WeatherBench/`, and `$DATA_DIR/ERSST/`
  following the layout described in the top-level `README.md`. The raw data
  is not redistributed in this repository.
