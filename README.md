# Scripts

Result-producing scripts for the paper *"Mitigating Behavioral Profiling Attacks on Wearable Sensor Data with Adaptive Differential Privacy"*.

Chart-rendering scripts are intentionally excluded — this folder contains only the pipeline that produces the underlying numerical results.

## Contents

| File | Purpose |
|---|---|
| `pipeline.py` | End-to-end three-stage pipeline: Stage 1 baseline attack models, Stage 2 IBM `diffprivlib` Laplace perturbation (adaptive on WISDM, fixed on Keystroke100), Stage 3 privacy–utility evaluation under clean-trained and attacker-aware adversaries. Writes result workbooks to `output/`. |
| `requirements.txt` | Python dependencies. |

## Datasets

Two public datasets are required. They are **not** included in this repository.

1. **WISDM Smartphone and Smartwatch Activity and Biometrics Dataset** — download from the UCI ML repository (DOI: 10.24432/C5HK59). Place the extracted archive so that the ARFF files live under:
   ```
   <DATASETS_ROOT>/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/wisdm-dataset/arff_files/watch/
   ```
2. **Keystroke100** (Loy et al., 2007). Place the per-user files under:
   ```
   <DATASETS_ROOT>/keystrokes/keystroke100/
   ```

Point the pipeline at `<DATASETS_ROOT>` by either:

- setting the environment variable `SMARTWATCH_DATASETS_DIR`, **or**
- placing a `Datasets/` directory next to `pipeline.py`.

## Reproducing the paper numbers

```bash
pip install -r requirements.txt
python pipeline.py
```

The pipeline uses a fixed random seed (`SEED = 42`) and deterministic splits throughout. Results are written to `output/wisdm_results_adp.xlsx` and `output/keystroke100_results_fixed_dp.xlsx`. Expected runtime is a few minutes on a single CPU core.

## Parameters

The adaptive DP sweep reported in the paper uses:

- `eps_t ∈ {0.05, 0.10, 0.20, 0.50, 1.00}` (typing-side budget)
- `eps_n = 2.0` (non-typing-side budget)
- `tau = 0.60` (gate threshold for diagnostics only)
- jitter fraction 0.15 (multiplicative `Unif[-0.15, 0.15]` on `eps_i`, then clipped to `[min(eps_t, eps_n), max(eps_t, eps_n)]`)

The Keystroke100 fixed-DP benchmark sweeps `eps_f ∈ {0.05, 0.10, 0.50, 1.00, 2.00}`.
