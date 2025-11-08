# FRSR Calibrate (v2.0) — Minimal CPL bridge to CLASS/MontePython

## Purpose
`frsr_calibrate` takes an FRSR calibration (finite‑band model) and emits a **minimal CLASS background** in **CPL form**—just enough to run theory (H(z), distances, Cℓ, …) and plug into MontePython.  
It intentionally avoids heavy pipeline logic so you can:
- generate `frsr_background.ini` with `(Ω_fld, w0_fld, wa_fld, cs2_fld, use_ppf)`,
- keep provenance in a simple run folder,
- and reuse the same CPL pair with CLASS or MontePython.

**Non‑goals:** This repo does *not* run CLASS/MCMC for you; it only prepares the bridge files and optional helpers.

## Background (DOIs)
- **FRSR v1 (concept & master equation)** — Zenodo DOI: [10.5281/zenodo.17393804](https://doi.org/10.5281/zenodo.17393804)
  *(Sang Wook Lee, 2025)*
- **FRSR v2.0 — Curvature Dynamics from a Spectrally-Constrained Quantum Entanglement Field**  
*Sang Wook Lee (2025)* — Zenodo DOI: [Pending publication](https://zenodo.org/)

> In v2.0/2.1 we treat the late‑time effect as **time curvature** mapped to CPL (w₀, wₐ). Calibration can use a spectral kernel, but runtime is CPL‑only (keeps CLASS and MP simple).

---

## Config files: defaults vs calibration

- **`src/frsr/config/frsr.default.toml`** — holds *sane defaults* used by the library (anchors like `H0_km_s_Mpc`, `Ω_b0`, `Ω_cdm0`, and the calibration dials for the spectral kernel). It’s read by `frsr.io.frsr_loader` and some CLI tools to avoid repeating boilerplate. Most users don’t need to edit it.
- **`src/frsr/config/frsr.calibration.toml`** — where *scan/calibration grids and policies* live in v2.0. Change this if you are exploring kernels or grid searches.
- **Minimal bridge path (this README):** `run_calibrate.py` does **not** require either TOML. It only needs an `Ω_de0` (from `params.json` or `--defaults`) and the CPL pair `(w0, wa)` to emit `frsr_background.ini` for CLASS.

**When should I touch `frsr.default.toml`?**
- If you want to change the **global default anchors** or **runtime dials** across multiple runs without passing them in every time.
- If you’re writing a small script that uses `frsr.io.frsr_loader` to build a background from defaults, then exporting to CLASS.

**When can I ignore it?**
- If you follow the minimal flow shown here (provide `Ω_de0` + `(w0, wa)` and generate `frsr_background.ini`), you can ignore the TOML files entirely.

## How to

### 0) Install
```bash
python -m pip install -r requirements.txt
# optional: editable install for CLI entry points
python -m pip install -e .
```

### 1) Create a run directory and (optionally) a params.json
`params.json` may include:
```json
{
  "class_params": {
    "Omega_de0": 0.68
  }
}
```
If missing, you can pass a fallback with `--defaults`.

### 2) Emit a minimal CLASS background (.ini, CPL only)
```bash
python -m frsr.cli.run_calibrate \
  --run-dir runs/calibrate/demo \
  --params runs/calibrate/demo/params.json \
  --defaults 0.68 \
  --w0 -0.98 --wa 0.0
```
This writes:
- `runs/calibrate/demo/frsr_background.ini`  (Ω_fld, w0_fld, wa_fld, cs2_fld, use_ppf)  
- log lines explaining which source provided `Ω_de0`

You can then point CLASS to this `.ini` (or copy its `[background]` block into your CLASS config).

### 3) (Optional) MontePython usage
If you sample **(ε, α)** and map to CPL inside MP, use the helper hook:
- `frsr.utils.frsr_mp_hook` (see code under `src/frsr/utils/frsr_mp_hook.py`)
- Typical mapping (legacy proxy): `w0 = -1 + ε`, `wa = −α·ε`  
  For v2.0 we recommend mapping from the **spectral slope** to CPL *outside* MP and passing `(w0, wa)` directly.

---

## Examples

**Minimal end‑to‑end (CPL only):**
```bash
# 1) Prepare CPL background
python -m frsr.cli.run_calibrate \
  --run-dir runs/calibrate/demo \
  --defaults 0.68 \
  --w0 -0.98 --wa 0.0

# 2) Run CLASS with the generated .ini (pseudo):
class_public/class \
  frsr_background.ini \
  output=demo_root
```

**Smoke tools (included):**
```
runs/smoke_tools/
  ├─ frsr_background.ini      # a tiny example background
  ├─ frsr_exp_sc0.param       # example kernel knobs (exp, sc=0)
  ├─ frsr_exp_sc02.param      # example kernel knobs (exp, sc>0)
  ├─ frsr_power2.param        # example kernel knobs (power2)
  └─ frsr_ladder.csv          # optional diagnostics
```

---

## Pipelines (visual)

### Baseline / single-run pipeline (no sampling)
```
+-----------------------------------+
| frsr.cli.run_calibrate.py (FRSR)  |
|  - fit finite bands               |
|  - produce minimal bridge data    |
+------------------+----------------+
                   | writes
                   v
  runs/.../params.json         (FRSR fit; contains ε, α, Ω’s…)
  runs/.../frsr_ladder.csv     (optional diagnostics)
  runs/.../frsr_background.ini (CLASS background: Ω_fld, w0_fld, wa_fld, cs2_fld, use_ppf)
  runs/.../plots/*.png         (diagnostics)
                   |
                   | feed CPL params only
                   v
+-----------------------------------+
|               CLASS               |
|   (v3.3.3; **CPL**: w0_fld, wa_fld)|
+------------------+----------------+
                   | produces
                   v
  class_public/output/<root>_*  (C_ℓ, P(k), d_A, H(z), etc.)
                   |
                   | consumed by
                   v
+-----------------------------------+
|            MontePython            |
|        (MCMC over CPL+base)       |
|  uses .param (w0_fld, wa_fld, …)  |
+-----------------------------------+
```

### MontePython MCMC pipeline (sampling ε, α)
```
                           (parallel chains)
┌─────────────────────────────────────────────────────────────────────────┐
│                             MontePython                                 │
│                     (sampler / MCMC controller)                         │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ proposes new draw of cosmological params
                │   sample (ε, α) then map → (w0_fld, wa_fld)
                ▼
        ┌───────────────────────────────┐
        │ frsr_mp_hook.py (helper)      │
        │  • w0_fld = -1 + ε            │
        │  • wa_fld = -(α·ε)            │
        │  • emits CPL params           │
        └───────────────┬───────────────┘
                        │ passes CPL args to CLASS
                        ▼
               ┌─────────────────────────┐
               │ CLASS (v3.3.3)          │
               │  • uses CPL: w0_fld,    │
               │    wa_fld, Omega_fld,   │
               │    cs2_fld, use_ppf     │
               │  • returns theory (H(z),│
               │    distances, C_ℓ, …)   │
               └──────────────┬──────────┘
                              │
                              ▼
             ┌───────────────────────────────────┐
             │ MontePython likelihood(s)         │
             │  • Planck/SNe/BAO/WL…             │
             │  • compare CLASS outputs to data  │
             │  • return logL                    │
             └───────────────────────────────────┘
                              │
                              └── accept / reject → iterate
```

---

## Notes & Policy
- Runtime path is **CPL → CLASS**; no tabulated `w(a)`.
- Keep `cs2_fld=1` and `use_ppf=yes` unless you know you need perturbation variants.
- Spectral kernels are calibration‑time dials; at runtime we only use `(w0, wa)`.