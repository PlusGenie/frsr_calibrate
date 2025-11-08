#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2025 Sangwook Lee @ Plusgenie Limited
# This file is part of FRSR (Finite Residual Rule of Sum).
#
# FRSR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FRSR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FRSR.  If not, see <https://www.gnu.org/licenses/>.

# Author: Sangwook Lee (aladdin@plusgenie.com)
# Date: 2025-10-27

"""
FRSR sensitivity explorer: Ω_de,0 as a function of (N, r, α, β, 𝔈0, k_IR)

Purpose
-------
This standalone tool lets you *vary the ladder parameters* and directly
compute the *implied* dark-energy density today,
    Ω_de,0 = (Λ_model c^2) / (3 H0^2),
where Λ_model is obtained by inverting the FRSR master relation

    Σ_i 𝔈_i k_i^3 Δk_i  =  𝓒_FRSR_model
    Λ_model  =  λ_bare + κ * (2 G ħ / (π c^3)) * 𝓒_FRSR_model.

This exposes how Ω_de,0 shifts when you change (N, r, α, β, 𝔈0, k_IR).
It is *deliberately decoupled* from the previous calibrator logic so you can
build intuition without auto-forcing the sum to match Λ_obs.

New in this update
------------------
• Added a two-component (“slow–fast”) bi-exponential decay model:
  𝔈_i = E0[(1−η) e^{−β₁(i−1)} + η e^{−β₂(i−1)}], with 0<η<1 and β₂>β₁.
• CLI flags: --Ei_model bi_exp, plus --beta1, --beta2, --eta.

Outputs
-------
- A timestamped run directory under a base output directory (default: `runs/diagnostics/`), containing:
  - `summary.txt`         : text summary of inputs and results
  - `sweep.csv`           : table of parameter value vs Λ_model, Ω_de,0, etc. (if sweeping)
  - `omega_vs_param.png`  : plot (if sweeping 1-D)
  - `omega_vs_<param>__<outer>=<value>.png` : plots for each value of outer sweep (if 2-D sweep)
  - `point.csv`           : single-point evaluation (if NOT sweeping)

CLI Features
------------
1) `--out` lets you choose a base output directory. Results are written to a timestamped subfolder under this base. By default, this is `runs/diagnostics/`.
2) 2‑D parameter sweep: Use `--also_sweep <param> --list a,b,c` to perform a sweep over two parameters. The script will loop over the comma-separated list for the outer parameter, and for each, sweep the primary parameter with `--sweep`.

Example (2‑D sweep and custom output base):
    python src/frsr_cal_dark_energy_de_today.py \
      --Ei_model tempered_power \
      --N 6 --r 3.0 --alpha 0.05 --beta 4.62 \
      --E0 0.1 --kIR 7.76e4 --gamma_temper 1.1 \
      --sweep n_power --start 4.0 --stop 6.0 --num 11 \
      --also_sweep k_star --list 2e5,3e5,5e5,7e5,1e6 \
      --out runs/diagnostics/omega_fld_explorer

Other usage:
(1) Single-point evaluation (default ansatz: exp by band index):
    python src/frsr_cal_dark_energy_de_today.py \
      --N 6 --r 3.0 --alpha 0.05 --beta 4.62 --E0 0.1 --kIR 7.76e4 \
      --H0_km_s_Mpc 67.4 --kappa 1.0 --lambda_bare 0.0 --Lambda_obs 1.11e-52

(2) Choose a different ansatz for 𝔈_i (weights):
    # power law in scale
    --Ei_model power --n_power 1.0

    # mixed (exp in i) × (power in k)
    --Ei_model mixed --beta 4.0 --n_power 0.5

    # exponential in k/k_IR
    --Ei_model expk --gamma_expk 0.8

    # log-normal around k_*=kIR (default)
    --Ei_model lognorm --sigma_lognorm 1.0 [--k_star <value>]

    # tempered power law in k with smooth UV cutoff
    --Ei_model tempered_power --n_power 3.5 --gamma_temper 2.0 --k_star 1.0e6

(3) Sweep one parameter (e.g., β from 3 to 7 in 41 steps):
    python src/frsr_cal_dark_energy_de_today.py \
      --N 6 --r 3.0 --alpha 0.05 --beta 4.62 --E0 0.1 --kIR 7.76e4 \
      --H0_km_s_Mpc 67.4 --kappa 1.0 --lambda_bare 0.0 --Lambda_obs 1.11e-52 \
      --Ei_model mixed --n_power 0.5 \
      --sweep beta --start 3.0 --stop 7.0 --num 41

Notes on decay/weighting models
-------------------------------
We support several physically motivated ansätze for the per-band entanglement
weights 𝔈_i. Each produces a distinct pattern for how power is distributed across
the finite ladder:

• exp (default):           𝔈_i = E0 · e^{−β(i−1)}
  – Single exponential in band index. Controlled by β; larger β damps the UV faster.

• bi_exp (slow–fast):      𝔈_i = E0[(1−η) e^{−β₁(i−1)} + η e^{−β₂(i−1)}]
  – Mixture of two exponentials with β₂>β₁. η is the fast fraction (0<η<1).
    Useful when you need a mild IR tail plus a steeper UV fall-off simultaneously.

• power:                   𝔈_i = E0 · (k_i/k_IR)^{−n}
  – Scale-free fall-off in k. Heavier IR weighting than pure exp for small n.

• mixed:                   𝔈_i = E0 · e^{−β(i−1)} · (k_i/k_IR)^{−n}
  – Combines exp in i with a scale power law in k.

• expk / lognorm / tempered_power:
  – Decay directly in k/k_IR (expk), a log-normal peak (lognorm), or a tempered UV cutoff.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from datetime import datetime
from typing import List

# Optional symbolic math
try:
    import sympy as sp
    _HAVE_SYMPY = True
except Exception:
    _HAVE_SYMPY = False

# ---------------------------
# Physical constants (SI)
# ---------------------------
G = 6.674_30e-11          # m^3 kg^-1 s^-2
c = 2.997_924_58e8        # m s^-1
hbar = 1.054_571_817e-34  # J s

DEFAULT_LAMBDA_OBS = 1.11e-52  # m^-2 (reference only)


# ---------------------------
# Helpers
# ---------------------------
def entanglement_weight(i: int,
                        k_i: float,
                        k_IR: float,
                        E0: float,
                        model: str,
                        beta: float,
                        beta1: float,
                        beta2: float,
                        eta: float,
                        n_power: float,
                        gamma_expk: float,
                        gamma_temper: float,
                        k_star: float | None,
                        sigma_lognorm: float) -> float:
    """
    Return 𝔈_i for the requested ansatz.

    Models:
      - "exp"     : 𝔈_i = E0 * exp(-β (i-1))
      - "power"   : 𝔈_i = E0 * (k_i/k_IR)^(-n_power)
      - "mixed"   : 𝔈_i = E0 * exp(-β (i-1)) * (k_i/k_IR)^(-n_power)
      - "expk"    : 𝔈_i = E0 * exp(-gamma_expk * (k_i/k_IR))
      - "lognorm" : 𝔈_i = E0 * exp( - [ln(k_i/k_*)]^2 / (2 σ^2) )
      - "tempered_power" :
                    𝔈_i = E0 * (k_i/k_IR)^(-n_power) * exp( - (k_i/k_*)^(gamma_temper) )
      - "bi_exp"  : 𝔈_i = E0 * [ (1-η)·exp(-β1(i-1)) + η·exp(-β2(i-1)) ],
                    with 0 < η < 1 and β2 > β1 (slow–fast mixture)
    """
    if model == "exp":
        return E0 * math.exp(-beta * (i - 1))
    elif model == "bi_exp":
        eta_clipped = min(max(eta, 0.0), 1.0)
        b1 = float(beta1)
        b2 = float(beta2)
        return E0 * ((1.0 - eta_clipped) * math.exp(-b1 * (i - 1)) + eta_clipped * math.exp(-b2 * (i - 1)))
    elif model == "power":
        return E0 * (k_i / k_IR) ** (-n_power)
    elif model == "mixed":
        return E0 * math.exp(-beta * (i - 1)) * (k_i / k_IR) ** (-n_power)
    elif model == "expk":
        return E0 * math.exp(-gamma_expk * (k_i / k_IR))
    elif model == "lognorm":
        k0 = k_IR if (k_star is None or k_star <= 0.0) else k_star
        if sigma_lognorm <= 0.0:
            # fall back to a delta-like weight at k*=k0 (numerically: treat as exp(-∞) away from k0)
            return 0.0 if abs(k_i - k0) / max(k0, 1e-30) > 1e-6 else E0
        lnratio = math.log(k_i / k0)
        return E0 * math.exp(- (lnratio * lnratio) / (2.0 * sigma_lognorm * sigma_lognorm))
    elif model == "tempered_power":
        k0 = k_IR if (k_star is None or k_star <= 0.0) else k_star
        # IR: ~ (k/k_IR)^(-n); UV: extra exp(-(k/k0)^gamma_temper)
        return E0 * (k_i / k_IR) ** (-n_power) * math.exp(- (k_i / k0) ** max(gamma_temper, 1e-12))
    else:
        raise ValueError(f"Unknown Ei_model: {model}")

def kmps_per_Mpc_to_Hz(H_km_s_Mpc: float) -> float:
    """Convert H0 in km/s/Mpc to s^-1."""
    Mpc_m = 3.085_677_581_491_367e22
    return (H_km_s_Mpc * 1_000.0) / Mpc_m

def make_run_dir(base_dir: str | None = None, label: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = base_dir if base_dir is not None else os.path.join("runs", "diagnostics")
    name = f"{ts}_{label}" if label else f"{ts}_omega_fld_explorer"
    rd = os.path.abspath(os.path.join(base, name))
    os.makedirs(rd, exist_ok=True)
    return rd

def write_summary(run_dir: str, text: str) -> None:
    path = os.path.join(run_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(text)

def write_point_csv(run_dir: str, row: dict) -> None:
    path = os.path.join(run_dir, "point.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

def write_sweep_csv(run_dir: str, rows: List[dict]) -> str:
    path = os.path.join(run_dir, "sweep.csv")
    if not rows:
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path

# ---------------------------
# Symbolic helpers (SymPy)
# ---------------------------
def symbolic_C_FRSR_geometric(E0: float, alpha: float, k_IR: float, r: float,
                              N: int | None,
                              *,
                              model: str,
                              beta: float,
                              n_power: float) -> tuple:
    """
    Closed-form C_FRSR using SymPy for models that reduce to a geometric series
    over i (band index). Supported:
      - exp:      Ei = E0 * exp(-beta*(i-1))              -> q = r**4*exp(-beta)
      - power:    Ei = E0 * (k_i/k_IR)**(-n)              -> q = r**(4-n)
      - mixed:    Ei = E0 * exp(-beta*(i-1)) * r**(-n(i-1))-> q = r**(4-n)*exp(-beta)
    Returns (C_symbolic, q_symbolic, notes) where C_symbolic is a SymPy expression.
    If N is None, compute the N→∞ limit assuming |q|<1.
    """
    if not _HAVE_SYMPY:
        raise RuntimeError("SymPy not available; install sympy to use --symbolic.")
    i = sp.symbols('i', integer=True, positive=True)
    E0_s, alpha_s, kIR_s, r_s, beta_s, n_s = sp.symbols('E0 alpha kIR r beta n', positive=True)
    # Common front factor: E0 * alpha * k_IR**4
    front = E0_s * alpha_s * kIR_s**4
    if model == "exp":
        q = r_s**4 * sp.exp(-beta_s)
    elif model == "power":
        q = r_s**(4 - n_s)
    elif model == "mixed":
        q = r_s**(4 - n_s) * sp.exp(-beta_s)
    else:
        raise ValueError("symbolic_C_FRSR_geometric: unsupported model for closed form")
    term0 = sp.Integer(1)  # first term when i=1 is q**0
    if N is None:
        S = term0 / (1 - q)
        notes = "N→∞ limit; valid if |q| < 1."
    else:
        S = (1 - q**N) / (1 - q)
        notes = f"Finite sum with N={N}."
    C_sym = sp.simplify(front * S)
    subs_map = {E0_s: E0, alpha_s: alpha, kIR_s: k_IR, r_s: r, beta_s: beta, n_s: n_power}
    C_eval = sp.simplify(C_sym.subs(subs_map))
    q_eval = sp.simplify(q.subs(subs_map))
    return C_eval, q_eval, notes

# ---------------------------
# FRSR core
# ---------------------------
def ladder_sum_C_FRSR(N: int,
                      r: float,
                      alpha: float,
                      beta: float,
                      k_IR: float,
                      E0: float,
                      *,
                      Ei_model: str,
                      beta1: float,
                      beta2: float,
                      eta: float,
                      n_power: float,
                      gamma_expk: float,
                      gamma_temper: float,
                      k_star: float | None,
                      sigma_lognorm: float) -> float:
    """
    Compute 𝓒_FRSR = Σ_i 𝔈_i k_i^3 Δk_i  (units: m^-4)
    with k_i = k_IR r^(i-1), Δk_i = α k_i, and 𝔈_i from the chosen ansatz.
    """
    C = 0.0
    for i in range(1, N + 1):
        k_i = k_IR * (r ** (i - 1))
        dk_i = alpha * k_i
        Ei = entanglement_weight(i=i,
                                 k_i=k_i,
                                 k_IR=k_IR,
                                 E0=E0,
                                 model=Ei_model,
                                 beta=beta,
                                 beta1=beta1,
                                 beta2=beta2,
                                 eta=eta,
                                 n_power=n_power,
                                 gamma_expk=gamma_expk,
                                 gamma_temper=gamma_temper,
                                 k_star=k_star,
                                 sigma_lognorm=sigma_lognorm)
        C += Ei * (k_i ** 3) * dk_i
    return C

def lambda_from_C(C_FRSR_model: float, kappa: float, lambda_bare: float) -> float:
    """
    Invert master relation to get Λ_model from 𝓒_FRSR_model:
      Λ_model = λ_bare + κ * (2 G ħ / (π c^3)) * 𝓒_FRSR_model
    """
    return lambda_bare + kappa * (2.0 * G * hbar / (math.pi * (c ** 3))) * C_FRSR_model

def omega_de0_from_lambda(Lambda_model: float, H0_Hz: float) -> float:
    """Ω_de,0 = (Λ c^2)/(3 H0^2)."""
    return (Lambda_model * (c ** 2)) / (3.0 * (H0_Hz ** 2))


# ---------------------------
# Plotting
# ---------------------------
def plot_sweep(
    run_dir: str,
    param_name: str,
    xs: List[float],
    omegas: List[float],
    target_omega: float | None,
    *,
    out_name: str | None = None,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available ({e}); skipping plot.")
        return

    plt.figure()
    plt.plot(xs, omegas, marker="o", linewidth=1)
    plt.xlabel(param_name)
    plt.ylabel(r"$\Omega_{\rm de,0}$")
    plt.title(rf"FRSR sensitivity: $\Omega_{{\rm de,0}}$ vs {param_name}")
    plt.grid(True)
    if target_omega is not None:
        plt.axhline(target_omega, linestyle="--")
    if out_name is None:
        out_name = "omega_vs_param.png"
    out = os.path.join(run_dir, out_name)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"[ok] Plot saved: {out}")


# ---------------------------
# CLI
# ---------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Explore Ω_de,0 implied by FRSR ladder parameters (N, r, α, β, E0, k_IR)."
    )
    # Ladder parameters
    ap.add_argument("--N", type=int, default=6, help="Number of bands.")
    ap.add_argument("--r", type=float, default=3.0, help="Geometric spacing ratio.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Fractional bandwidth, Δk_i = α k_i.")
    ap.add_argument("--beta", type=float, default=4.62, help="Entanglement decay per band.")
    ap.add_argument("--E0", type=float, default=0.1, help="Base entanglement weight (i=1).")
    ap.add_argument("--kIR", type=float, default=7.76e4, help="IR pivot momentum (m^-1).")

    # Entanglement-weight ansatz
    ap.add_argument("--Ei_model", type=str, default="exp",
                    choices=["exp", "bi_exp", "power", "mixed", "expk", "lognorm", "tempered_power"],
                    help="Choose ansatz for band weights 𝔈_i (exp/bi_exp/power/mixed/expk/lognorm/tempered_power).")
    ap.add_argument("--beta1", type=float, default=4.0,
                    help="[bi_exp] Slow-component decay β1 (should be < β2).")
    ap.add_argument("--beta2", type=float, default=6.0,
                    help="[bi_exp] Fast-component decay β2 (should exceed β1).")
    ap.add_argument("--eta", type=float, default=0.2,
                    help="[bi_exp] Fast-component fraction η (0 < η < 1).")
    ap.add_argument("--n_power", type=float, default=0.0,
                    help="Exponent n for power-law parts (power/mixed).")
    ap.add_argument("--gamma_expk", type=float, default=0.0,
                    help="Gamma for exp-in-k model (expk).")
    ap.add_argument("--gamma_temper", type=float, default=1.0,
                    help="Gamma for tempered-power UV cutoff (tempered_power).")
    ap.add_argument("--k_star", type=float, default=None,
                    help="Preferred scale k_* for log-normal / tempered_power (defaults to kIR if omitted).")
    ap.add_argument("--sigma_lognorm", type=float, default=1.0,
                    help="Width σ for log-normal model (in ln k).")

    # Cosmology / couplings
    ap.add_argument("--kappa", type=float, default=1.0, help="Global coupling κ.")
    ap.add_argument("--lambda_bare", type=float, default=0.0, help="Bare curvature λ_bare (m^-2).")
    ap.add_argument("--Lambda_obs", type=float, default=DEFAULT_LAMBDA_OBS, help="Reference Λ_obs (m^-2) for comparison.")
    ap.add_argument("--H0_km_s_Mpc", type=float, default=67.4, help="Hubble constant H0 (km/s/Mpc).")

    # Optional sweep of a single parameter
    sweep_choices = ["N", "r", "alpha", "beta", "beta1", "beta2", "eta",
                     "E0", "kIR", "n_power", "gamma_expk", "gamma_temper", "sigma_lognorm", "k_star"]
    ap.add_argument("--sweep", type=str, default=None,
                    choices=sweep_choices,
                    help="Sweep this parameter while holding others fixed.")
    ap.add_argument("--start", type=float, default=None, help="Sweep start (inclusive). For N, value will be rounded.")
    ap.add_argument("--stop", type=float, default=None, help="Sweep stop (inclusive). For N, value will be rounded.")
    ap.add_argument("--num", type=int, default=25, help="Number of points in sweep (>=2).")

    # New CLI features
    ap.add_argument("--out", type=str, default=None, help="Base directory to write results (a timestamped subfolder will be created inside).")
    ap.add_argument("--label", type=str, default=None, help="Optional suffix for the timestamped run directory name.")
    ap.add_argument("--also_sweep", type=str, default=None,
                    choices=sweep_choices,
                    help="Optional second parameter to sweep (outer loop; requires --list).")
    ap.add_argument("--list", type=str, default=None,
                    help="Comma-separated values for also_sweep (e.g., 2e5,3e5,5e5).")

    ap.add_argument("--symbolic", action="store_true",
                    help="Also compute a SymPy closed-form C_FRSR (when supported) and report convergence (q).")
    ap.add_argument("--N_infinite", action="store_true",
                    help="When using --symbolic with a geometric model, take the N→∞ limit instead of finite N.")
    ap.add_argument("--no_plot", action="store_true",
                    help="Disable plotting (useful for headless CI).")
    return ap


def main(argv=None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    H0_Hz = kmps_per_Mpc_to_Hz(args.H0_km_s_Mpc)
    run_dir = make_run_dir(args.out, args.label)

    if args.symbolic and not _HAVE_SYMPY:
        print("[warn] --symbolic requested but SymPy is not available. Proceeding without symbolic output.")

    # Helper to compute everything at once
    def evaluate(
        N,
        r,
        alpha,
        beta,
        E0,
        kIR,
        Ei_model,
        n_power,
        gamma_expk,
        gamma_temper,
        k_star,
        sigma_lognorm,
        beta1,
        beta2,
        eta,
    ):
        C = ladder_sum_C_FRSR(N, r, alpha, beta, kIR, E0,
                              Ei_model=Ei_model,
                              beta1=beta1,
                              beta2=beta2,
                              eta=eta,
                              n_power=n_power,
                              gamma_expk=gamma_expk,
                              gamma_temper=gamma_temper,
                              k_star=k_star,
                              sigma_lognorm=sigma_lognorm)
        Lam = lambda_from_C(C, args.kappa, args.lambda_bare)
        Omega = omega_de0_from_lambda(Lam, H0_Hz)
        return C, Lam, Omega

    # Helper to set parameter by name
    def set_param_by_name(
        name,
        x,
        N,
        r,
        alpha,
        beta,
        E0,
        kIR,
        n_power,
        gamma_expk,
        gamma_temper,
        sigma_lognorm,
        k_star,
        beta1,
        beta2,
        eta,
    ):
        if name == "N": N = max(1, int(round(x)))
        elif name == "r": r = float(x)
        elif name == "alpha": alpha = float(x)
        elif name == "beta": beta = float(x)
        elif name == "beta1": beta1 = float(x)
        elif name == "beta2": beta2 = float(x)
        elif name == "eta": eta = float(x)
        elif name == "E0": E0 = float(x)
        elif name == "kIR": kIR = float(x)
        elif name == "n_power": n_power = float(x)
        elif name == "gamma_expk": gamma_expk = float(x)
        elif name == "gamma_temper": gamma_temper = float(x)
        elif name == "sigma_lognorm": sigma_lognorm = float(x)
        elif name == "k_star": k_star = float(x)
        else: raise ValueError(f"Unknown sweep param: {name}")
        return (
            N,
            r,
            alpha,
            beta,
            E0,
            kIR,
            n_power,
            gamma_expk,
            gamma_temper,
            sigma_lognorm,
            k_star,
            beta1,
            beta2,
            eta,
        )

    def linspace(a, b, n):
        if n == 1:
            return [a]
        step = (b - a) / (n - 1)
        return [a + i * step for i in range(n)]

    # If sweeping, build the grid for the chosen parameter(s)
    if args.sweep:
        if args.start is None or args.stop is None or args.num < 2:
            raise SystemExit("--sweep requires --start, --stop, and --num >= 2.")

        # 2-D sweep support
        if args.also_sweep:
            if not args.list:
                raise SystemExit("--also_sweep requires --list a,b,c")
            outer_values = [float(v) for v in args.list.split(",") if v.strip() != ""]
        else:
            outer_values = [None]

        xs = linspace(args.start, args.stop, args.num)
        all_rows: List[dict] = []
        target_omega = (args.Lambda_obs * (c**2)) / (3.0 * (H0_Hz**2))

        def _sanitize_for_filename(s: str) -> str:
            return re.sub(r"[^A-Za-z0-9_.+-]", "_", s)

        for ov in outer_values:
            # fixed baseline for each outer value
            N = args.N
            r = args.r
            alpha = args.alpha
            beta = args.beta
            E0 = args.E0
            kIR = args.kIR
            beta1 = args.beta1
            beta2 = args.beta2
            eta = args.eta
            n_power = args.n_power
            gamma_expk = args.gamma_expk
            gamma_temper = args.gamma_temper
            sigma_lognorm = args.sigma_lognorm
            k_star = args.k_star
            if args.also_sweep:
                (
                    N,
                    r,
                    alpha,
                    beta,
                    E0,
                    kIR,
                    n_power,
                    gamma_expk,
                    gamma_temper,
                    sigma_lognorm,
                    k_star,
                    beta1,
                    beta2,
                    eta,
                ) = set_param_by_name(
                    args.also_sweep,
                    ov,
                    N,
                    r,
                    alpha,
                    beta,
                    E0,
                    kIR,
                    n_power,
                    gamma_expk,
                    gamma_temper,
                    sigma_lognorm,
                    k_star,
                    beta1,
                    beta2,
                    eta,
                )
            local_omegas = []
            for x in xs:
                (
                    N1,
                    r1,
                    alpha1,
                    beta_band,
                    E01,
                    kIR1,
                    n_power1,
                    gamma_expk1,
                    gamma_temper1,
                    sigma_lognorm1,
                    k_star1,
                    beta1_slow,
                    beta2_fast,
                    eta_fast,
                ) = set_param_by_name(
                    args.sweep,
                    x,
                    N,
                    r,
                    alpha,
                    beta,
                    E0,
                    kIR,
                    n_power,
                    gamma_expk,
                    gamma_temper,
                    sigma_lognorm,
                    k_star,
                    beta1,
                    beta2,
                    eta,
                )
                C, Lam, Omega = evaluate(
                    N1,
                    r1,
                    alpha1,
                    beta_band,
                    E01,
                    kIR1,
                    args.Ei_model,
                    n_power1,
                    gamma_expk1,
                    gamma_temper1,
                    k_star1,
                    sigma_lognorm1,
                    beta1_slow,
                    beta2_fast,
                    eta_fast,
                )
                all_rows.append(dict(
                    sweep_param=args.sweep, value=float(x),
                    also_sweep_param=(args.also_sweep or ""), also_value=(ov if args.also_sweep else ""),
                    Ei_model=args.Ei_model, n_power=n_power1, gamma_expk=gamma_expk1, gamma_temper=gamma_temper1,
                    beta=beta_band, beta1=beta1_slow, beta2=beta2_fast, eta=eta_fast,
                    sigma_lognorm=sigma_lognorm1, k_star=(k_star1 if k_star1 is not None else kIR1),
                    N=N1, r=r1, alpha=alpha1, beta=beta_band, E0=E01, kIR=kIR1,
                    C_FRSR=C, Lambda_model=Lam, Omega_de0=Omega,
                    Omega_de0_ref=target_omega,
                    dOmega=Omega - target_omega, dLambda=Lam - args.Lambda_obs,
                ))
                local_omegas.append(Omega)
            # Plot for each outer value if 2-D sweep
            if args.also_sweep and not args.no_plot:
                safe_outer_val = _sanitize_for_filename(f"{ov:g}")
                out_png = f"omega_vs_{_sanitize_for_filename(args.sweep)}__{_sanitize_for_filename(args.also_sweep)}={safe_outer_val}.png"
                plot_sweep(
                    run_dir,
                    f"{args.sweep}  (model={args.Ei_model}; {args.also_sweep}={ov:g})",
                    xs,
                    local_omegas,
                    target_omega,
                    out_name=out_png,
                )
        # 1-D plot (if only 1 sweep)
        if (not args.also_sweep) and (not args.no_plot):
            omegas = [row["Omega_de0"] for row in all_rows]
            plot_sweep(run_dir, f"{args.sweep}  (model={args.Ei_model})", xs, omegas, target_omega)

        # Optional symbolic report (only once per sweep, using baseline values)
        if args.symbolic and _HAVE_SYMPY and args.Ei_model in ("exp", "power", "mixed"):
            try:
                N_sym = None if args.N_infinite else args.N
                C_sym, q_sym, sym_note = symbolic_C_FRSR_geometric(
                    E0=args.E0, alpha=args.alpha, k_IR=args.kIR, r=args.r, N=N_sym,
                    model=args.Ei_model, beta=args.beta, n_power=args.n_power
                )
                sym_txt = f"""[symbolic]
model={args.Ei_model}  ({sym_note})
q (convergence factor) = {sp.N(q_sym):.6g}
criterion (|q|<1 for N→∞) -> {abs(float(sp.N(q_sym))) < 1.0 if N_sym is None else 'N/A (finite N)'}
C_FRSR (closed form)   = {sp.N(C_sym):.6e}  (m^-4)
front = E0*alpha*k_IR^4 = {args.E0*args.alpha*(args.kIR**4):.6e}
"""
                with open(os.path.join(run_dir, "symbolic.txt"), "w") as f:
                    f.write(sym_txt)
                print("[ok] Symbolic summary written:", os.path.join(run_dir, "symbolic.txt"))
            except Exception as e:
                print(f"[warn] symbolic computation failed: {e}")

        csv_path = write_sweep_csv(run_dir, all_rows)
        print(f"[ok] Sweep table saved: {csv_path}")

        # Summary text
        sweep_info = f"Sweeping: {args.sweep} from {args.start} to {args.stop} ({args.num} steps)"
        if args.also_sweep:
            sweep_info += f"\nAlso sweeping: {args.also_sweep} over {args.list}"
        write_summary(run_dir, f"""FRSR Ω_de,0 sensitivity — parameter sweep
Base output dir: {args.out or 'runs/diagnostics'}
Run dir   : {run_dir}
H0        : {args.H0_km_s_Mpc} km/s/Mpc
kappa     : {args.kappa}
lambda_bare: {args.lambda_bare:.6e} m^-2
Lambda_obs: {args.Lambda_obs:.6e} m^-2
Reference Ω_de,0 from Λ_obs: {target_omega:.6e}

{sweep_info}

Fixed (non-swept) baseline:
  N={args.N}, r={args.r}, α={args.alpha}, β={args.beta}, 𝔈0={args.E0}, k_IR={args.kIR:.6e} m^-1
  ansatz={args.Ei_model}, n_power={args.n_power}, gamma_expk={args.gamma_expk}, gamma_temper={args.gamma_temper}, sigma_lognorm={args.sigma_lognorm}, k_*={(args.k_star if args.k_star is not None else args.kIR):.6e} m^-1

Files:
  - sweep.csv
  - omega_vs_param.png (if 1-D sweep)
  - omega_vs_<param>__<outer>=<value>.png (if 2-D sweep)
""")
        print(f"[run dir] {run_dir}")
        return 0

    # Otherwise: single-point evaluation only
    C, Lam, Omega = evaluate(
        args.N, args.r, args.alpha, args.beta, args.E0, args.kIR,
        args.Ei_model, args.n_power, args.gamma_expk, args.gamma_temper, args.k_star, args.sigma_lognorm,
        args.beta1, args.beta2, args.eta,
    )
    ref_Omega = (args.Lambda_obs * (c**2)) / (3.0 * (H0_Hz**2))

    symbolic_note = ""
    if args.symbolic and _HAVE_SYMPY and args.Ei_model in ("exp", "power", "mixed"):
        try:
            N_sym = None if args.N_infinite else args.N
            C_sym, q_sym, sym_note = symbolic_C_FRSR_geometric(
                E0=args.E0, alpha=args.alpha, k_IR=args.kIR, r=args.r, N=N_sym,
                model=args.Ei_model, beta=args.beta, n_power=args.n_power
            )
            symbolic_note = f"""
[symbolic]
model={args.Ei_model}  ({sym_note})
q (convergence factor) = {sp.N(q_sym):.6g}
criterion (|q|<1 for N→∞) -> {abs(float(sp.N(q_sym))) < 1.0 if N_sym is None else 'N/A (finite N)'}
C_FRSR (closed form)   = {sp.N(C_sym):.6e}  (m^-4)
front = E0*alpha*k_IR^4 = {args.E0*args.alpha*(args.kIR**4):.6e}
"""
            with open(os.path.join(run_dir, "symbolic.txt"), "w") as f:
                f.write(symbolic_note)
            print("[ok] Symbolic summary written:", os.path.join(run_dir, "symbolic.txt"))
        except Exception as e:
            symbolic_note = f"\n[symbolic] failed: {e}\n"

    write_point_csv(run_dir, dict(
        N=args.N, r=args.r, alpha=args.alpha, beta=args.beta, E0=args.E0, kIR=args.kIR,
        beta1=args.beta1, beta2=args.beta2, eta=args.eta,
        kappa=args.kappa, lambda_bare=args.lambda_bare, H0_km_s_Mpc=args.H0_km_s_Mpc,
        Ei_model=args.Ei_model, n_power=args.n_power, gamma_expk=args.gamma_expk, gamma_temper=args.gamma_temper,
        k_star=(args.k_star if args.k_star is not None else args.kIR),
        sigma_lognorm=args.sigma_lognorm,
        C_FRSR=C, Lambda_model=Lam, Omega_de0=Omega,
        Omega_de0_ref=ref_Omega, dOmega=Omega - ref_Omega, dLambda=Lam - args.Lambda_obs
    ))

    write_summary(run_dir, f"""FRSR Ω_de,0 at a single point
Run dir   : {run_dir}
H0        : {args.H0_km_s_Mpc} km/s/Mpc
kappa     : {args.kappa}
lambda_bare: {args.lambda_bare:.6e} m^-2
Lambda_obs: {args.Lambda_obs:.6e} m^-2

Inputs:
  N={args.N}, r={args.r}, α={args.alpha}, β={args.beta}, 𝔈0={args.E0}, k_IR={args.kIR:.6e} m^-1

Ansatz:
  model={args.Ei_model}
  n_power={args.n_power}, gamma_expk={args.gamma_expk}, gamma_temper={args.gamma_temper}, sigma_lognorm={args.sigma_lognorm}, k_*={(args.k_star if args.k_star is not None else args.kIR):.6e} m^-1

Results:
  𝓒_FRSR (model)      = {C:.6e} m^-4
  Λ_model              = {Lam:.6e} m^-2
  Ω_de,0 (from model)  = {Omega:.6e}
  Ω_de,0 (from Λ_obs)  = {ref_Omega:.6e}
  ΔΩ = model - ref     = {Omega - ref_Omega:.6e}
  ΔΛ = model - Λ_obs   = {Lam - args.Lambda_obs:.6e}
{symbolic_note}
Interpretation tip:
- If ΔΩ ≈ 0, your chosen ladder reproduces the observed Λ (for the given κ and λ_bare).
- If ΔΩ ≠ 0, this shows how much your ladder would *want* dark energy to differ.
""")

    print(f"[ok] Single-point results written under: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
