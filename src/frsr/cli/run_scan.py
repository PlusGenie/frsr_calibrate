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

from __future__ import annotations

import os
import sys
from typing import Sequence, List, Tuple, Optional

from .frsr_calibrator import build_parser
from frsr.core.frsr_background import (
    compute_C_FRSR, adjust_beta, solve_kIR_for_E0_target,
    build_ladder, sane_kir, within_unit_interval, spread_metric,
    kmps_per_Mpc_to_Hz, c
)
from frsr.io.frsr_loader import make_run_dir, dump_env, dump_command, dump_params, write_csv
from frsr.io.frsr_plotting import maybe_plot
from frsr.io.frsr_export_cpl import write_cpl_background_ini
from frsr.core.frsr_constraints_bridge import run_constraints_and_write


def _parse_list(arg: str, cast):
    return [cast(x.strip()) for x in arg.split(",") if x.strip()]


def _evaluate_candidate(
    N: int, r: float, alpha: float, beta: float,
    C_FRSR: float, E0_target: float, q_cap: float
) -> Optional[Tuple[dict, List[Tuple[int, float, float, float, float]]]]:
    """
    Try to build a valid ladder for the candidate; return (summary, rows) or None.
    The summary dict includes: N,r,alpha,beta,k_IR,E0,q,spread,C_FRSR,sum_terms,rel_err.
    """
    # Ensure q<1 by minimally increasing beta if needed
    beta_eff, q = adjust_beta(r, beta, q_cap)

    # In scan mode we always solve k_IR to hit a common E0_target (auto-kIR behavior)
    try:
        k_IR, q = solve_kIR_for_E0_target(C_FRSR, N, r, alpha, beta_eff, E0_target)
    except Exception:
        return None
    if not sane_kir(k_IR):
        return None

    E0 = E0_target
    rows = build_ladder(N, r, alpha, beta_eff, k_IR, E0)
    Ei_list = [row[3] for row in rows]
    terms = [row[4] for row in rows]

    if not within_unit_interval(Ei_list):
        return None

    lhs_sum = sum(terms)
    rel_err = abs(lhs_sum - C_FRSR) / max(abs(C_FRSR), 1e-300)

    # Spread is our balance metric (lower is better)
    sp = spread_metric(terms)

    summary = dict(
        N=N, r=r, alpha=alpha, beta=beta_eff,
        k_IR=k_IR, E0=E0, q=q,
        spread=sp, C_FRSR=C_FRSR, sum_terms=lhs_sum, rel_err=rel_err
    )
    return summary, rows


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run a grid scan over (N, r, alpha, beta), select the most balanced ladder,
    and generate artifacts similar to the single-run flow.
    """
    ap = build_parser()
    args = ap.parse_args(argv)

    if not getattr(args, "scan", False):
        print("[error] run_scan requires --scan (use python -m frsr.cli.run_calibrate for single runs).", file=sys.stderr)
        return 2

    # Build grids
    Ns      = _parse_list(args.Ns, int)
    rs      = _parse_list(args.rs, float)
    alphas  = _parse_list(args.alphas, float)
    betas   = _parse_list(args.betas, float)

    if not (Ns and rs and alphas and betas):
        print("[scan] One or more grids are empty. Check --Ns/--rs/--alphas/--betas.", file=sys.stderr)
        return 2

    # Fixed cosmology -> target finite residual constant
    C_FRSR = compute_C_FRSR(args.Lambda_obs, args.lambda_bare, args.kappa)

    best: Optional[Tuple[dict, List[Tuple[int, float, float, float, float]]]] = None

    # Iterate candidates
    for N in Ns:
        for r in rs:
            for alpha in alphas:
                for beta in betas:
                    cand = _evaluate_candidate(
                        N, r, alpha, beta,
                        C_FRSR=C_FRSR,
                        E0_target=getattr(args, "E0_target_scan", 0.1),
                        q_cap=args.q_cap
                    )
                    if cand is None:
                        continue
                    summary, rows = cand
                    if best is None:
                        best = (summary, rows)
                        continue
                    # Pick by smaller spread; tie-break by smaller rel_err
                    if summary["spread"] < best[0]["spread"] - 1e-12 or (
                        abs(summary["spread"] - best[0]["spread"]) <= 1e-12
                        and summary["rel_err"] < best[0]["rel_err"]
                    ):
                        best = (summary, rows)

    if best is None:
        print("[scan] No acceptable calibration found with the given ranges.", file=sys.stderr)
        return 2

    summary, rows = best

    # A2 quick bound for the best candidate
    try:
        q_best = summary.get("q")
        N_best = summary.get("N")
        if q_best is not None and N_best is not None:
            if q_best >= 1.0:
                tail = float("inf")
            else:
                try:
                    tail = (q_best ** N_best) / max(1.0 - q_best, 1e-30)
                except OverflowError:
                    tail = float("inf")
            print(f"[A2 quick] q = {q_best:.6f}, N = {N_best}, tail_inf≈q^N/(1-q) = {tail:.4e}")
    except Exception:
        pass

    # Make run dir and write basic artifacts
    run_dir = make_run_dir("scan", args)
    os.makedirs(os.path.join(run_dir, "variants"), exist_ok=True)

    # Console report
    print("=" * 76)
    print("FRSR Grid Scan — Best Balanced Ladder")
    print("-" * 76)
    for k in ["N", "r", "alpha", "beta", "k_IR", "E0", "q", "spread", "C_FRSR", "sum_terms", "rel_err"]:
        v = summary[k]
        if isinstance(v, float):
            if k in ("C_FRSR", "sum_terms"):
                print(f"{k:12s}: {v:.6e}")
            else:
                print(f"{k:12s}: {v:.6g}")
        else:
            print(f"{k:12s}: {v}")
    print("-" * 76)
    print(f"{'i':>2}  {'k_i [m^-1]':>12}  {'Δk_i [m^-1]':>12}  {'𝔈_i':>12}  {'term_i [m^-4]':>14}")
    for i, k_i, dk_i, Ei, term_i in rows:
        print(f"{i:2d}  {k_i: .6e}  {dk_i: .6e}  {Ei: .6e}  {term_i: .6e}")

    # CSV + plots
    if args.csv:
        csv_path = os.path.join(run_dir, args.csv)
        write_csv(rows, csv_path)
    if args.plot:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            maybe_plot(rows, args.plot, run_dir, args.logy)
        except Exception:
            print("[warn] matplotlib not available; skipping plot.", file=sys.stderr)

    # Params file for reproducibility and downstream tools
    params = dict(
        mode="scan",
        summary=summary,
        explanations=dict(
            Lambda_obs="Observed cosmological constant (m^-2).",
            lambda_bare="Bare curvature term (m^-2).",
            kappa="Global coupling (dimensionless).",
            N="Number of finite negative-energy bands.",
            r="Geometric spacing between bands; k_i = k_IR r^(i-1).",
            alpha="Fractional bandwidth; Δk_i = α k_i.",
            beta="Entanglement decay per band; 𝔈_i = 𝔈0 e^{-β(i-1)}.",
            q="Convergence factor q = r^4 e^{-β} (<1).",
            k_IR="IR pivot momentum (m^-1).",
            E0="Base entanglement weight at i=1.",
            C_FRSR="Finite residual constant; Σ 𝔈_i k_i^3 Δk_i (m^-4).",
        )
    )
    dump_env(run_dir)
    dump_command(run_dir, list(argv) if argv is not None else [])
    dump_params(run_dir, params)

    # CLASS-facing pieces
    Lambda0 = args.Lambda_obs - args.lambda_bare
    H0_Hz   = kmps_per_Mpc_to_Hz(getattr(args, "H0_km_s_Mpc", 67.4))
    Omega_de0 = (Lambda0 * (c**2)) / (3.0 * (H0_Hz**2))

    # Optional: generate CLASS artifacts for the best candidate
    if getattr(args, "scan_generate_background", True):
        try:
            # Emit a minimal CPL background for CLASS (no tabulated w)
            bg_out = os.path.join(run_dir, "frsr_background.cpl.ini")
            # Defaults here are placeholders; tune via CLI/parser if desired.
            h_val = getattr(args, "H0_km_s_Mpc", 67.4) / 100.0  # convert H0 to h if provided
            write_cpl_background_ini(
                out_path=bg_out,
                Omega_de0=Omega_de0,
                h=h_val,
                omega_b=getattr(args, "omega_b", 0.0224),
                omega_cdm=getattr(args, "omega_cdm", 0.119),
                w0=getattr(args, "w0", -1.0),
                wa=getattr(args, "wa", 0.0),
            )
        except Exception as e:
            print(f"[warn] [scan] CLASS artifact generation failed: {e}", file=sys.stderr)

    # Constraints
    try:
        ok_constraints, constraints_report = run_constraints_and_write(
            run_dir, rows,
            N=summary["N"], r=summary["r"], alpha=summary["alpha"], beta_used=summary["beta"],
            k_IR=summary["k_IR"], E0=summary["E0"], q=summary["q"], Omega_de0=Omega_de0,
            options=dict(
                ede_max=0.03,
                A2_rel_tol=getattr(args, "A2_rel_tol", 1e-2),
                target_cum_frac=getattr(args, "prune_target_cum", 0.99),
                min_band_frac=getattr(args, "prune_min_band", 5e-4),
            ),
        )
        print(
            f"[constraints] pass={ok_constraints} "
            f"A1={constraints_report.get('A1_convergence', {}).get('pass', '?')} "
            f"A2(auto)={constraints_report.get('A2_N_insensitivity_auto', {}).get('pass', 'skipped')} "
            f"B2(EDE)={constraints_report.get('B2_EDE_bound', {}).get('pass', 'skipped')}"
        )
    except Exception as e:
        print(f"[warn] [scan] Constraints evaluation failed: {e}", file=sys.stderr)

    print(f"[run dir] {run_dir}")
    print("=" * 76)
    print("Done.")
    return 0
  
