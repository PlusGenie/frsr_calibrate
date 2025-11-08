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
import argparse

DEFAULT_LAMBDA_OBS = 1.11e-52  # m^-2

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Calibrate a finite-band ladder for FRSR to match Λ_obs."
    )
    # Cosmology / constants
    ap.add_argument("--Lambda_obs", type=float, default=DEFAULT_LAMBDA_OBS, help="Observed Λ (m^-2).")
    ap.add_argument("--lambda_bare", type=float, default=0.0, help="Bare geometric term λ_bare (m^-2).")
    ap.add_argument("--kappa", type=float, default=1.0, help="Global coupling κ (dimensionless).")

    # Ladder parameters
    ap.add_argument("--N", type=int, default=6, help="Number of finite negative-energy bands.")
    ap.add_argument("--r", type=float, default=3.0, help="Geometric spacing ratio between bands.")
    ap.add_argument("--alpha", type=float, default=0.1, help="Fractional width Δk_i = α k_i.")
    ap.add_argument("--beta", type=float, default=0.7, help="Entanglement decay rate per band.")

    # Auto-control of q = r^4 e^{-β}
    ap.add_argument("--q_cap", type=float, default=0.8, help="If q>=1, increase β so that q=q_cap (<1).")

    # k_IR options
    ap.add_argument("--kIR", type=float, default=None, help="Set k_IR directly (m^-1).")
    ap.add_argument("--kIR_from_H0", type=float, default=70.0,
                    help="If kIR not provided and --auto_kIR is off: use H0 (km/s/Mpc) with k_IR = H0/c.")
    ap.add_argument("--auto_kIR", action="store_true", help="Solve k_IR so that 𝔈0 hits --E0_target.")
    ap.add_argument("--E0_target", type=float, default=0.1, help="Desired 𝔈0 (0<E0_target≤1) when --auto_kIR.")

    # Output
    ap.add_argument("--csv", type=str, default="frsr_ladder.csv", help="CSV output filename (written in run dir).")
    ap.add_argument("--plot", type=str, default="frsr_ladder.png", help="Base PNG name for plots (written in run dir).")
    ap.add_argument("--outdir", type=str, default=None, help="Write all outputs to this folder. If omitted, a run folder is created under runs/{calibrate|scan}/.")
    ap.add_argument("--deterministic", action="store_true", help="Use a hash-based slug for the run folder instead of a timestamp.")
    ap.add_argument("--logy", action="store_true", help="Use log-scale on the per-band contribution plot.")

    # Symbolic reporting
    ap.add_argument("--symbolic", action="store_true",
                    help="Use SymPy to print closed-form expressions and export LaTeX for FRSR sums.")
    ap.add_argument("--symbolic_tex", type=str, default=None,
                    help="Path to write LaTeX equations; defaults to <run_dir>/symbolic.tex when --symbolic is set.")

    # CLASS hookup options
    ap.add_argument("--H0_km_s_Mpc", type=float, default=67.4,
                    help="H0 used to compute Ω_de,0 = (Λ0 c^2)/(3 H0^2) for CLASS (km/s/Mpc).")
    ap.add_argument("--emit_class_snippet", action="store_true", default=True,
                    help="Write a class_snippet.ini with Omega_fld and w-table settings in the run directory. (default: on)")

    # Scan mode
    ap.add_argument("--scan", action="store_true", help="Grid-scan over N,r,alpha,beta to find balanced ladders.")
    ap.add_argument("--Ns", type=str, default="6,7,8", help="Comma list for N in scan.")
    ap.add_argument("--rs", type=str, default="2.5,3.0,3.5", help="Comma list for r in scan.")
    ap.add_argument("--alphas", type=str, default="0.05,0.1,0.2", help="Comma list for alpha in scan.")
    ap.add_argument("--betas", type=str, default="4.0,5.0,6.0", help="Comma list for beta in scan.")
    ap.add_argument("--E0_target_scan", type=float, default=0.1, help="E0 target used in scan auto-kIR.")
    ap.add_argument("--max_spread", type=float, default=0.8, help="Accept solutions with log-spread <= this.")
    ap.add_argument("--scan_generate_background", action="store_true", default=True,
                    help="[scan mode] Also generate w_of_a_frsr.txt, class_snippet.ini, and frsr_background.ini for the best candidate (default: on).")

    # Constraints & pruning
    ap.add_argument("--A2_rel_tol", type=float, default=1e-2,
                    help="Tolerance for the automatic N→2N stability check (constraint A2).")
    ap.add_argument("--auto_prune_bands", action="store_true",
                    help="Prune negligible high-index bands using the D2 suggestion before reporting outputs.")
    ap.add_argument("--prune_target_cum", type=float, default=0.99,
                    help="Pruning target cumulative fraction; keep bands until cumulative ≥ this value.")
    ap.add_argument("--prune_min_band", type=float, default=5e-4,
                    help="Always keep bands whose individual fraction exceeds this threshold during pruning.")

    return ap


# --- CLI Entrypoint ---
def main(argv: list[str] | None = None) -> int:
    """
    Minimal CLI entrypoint for the fast-ladder calibrator.
    Currently supports --help and argument parsing only.
    Future versions may dispatch to subcommands (lock/export/scan).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # If invoked without any actionable flags, just show help.
    # (Keeps the console script functional while we wire full commands.)
    if argv is None or len(argv) == 0:
        parser.print_help()
        return 0

    # Placeholder: successful parse is considered a no-op for now.
    # Integrate with lock/export/scan dispatch here later.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
