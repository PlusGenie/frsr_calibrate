#!/usr/bin/env python3

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
frsr_make_class_background.py  (CPL-only)

Purpose
-------
Create a minimal CLASS background .ini for **CPL dark energy only**
we drive CLASS with (w0_fld, wa_fld) exclusively.

Usage
-----
Examples:
  # 1) From explicit CPL parameters
  python frsr_make_class_background.py \
    --w0 -0.98 --wa 0.05 \
    --h 0.674 --omega-b 0.048 --omega-cdm 0.262 --omega-k 0.0 \
    --omega-fld 0.697 \
    --out frsr_background.ini

  # 2) From FRSR parameters (ε, α) mapped to CPL: w0=-1+ε, wa=-(α·ε)
  python frsr_make_class_background.py \
    --epsilon 0.02 --alpha 0.5 \
    --h 0.674 --omega-b 0.048 --omega-cdm 0.262 \
    --omega-fld 0.697 \
    --out frsr_background.ini

What it does
------------
- Resolves Omega_fld, h from params.json when available (optional).
- Writes a minimal, CLASS-compatible ini with analytic CPL knobs:
    * w0_fld, wa_fld
    * No tabulated w(a); `use_tabulated_w` is always `no` and no `w_table` key.
- **Important (CLASS constraint):** CLASS requires either **Omega_fld** *or*
  **Omega_Lambda**, but **not both**. We only emit `Omega_fld`.

Notes
-----
- If H0 is present in params.json, we compute h = H0/100.
- If Omega_fld is absent, we try to compute it from Lambda0 & H0 via
  Omega_fld = Lambda0*c^2/(3 H0^2). If still absent, we fall back to 0.697 with a warning.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional
import re

# --- Add FRSR CPL mapping imports here ---
from frsr.core.frsr_background import map_kernel_to_cpl, SpectralKnobs, Kernel

from frsr.utils.log import init_logging, get_logger

C = 299792458.0  # m/s


def _abs(path: Optional[str]) -> Optional[str]:
    return os.path.abspath(path) if path else None


def _read_params(params_path: Optional[str]) -> dict:
    if not params_path:
        return {}
    try:
        with open(params_path, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not read params.json: {e}")
        return {}


def _infer_h(params: dict, h_cli: Optional[float]) -> Optional[float]:
    if h_cli is not None:
        return float(h_cli)
    # try various keys
    H0 = (
        params.get("class_params", {}).get("H0_km_s_Mpc")
        or params.get("inputs", {}).get("H0_km_s_Mpc")
        or params.get("H0_km_s_Mpc")
    )
    if H0 is not None:
        try:
            return float(H0) / 100.0
        except Exception:
            pass
    return None


def _infer_Omega_fld(params: dict, omega_fld_cli: Optional[float], h: Optional[float]) -> Optional[float]:
    if omega_fld_cli is not None:
        return float(omega_fld_cli)
    # direct value in params?
    for scope in ("class_params", "derived", "inputs"):
        v = params.get(scope, {}).get("Omega_fld")
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    # compute from Lambda0 and H0 if available
    Lambda0 = (
        params.get("class_params", {}).get("Lambda0_m^-2")
        or params.get("inputs", {}).get("Lambda0_m^-2")
        or params.get("Lambda0_m^-2")
    )
    H0_km_s_Mpc = (
        params.get("class_params", {}).get("H0_km_s_Mpc")
        or params.get("inputs", {}).get("H0_km_s_Mpc")
        or params.get("H0_km_s_Mpc")
    )
    if Lambda0 is not None and H0_km_s_Mpc is not None:
        try:
            Lambda0 = float(Lambda0)
            H0 = float(H0_km_s_Mpc) * 1000.0 / (3.085677581491367e22)  # s^-1
            return (Lambda0 * (C**2)) / (3.0 * (H0**2))
        except Exception:
            pass
    return None


# --- Insert helper to infer kernel spec from params dict ---
def _infer_kernel_spec(params: dict) -> Optional[SpectralKnobs]:
    """
    Try to build a SpectralKnobs from params.json structure.

    Expected shapes (any of these):
      params["frsr.model"] = {"kernel": "power2"|"exp", "s0": ..., "xi": ..., "sc": ...}
      params["model"]      = {"kernel": "power2"|"exp", "s0": ..., "xi": ..., "sc": ...}

    Returns:
      SpectralKnobs or None if unavailable.
    """
    src = params.get("frsr.model") or params.get("model") or {}
    kernel_name = (src.get("kernel") or "").strip().lower()
    if not kernel_name:
        return None

    if kernel_name in ("power2", "power", "p2"):
        s0 = float(src.get("s0", 0.0))
        sc = float(src.get("sc", 0.0))
        if s0 <= 0.0:
            return None
        return SpectralKnobs(kernel=Kernel.POWER2, s0=s0, sc=sc)
    if kernel_name in ("exp", "exponential"):
        xi = float(src.get("xi", 0.0))
        sc = float(src.get("sc", 0.0))
        if xi <= 0.0:
            return None
        return SpectralKnobs(kernel=Kernel.EXP, xi=xi, sc=sc)
    return None


def _warn_default(name: str, val) -> None:
    log.warning(f"Using default {name} = {val}")


def _set_class_root(ini_path: str, run_dir: str) -> str:
    """
    Ensure a single CLASS `root = ...` line exists in the output .ini.

    Policy:
      - Write results to `<run_dir>/out/<slug>_` where `slug = basename(run_dir)`
        sanitized to `[A-Za-z0-9_]+`.
      - Ensure `<run_dir>/out` exists.
      - If a `root` line already exists anywhere, replace it.
      - Otherwise, insert `root = ...` at the very top of the file.
    """
    log = get_logger()
    out_dir = os.path.join(run_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    slug = os.path.basename(run_dir) or "frsr"
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", slug)
    root_value = os.path.join(out_dir, f"{slug}_")

    with open(ini_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    replaced = False
    for line in lines:
        if line.strip().startswith("root"):
            new_lines.append(f"root            = {root_value}\n")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        # Insert at the very top
        new_lines.insert(0, f"root            = {root_value}\n")

    with open(ini_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    log.info("[ok] Set CLASS root to: {}", root_value)
    return root_value


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a CLASS background .ini with CPL dark energy only.")
    ap.add_argument("--params", type=str, default=None, help="Path to params.json from calibration (optional).")
    ap.add_argument("--out", type=str, default="frsr_background.ini", help="Output path for the generated ini.")
    ap.add_argument("--root", type=str, default=None, help="CLASS output root name (default: derived from run dir). Use 'auto' to force derivation.")
    ap.add_argument("--log-level", type=str, default="INFO", help="Logging level (default: INFO)")

    # Cosmology knobs (optional; override params)
    ap.add_argument("--h", type=float, default=None, help="Little h (H0/100). If omitted, inferred from params.json if possible.")
    ap.add_argument("--omega-b", type=float, default=None, help="Omega_b (baryons). Default 0.048 if unspecified.")
    ap.add_argument("--omega-cdm", type=float, default=None, help="Omega_cdm (cold dark matter). Default 0.262 if unspecified.")
    ap.add_argument("--omega-k", type=float, default=None, help="Omega_k curvature. Default 0.0 if unspecified.")
    ap.add_argument("--omega-fld", type=float, default=None, help="Omega_fld (dark energy fluid). If omitted, inferred or computed from params.json.")

    # CPL dark energy parameters (optional analytic mode)
    ap.add_argument("--w0", type=float, default=None,
                    help="Specify CPL equation-of-state parameter w0 (overrides epsilon/alpha).")
    ap.add_argument("--wa", type=float, default=None,
                    help="Specify CPL evolution parameter wa (overrides epsilon/alpha).")

    # FRSR epsilon and alpha mapping to CPL
    ap.add_argument("--epsilon", type=float, default=None,
                    help="FRSR ε; maps to CPL via w0=-1+ε, wa=-(α·ε) if --alpha also given.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="FRSR α; used with --epsilon.")

    args = ap.parse_args()

    init_logging(args.log_level)
    global log
    log = get_logger()

    params = _read_params(args.params)

    h = _infer_h(params, args.h)
    if h is None:
        h = 0.674
        _warn_default("h", h)

    Ob = args.omega_b if args.omega_b is not None else 0.048
    if args.omega_b is None:
        _warn_default("Omega_b", Ob)

    Ocdm = args.omega_cdm if args.omega_cdm is not None else 0.262
    if args.omega_cdm is None:
        _warn_default("Omega_cdm", Ocdm)

    Ok = args.omega_k if args.omega_k is not None else 0.0
    if args.omega_k is None:
        _warn_default("Omega_k", Ok)

    Ofld = _infer_Omega_fld(params, args.omega_fld, h)
    if Ofld is None:
        Ofld = 0.697
        _warn_default("Omega_fld", Ofld)

    # Helper to resolve w0, wa with precedence and validation
    def resolve_w0_wa() -> tuple[float, float]:
        """
        Resolve (w0, wa) in the following precedence:
          1) Explicit --w0 and --wa CLI.
          2) FRSR kernel in params.json mapped via map_kernel_to_cpl().
          3) (epsilon, alpha) mapping: w0=-1+ε, wa=-(α·ε).
          4) Fallback ΛCDM: (-1.0, 0.0).
        """
        w0_cli = args.w0
        wa_cli = args.wa
        epsilon = args.epsilon
        alpha = args.alpha

        # (1) Explicit CPL
        if (w0_cli is not None) or (wa_cli is not None):
            if (w0_cli is None) or (wa_cli is None):
                log.error("Both --w0 and --wa must be provided together, or omit both.")
                sys.exit(3)
            return float(w0_cli), float(wa_cli)

        # (2) Kernel → CPL via single-source map
        spec = _infer_kernel_spec(params)
        if spec is not None:
            try:
                w0_m, wa_m = map_kernel_to_cpl(spec)
                log.info("CPL derived from kernel: {} -> (w0={}, wa={})", spec.kernel, w0_m, wa_m)
                return float(w0_m), float(wa_m)
            except Exception as e:
                log.warning("Kernel-to-CPL mapping failed, will try epsilon/alpha next: {}", e)

        # (3) ε/α → CPL
        if (epsilon is not None) and (alpha is not None):
            return -1.0 + float(epsilon), -float(alpha) * float(epsilon)

        # (4) Default LCDM
        log.warning("Using default CPL parameters: w0=-1.0, wa=0.0 (ΛCDM default)")
        return -1.0, 0.0

    w0_val, wa_val = resolve_w0_wa()

    # Make sure directory exists
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Derive run_dir
    if args.params:
        run_dir = os.path.dirname(os.path.abspath(args.params))
    else:
        run_dir = os.getcwd()

    # Write ini
    lines = []
    lines.append("# Auto-generated by frsr_make_class_background.py\n")
    lines.append("# Minimal CLASS background with CPL dark energy only\n")
    lines.append("\n")
    lines.append(f"h                = {h}\n")
    lines.append(f"Omega_b          = {Ob}\n")
    lines.append(f"Omega_cdm        = {Ocdm}\n")
    if abs(Ok) > 0.0:
        lines.append(f"Omega_k          = {Ok}\n")
    lines.append("\n")
    lines.append("# CPL fluid dark energy\n")
    # CLASS expects a single dark-energy budget; we emit only Omega_fld here.
    lines.append(f"Omega_fld        = {Ofld}\n")
    lines.append("\n")
    lines.append("# Analytic CPL equation of state: w(a) = w0 + wa*(1 - a)\n")
    lines.append("use_tabulated_w  = no\n")
    lines.append(f"w0_fld           = {w0_val}\n")
    lines.append(f"wa_fld           = {wa_val}\n")
    lines.append("# Tip: Requires CLASS with CPL support (v3.3.3+).\n")

    lines.append("\n")
    lines.append("# Optional\n")
    lines.append("cs2_fld          = 1.0\n")
    lines.append("z_max_pk         = 3.0\n")
    # Emit background & parameters by default (safe and useful); leave spectra commented
    lines.append("\n")
    lines.append("# Output files (safe defaults)\n")
    lines.append("write background = yes\n")
    lines.append("write parameters = yes\n")
    lines.append("# To also write matter power spectrum, uncomment the next line:\n")
    lines.append("# output           = mPk\n")
    lines.append("# To write transfer functions, uncomment the next two lines:\n")
    lines.append("# write transfer   = yes\n")
    lines.append("# transfer_kmax    = 0.5\n")

    # Write the ini file without root line first
    with open(out_path, "w") as f:
        f.writelines(lines)

    try:
        _set_class_root(out_path, run_dir)
    except Exception as e:
        log.warning("Could not set CLASS root automatically: {}", e)

    log.info(
        "\n============================================================================\n"
        "FRSR → CLASS background ini\n"
        "----------------------------------------------------------------------------\n"
        f"ini written   : {out_path}\n"
        f"h             : {h}\n"
        f"Omega_b,cdm,k : {Ob}, {Ocdm}, {Ok}\n"
        f"Omega_fld     : {Ofld}\n"
        "----------------------------------------------------------------------------\n"
        "Mode: CPL analytic dark energy\n"
        f"w0 = {w0_val}, wa = {wa_val}\n"
        "----------------------------------------------------------------------------\n"
        "Tip: run CLASS via e.g. ./class " + out_path + "\n"
        "Note: CLASS v3.3.3 writes files as '<prefix>_00_background.dat' in the directory part of 'root'.\n"
        "      If 'root' ends with '/', filenames start with '00_…' (no prefix).\n"
        "============================================================================"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
