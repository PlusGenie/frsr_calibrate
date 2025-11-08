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

# Purpose: Export CLASS-compatible background .ini files in CPL mode (no w_table).

from __future__ import annotations
import os, sys, json
from typing import Optional

from frsr.core.frsr_background import map_kernel_to_cpl, SpectralKnobs, Kernel
# Reuse H0 conversion and c from the core background module, if needed later
# (not strictly required here but harmless to import)
try:
    from frsr.core.frsr_background import kmps_per_Mpc_to_Hz, c  # noqa: F401
except Exception:
    kmps_per_Mpc_to_Hz = None
    c = 2.997_924_58e8
def _resolve_cpl_from_model_dict(model: dict, w0_fld: Optional[float], wa_fld: Optional[float]) -> tuple[float, float]:
    """
    Resolve (w0, wa) with precedence:
      1) Explicit w0_fld/wa_fld arguments (if both provided).
      2) Explicit w0/wa present in model dict.
      3) Kernel→CPL via map_kernel_to_cpl() using model dict (kernel + s0/xi/sc).
      4) Fallback (-0.98, 0.0) as a placeholder (warn).
    """
    # (1) CLI/explicit
    if (w0_fld is not None) and (wa_fld is not None):
        return float(w0_fld), float(wa_fld)

    # (2) Direct from model
    if "w0" in model and "wa" in model:
        try:
            return float(model["w0"]), float(model["wa"])
        except Exception:
            pass

    # (3) Kernel mapping
    kernel_name = (model.get("kernel") or "").strip().lower()
    if kernel_name:
        try:
            if kernel_name in ("power2", "power", "p2"):
                s0 = float(model.get("s0", 0.0))
                sc = float(model.get("sc", 0.0))
                if s0 > 0.0:
                    w0_m, wa_m = map_kernel_to_cpl(SpectralKnobs(kernel=Kernel.POWER2, s0=s0, sc=sc))
                    return float(w0_m), float(wa_m)
            elif kernel_name in ("exp", "exponential"):
                xi = float(model.get("xi", 0.0))
                sc = float(model.get("sc", 0.0))
                if xi > 0.0:
                    w0_m, wa_m = map_kernel_to_cpl(SpectralKnobs(kernel=Kernel.EXP, xi=xi, sc=sc))
                    return float(w0_m), float(wa_m)
        except Exception:
            pass

    # (4) Fallback (warn)
    sys.stderr.write("[warn] Could not resolve (w0,wa) from model/kernel; using placeholder (-0.98, 0.0)\n")
    return -0.98, 0.0


def patch_class_root(ini_path: str, run_dir: str) -> None:
    """
    Patch 'root =' line in the ini to something unique under CLASS/output/.
    If absent, insert it near the top.
    """
    # Build a unique token using run_dir
    src_dir = os.path.dirname(os.path.abspath(__file__))
    rel = os.path.relpath(run_dir, start=src_dir)
    token = "src_" + rel.replace(os.sep, "_").replace(".", "_").replace("-", "_")
    class_root = f"output/{token}"

    try:
        with open(ini_path, "r") as f:
            lines = f.readlines()
        new_lines = []
        replaced = False
        for line in lines:
            if line.strip().startswith("root"):
                new_lines.append(f"root            = {class_root}\n")
                replaced = True
            else:
                new_lines.append(line)
        if not replaced:
            insert_at = 2 if len(new_lines) >= 2 else len(new_lines)
            new_lines.insert(insert_at, f"root            = {class_root}\n")
        with open(ini_path, "w") as f:
            f.writelines(new_lines)
        print(f"[ok] Set CLASS root to: {class_root}")
    except Exception as e:
        print(f"[warn] Could not set CLASS root in {ini_path}: {e}")


def write_cpl_background_ini(
    out_path: str,
    *,
    h: float,
    Omega_b: float,
    Omega_cdm: float,
    Omega_de0: float,
    Omega_k: float = 0.0,
    Omega_r: float | None = None,
    w0_fld: float = -0.98,
    wa_fld: float = 0.0,
    cs2_fld: float = 1.0,
    use_ppf: bool = True,
    output: str = "mPk",
    P_k_max_h_over_Mpc: float = 2.0,
    z_pk: str = "0,0.5,1,2",
) -> None:
    """
    Write a CPL-only CLASS background ini file.

    This function generates a CLASS-compatible .ini file specifying cosmological parameters
    for a dark energy model with CPL (Chevallier-Polarski-Linder) parameterization.

    Key CLASS input parameters:
    - Omega_fld: fractional density of the dark energy fluid component.
    - Omega_Lambda: cosmological constant density fraction, set to zero here as CPL is used.
    - w0_fld: present-day equation of state parameter for dark energy.
    - wa_fld: evolution parameter of the dark energy equation of state.
    - cs2_fld: sound speed squared of the dark energy fluid.
    - use_ppf: whether to use the Parameterized Post-Friedmann (PPF) approach for dark energy perturbations.

    Other parameters include the Hubble parameter h, baryon and cold dark matter densities,
    curvature Omega_k, radiation Omega_r, and output settings.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Auto-generated CLASS background ini (CPL-only)\n")
        f.write("# Use Omega_fld with CPL parameters w0_fld, wa_fld\n")
        f.write("write background = yes\n")
        f.write("overwrite_root = yes\n")
        f.write(f"h = {h:.6f}\n")
        f.write(f"Omega_b = {Omega_b}\n")
        f.write(f"Omega_cdm = {Omega_cdm}\n")
        f.write(f"Omega_k = {Omega_k}\n")
        if Omega_r is not None:
            f.write(f"Omega_r = {Omega_r}\n")
        f.write("\n")
        f.write("Omega_Lambda = 0\n")
        f.write(f"Omega_fld = {Omega_de0:.8f}\n")
        f.write(f"use_ppf = {'yes' if use_ppf else 'no'}\n")
        f.write(f"cs2_fld = {cs2_fld}\n")
        f.write(f"w0_fld = {w0_fld}\n")
        f.write(f"wa_fld = {wa_fld}\n")
        f.write("\n")
        f.write(f"output = {output}\n")
        f.write(f"P_k_max_h/Mpc = {P_k_max_h_over_Mpc}\n")
        f.write(f"z_pk = {z_pk}\n")


def write_cpl_background_from_params_json(
    params_json_path: str,
    out_path: str,
    *,
    w0_fld: float | None = None,
    wa_fld: float | None = None,
    cs2_fld: float = 1.0,
    use_ppf: bool = True,
) -> None:
    """
    Read cosmological parameters from JSON and write CPL-only CLASS background ini.

    This function produces CLASS-compatible .ini files where 'Omega_fld' represents
    the dark energy density fraction, and 'w0_fld', 'wa_fld' define the CPL equation
    of state parameters for dark energy. It reads parameters from a JSON file and
    converts them into the appropriate CLASS input format.
    """
    with open(params_json_path, "r") as f:
        params = json.load(f)

    class_params = params.get("class_params", {})
    anchors = params.get("anchors", {})

    H0_km_s_Mpc = None
    Omega_b0 = None
    Omega_cdm0 = None
    Omega_k0 = 0.0
    Omega_r0 = None
    Omega_de0 = None

    # Prefer anchors if present, else class_params
    if anchors:
        H0_km_s_Mpc = anchors.get("H0")
        Omega_b0 = anchors.get("Omega_b0")
        Omega_cdm0 = anchors.get("Omega_cdm0")
        Omega_k0 = anchors.get("Omega_k0", 0.0)
        Omega_r0 = anchors.get("Omega_r0")
        Omega_de0 = anchors.get("Omega_de0")
    else:
        H0_km_s_Mpc = class_params.get("H0_km_s_Mpc")
        Omega_b0 = class_params.get("Omega_b")
        Omega_cdm0 = class_params.get("Omega_cdm")
        Omega_k0 = class_params.get("Omega_k", 0.0)
        Omega_r0 = class_params.get("Omega_r")
        Omega_de0 = class_params.get("Omega_de0")

    if H0_km_s_Mpc is None or Omega_b0 is None or Omega_cdm0 is None:
        raise ValueError("Missing required cosmological parameters in JSON")

    h = float(H0_km_s_Mpc) / 100.0
    Omega_b = float(Omega_b0)
    Omega_cdm = float(Omega_cdm0)
    Omega_k = float(Omega_k0)
    Omega_r = float(Omega_r0) if Omega_r0 is not None else None

    if Omega_de0 is not None:
        Omega_de0 = float(Omega_de0)
    else:
        # Compute Omega_de0 as closure
        sum_others = Omega_b + Omega_cdm + Omega_k + (Omega_r if Omega_r is not None else 0.0)
        Omega_de0 = 1.0 - sum_others

    model_like = (
        params.get("model")
        or params.get("frsr", {}).get("model")
        or params.get("frsr.model")  # tolerates flat dicts
        or {}
    )
    w0_fld_val, wa_fld_val = _resolve_cpl_from_model_dict(model_like, w0_fld, wa_fld)

    write_cpl_background_ini(
        out_path,
        h=h,
        Omega_b=Omega_b,
        Omega_cdm=Omega_cdm,
        Omega_de0=Omega_de0,
        Omega_k=Omega_k,
        Omega_r=Omega_r,
        w0_fld=w0_fld_val,
        wa_fld=wa_fld_val,
        cs2_fld=cs2_fld,
        use_ppf=use_ppf,
    )


def export_class_background(frozen, out_path):
    """
    Convert FRSR-calibrated parameters into CLASS input format (.ini) with CPL dark energy.

    This function extracts cosmological parameters from a frozen dictionary and writes
    a CLASS-compatible background .ini file, where parameters like Omega_fld, w0_fld,
    and wa_fld define the dark energy density and its CPL equation of state.
    """
    anchors = frozen.get("anchors", {})
    model = frozen.get("model", {})

    H0 = anchors.get("H0")
    Omega_b0 = anchors.get("Omega_b0")
    Omega_cdm0 = anchors.get("Omega_cdm0")
    Omega_k0 = anchors.get("Omega_k0", 0.0)
    Omega_r0 = anchors.get("Omega_r0", 0.0)
    Omega_de0 = anchors.get("Omega_de0")

    if None in (H0, Omega_b0, Omega_cdm0):
        raise ValueError("Missing required anchors in frozen dict")

    h = H0 / 100.0

    if Omega_de0 is None:
        Omega_de0 = 1.0 - (Omega_b0 + Omega_cdm0 + Omega_k0 + Omega_r0)

    w0, wa = _resolve_cpl_from_model_dict(model, None, None)

    write_cpl_background_ini(
        out_path,
        h=h,
        Omega_b=Omega_b0,
        Omega_cdm=Omega_cdm0,
        Omega_de0=Omega_de0,
        Omega_k=Omega_k0,
        Omega_r=Omega_r0,
        w0_fld=w0,
        wa_fld=wa,
        cs2_fld=1.0,
        use_ppf=True,
    )
