# -*- coding: utf-8 -*-
# Author: Sangwook Lee (aladdin@plusgenie.com)
# Date: 2025-10-27

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import ValidationError

from ..config.settings import FRSRSettings
from frsr.utils.log import get_logger

log = get_logger()

# TOML loader (3.11+: tomllib; else fall back to tomli)
try:  # Python 3.11+
    import tomllib as _toml  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    try:
        import tomli as _toml  # type: ignore
    except Exception:  # pragma: no cover
        _toml = None


# --- TOML loader helpers ---
def _require_toml():
    if _toml is None:
        raise RuntimeError("No TOML parser available. Use Python 3.11+ or install 'tomli'.")

def _load_toml(path: str) -> Dict[str, Any]:
    path = os.path.expanduser(os.path.expandvars(path))
    _require_toml()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TOML not found: {path}")
    with open(path, "rb") as f:
        data = _toml.load(f)  # type: ignore
    if not isinstance(data, dict):
        raise ValueError("Top-level TOML must be a table/object.")
    log.debug("Loaded TOML from {}", os.path.abspath(path))
    return data

def _slug_from_args(mode: str, args, deterministic: bool) -> str:
    if deterministic:
        key = (
            f"{mode}|N={getattr(args,'N',None)}|r={getattr(args,'r',None)}|a={getattr(args,'alpha',None)}|"
            f"b={getattr(args,'beta',None)}|E0={getattr(args,'E0_target',getattr(args,'E0_target_scan',None))}|"
            f"L={args.Lambda_obs}|lb={args.lambda_bare}|kap={args.kappa}"
        )
        h = hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]
        return f"{mode}_{h}_N{getattr(args,'N','')}_r{getattr(args,'r','')}_a{getattr(args,'alpha','')}_b{getattr(args,'beta','')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_N{getattr(args,'N','')}_r{getattr(args,'r','')}_a{getattr(args,'alpha','')}_b{getattr(args,'beta','')}"

def make_run_dir(mode: str, args) -> str:
    if getattr(args, "outdir", None):
        run_dir = args.outdir
        os.makedirs(run_dir, exist_ok=True)
        return os.path.abspath(run_dir)

    base = os.path.join("runs", mode)
    os.makedirs(base, exist_ok=True)

    if getattr(args, "deterministic", False):
        slug = _slug_from_args(mode, args, True)
        run_dir = os.path.join(base, slug)
        os.makedirs(run_dir, exist_ok=True)
        return os.path.abspath(run_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx = 0
    try:
        for name in os.listdir(base):
            if not name.startswith(ts + "_"):
                continue
            parts = name.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                idx = max(idx, int(parts[1]) + 1)
    except Exception:
        idx = 0

    n = getattr(args, 'N', '')
    r = getattr(args, 'r', '')
    a = getattr(args, 'alpha', '')
    b = getattr(args, 'beta', '')
    param_slug = f"N{n}_r{r}_a{a}_b{b}"

    slug = f"{ts}_{idx:02d}_{param_slug}"
    run_dir = os.path.join(base, slug)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.abspath(run_dir)

def class_root_from_run_dir(run_dir: str) -> str:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    rel = os.path.relpath(run_dir, start=src_dir)  # e.g., '../runs/calibrate/...'
    token = "src_" + rel.replace(os.sep, "_").replace(".", "_").replace("-", "_")
    return f"output/{token}"

def dump_env(run_dir: str) -> None:
    path = os.path.join(run_dir, "env.txt")
    try:
        py = sys.executable
        ver = sys.version.replace("\n", " ")
        plat = platform.platform()
        with open(path, "w") as f:
            f.write(f"python: {py}\n")
            f.write(f"version: {ver}\n")
            f.write(f"platform: {plat}\n")
            try:
                out = subprocess.check_output([py, "-m", "pip", "freeze"], stderr=subprocess.STDOUT, text=True, timeout=10)
                f.write("\n[pip freeze]\n")
                f.write(out)
            except Exception as e:
                f.write(f"\n[pip freeze] skipped: {e}\n")
    except Exception:
        pass

def dump_command(run_dir: str, argv: list[str]) -> None:
    path = os.path.join(run_dir, "command.sh")
    argv = argv or []
    module = "frsr.cli.run_scan" if "--scan" in argv else "frsr.cli.run_calibrate"
    cmd = " ".join(["python", "-m", module] + argv)
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"# Re-run this calibration\n{cmd}\n")
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass

def dump_params(run_dir: str, params: dict) -> None:
    pj = os.path.join(run_dir, "params.json")
    with open(pj, "w") as f:
        json.dump(params, f, indent=2)
    readme = os.path.join(run_dir, "README_run.md")
    with open(readme, "w") as f:
        f.write("# FRSR ladder calibration run\n\n")
        f.write("Artifacts in this folder:\n\n")
        f.write("- `frsr_ladder.csv`: finite-band ladder table\n")
        f.write("- `frsr_ladder_Ei.png`, `_ki.png`, `_term.png`: diagnostic plots\n")
        f.write("- `frsr_ladder_panel.png`: combined summary figure\n")
        f.write("- `params.json`: input parameters with explanations\n")
        f.write("- `env.txt`: Python and packages\n")
        f.write("- `command.sh`: exact replay command\n")
        f.write("- `frsr_background.ini`: auto-generated CLASS input (you may copy/edit variants under `variants/`)\n")
        f.write("- `variants/`: put your edited INIs here (e.g., frsr_background.Pk.ini)\n")

def write_csv(rows: list[tuple], path: str) -> None:
    import csv, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log.debug("Writing CSV to {}", path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "k_i [m^-1]", "Δk_i [m^-1]", "𝔈_i [unitless]", "term_i [m^-4]"])
        for r in rows:
            w.writerow(list(r))
    log.success("CSV saved: {}", path)



@dataclass(frozen=True)
class FrozenConfig:
    anchors: dict
    knobs: dict
    derived: dict
    meta: dict
    params: Optional[Dict[str, Any]] = None

    def to_params_dict(self) -> Dict[str, Any]:
        if self.params is not None:
            return self.params
        return _build_params_dict(self.anchors, self.knobs, self.derived, self.meta)


def _build_params_dict(
    anchors: Dict[str, Any],
    knobs: Dict[str, Any],
    derived: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    anchors_dict = dict(anchors)
    knobs_dict = dict(knobs)
    derived_dict = dict(derived)
    meta_dict = dict(meta)

    H0_val = float(anchors_dict.get("H0", anchors_dict.get("H0_km_s_Mpc", 67.4)))
    anchors_dict.setdefault("H0", H0_val)
    anchors_dict.setdefault("H0_km_s_Mpc", H0_val)

    omega_fld = (
        knobs_dict.get("Omega_fld")
        or knobs_dict.get("Omega_de0")
        or derived_dict.get("Omega_residual_hint")
    )

    class_params = {
        "H0_km_s_Mpc": H0_val,
        "h": derived_dict.get("h", H0_val / 100.0),
        "Omega_b": float(anchors_dict.get("Omega_b0", 0.048)),
        "Omega_b0": float(anchors_dict.get("Omega_b0", 0.048)),
        "Omega_cdm": float(anchors_dict.get("Omega_cdm0", 0.262)),
        "Omega_cdm0": float(anchors_dict.get("Omega_cdm0", 0.262)),
        "Omega_r": float(anchors_dict.get("Omega_r0", 5e-5)),
        "Omega_r0": float(anchors_dict.get("Omega_r0", 5e-5)),
        "Omega_k": float(anchors_dict.get("Omega_k0", 0.0)),
        "Omega_k0": float(anchors_dict.get("Omega_k0", 0.0)),
        "Omega_fld": float(omega_fld) if omega_fld is not None else None,
    }

    eos_keys = {"model"}
    inputs_keys = {
        "N",
        "r",
        "alpha_band",
        "beta",
        "E0_target",
        "E0",
        "k_IR",
        "auto_kIR",
        "Lambda_obs",
        "q",
        "q_cap",
    }

    eos = {k: knobs_dict[k] for k in eos_keys if k in knobs_dict}
    inputs = {k: knobs_dict[k] for k in inputs_keys if k in knobs_dict}

    params = {
        "mode": eos.get("model", knobs_dict.get("model")),
        "use_tabulated_w": knobs_dict.get("use_tabulated_w", True),
        "anchors": anchors_dict,
        "knobs": knobs_dict,
        "derived": derived_dict,
        "meta": meta_dict,
        "class_params": class_params,
        "inputs": inputs,
        "eos": eos,
    }
    return params


# --- Unified TOML config resolver ---
def resolve_toml(config_path: str) -> 'FrozenConfig':
    """
    Load unified FRSR TOML and build a FrozenConfig.

    Expected tables (all optional, with safe defaults):
      [frsr.anchors]
      [frsr.model]
      [frsr.calibration]
      [frsr.scan]
      [class.background]
      [class.output]
      [montepython.param]
    Unspecified tables are filled with safe defaults; missing keys will be inferred where possible.
    """
    data = _load_toml(config_path)

    frsr = data.get("frsr", {})
    fr_anchors = frsr.get("anchors", {})
    fr_model = frsr.get("model", {})
    fr_calib = frsr.get("calibration", {})
    fr_scan = frsr.get("scan", {})

    # Model kernel dials (used by calibration; runtime is CPL-only)
    kernel = fr_model.get("kernel", "power2")
    xi = fr_model.get("xi", None)
    s0 = fr_model.get("s0", None)
    sc = fr_model.get("sc", None)

    cls = data.get("class", {})
    cls_bg = cls.get("background", {})
    cls_out = cls.get("output", {})

    mp = data.get("montepython", {})
    mp_param = mp.get("param", {})

    # Anchors
    H0 = float(fr_anchors.get("H0", fr_anchors.get("H0_km_s_Mpc", 67.4)))
    anchors: Dict[str, Any] = {
        "H0": H0,
        "H0_km_s_Mpc": H0,
        "Omega_b0": float(fr_anchors.get("Omega_b0", 0.048)),
        "Omega_cdm0": float(fr_anchors.get("Omega_cdm0", 0.262)),
        "Omega_r0": float(fr_anchors.get("Omega_r0", 5e-5)),
        "Omega_k0": float(fr_anchors.get("Omega_k0", 0.0)),
    }

    # EoS / model knobs (CPL is the only runtime path;
    eos: Dict[str, Any] = {
        "model": fr_model.get("family", fr_model.get("model", "CPL")),
    }

    # Inputs / calibration knobs
    inputs_keys = (
        "N", "r", "alpha_band", "beta", "E0_target", "E0", "k_IR",
        "auto_kIR", "Lambda_obs", "q", "q_cap"
    )
    inputs: Dict[str, Any] = {k: fr_calib[k] for k in inputs_keys if k in fr_calib}

    # Derived hints
    h = H0 / 100.0
    Omega_b0 = anchors["Omega_b0"]; Omega_cdm0 = anchors["Omega_cdm0"]
    Omega_r0 = anchors["Omega_r0"]; Omega_k0 = anchors["Omega_k0"]
    Omega_residual = 1.0 - (Omega_b0 + Omega_cdm0 + Omega_r0 + Omega_k0)
    derived: Dict[str, Any] = {
        "h": h,
        "Omega_residual_hint": Omega_residual,
        "flat_hint": abs(Omega_k0) < 1e-9,
    }

    # CLASS-facing parameters (background)
    class_params: Dict[str, Any] = {
        "H0_km_s_Mpc": H0,
        "h": float(cls_bg.get("h", h)),
        "Omega_b": float(cls_bg.get("Omega_b", anchors["Omega_b0"])),
        "Omega_cdm": float(cls_bg.get("Omega_cdm", anchors["Omega_cdm0"])),
        "Omega_r": float(cls_bg.get("Omega_r", anchors["Omega_r0"])),
        "Omega_k": float(cls_bg.get("Omega_k", anchors["Omega_k0"])),
        # If Omega_fld absent, leave None — downstream may derive from residual
        "Omega_fld": cls_bg.get("Omega_fld", None),
        # CPL runtime flags live in frsr_background.ini after calibration
        "use_ppf": bool(cls_bg.get("use_ppf", True)),
        "cs2_fld": float(cls_bg.get("cs2_fld", 1.0)),
        # Force CLASS to use the CPL fluid (Omega_Lambda=0); can be overridden if explicitly set in TOML
        "Omega_Lambda": float(cls_bg.get("Omega_Lambda", 0.0)),
    }

    # Aggregate "knobs" for back-compat where callers expect a flat dict
    knobs: Dict[str, Any] = {}
    knobs.update(eos)
    knobs.update(inputs)
    # Propagate kernel dials for calibration consumers
    knobs["kernel"] = kernel
    if xi is not None:
        knobs["xi"] = float(xi)
    if s0 is not None:
        knobs["s0"] = float(s0)
    if sc is not None:
        knobs["sc"] = float(sc)
    # Legacy switches used by some call sites:
    knobs["use_tabulated_w"] = bool(fr_model.get("use_tabulated_w", False))

    meta_key = json.dumps({"frsr": frsr, "class": cls, "montepython": mp}, sort_keys=True)
    meta_hash = hashlib.sha1(meta_key.encode("utf-8")).hexdigest()
    meta: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hash": meta_hash[:12],
        "files": {"config": os.path.abspath(config_path)},
    }

    params = {
        "mode": eos.get("model"),
        "anchors": anchors,
        "knobs": knobs,
        "derived": derived,
        "meta": meta,
        "class_params": class_params,
        "inputs": inputs,
        "eos": eos,
        "scan": fr_scan,
        "class_output": cls_out,
        "montepython_param": mp_param,
        "model": {"kernel": kernel, "xi": xi, "s0": s0, "sc": sc},
    }
    return FrozenConfig(anchors=anchors, knobs=knobs, derived=derived, meta=meta, params=params)


def _resolve_from_settings() -> FrozenConfig:
    """
    Build a FrozenConfig from pydantic settings (env/.env driven). TOML is preferred and used by default when a config path is provided.
    """
    load_dotenv()
    try:
        settings = FRSRSettings()
    except ValidationError as exc:
        raise RuntimeError(f"FRSRSettings validation failed: {exc}") from exc

    anchors = settings.anchors.model_dump()
    H0 = float(anchors.get("H0", anchors.get("H0_km_s_Mpc", 67.4)))
    anchors.setdefault("H0", H0)
    anchors.setdefault("H0_km_s_Mpc", H0)

    eos = settings.eos.model_dump()
    inputs = settings.inputs.model_dump()
    if "alpha" in inputs and "alpha_band" not in inputs:
        inputs["alpha_band"] = inputs["alpha"]
    class_params = settings.class_params.model_dump()

    knobs: Dict[str, Any] = {}
    knobs.update(eos)
    knobs.update(inputs)
    knobs.update({k: v for k, v in class_params.items() if v is not None})
    knobs["model"] = eos.get("model", settings.eos.model)
    knobs["use_tabulated_w"] = settings.use_tabulated_w

    Omega_b0 = float(anchors.get("Omega_b0", 0.048))
    Omega_cdm0 = float(anchors.get("Omega_cdm0", 0.262))
    Omega_r0 = float(anchors.get("Omega_r0", 5e-5))
    Omega_k0 = float(anchors.get("Omega_k0", 0.0))
    derived: Dict[str, Any] = {
        "h": H0 / 100.0,
        "Omega_residual_hint": 1.0 - (Omega_b0 + Omega_cdm0 + Omega_r0 + Omega_k0),
        "flat_hint": abs(Omega_k0) < 1e-9,
    }

    meta_key = json.dumps({"anchors": anchors, "knobs": knobs}, sort_keys=True)
    meta_hash = hashlib.sha1(meta_key.encode("utf-8")).hexdigest()

    meta: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hash": meta_hash[:12],
        "files": {
            "anchors": None,
            "knobs": None,
            "profile": None,
        },
        "source": "FRSRSettings",
    }

    params_dict = _build_params_dict(anchors, knobs, derived, meta)
    return FrozenConfig(anchors=anchors, knobs=knobs, derived=derived, meta=meta, params=params_dict)


# Helper: dump combined config to params.json in run_dir
def dump_combined_params(run_dir: str, cfg: FrozenConfig) -> str:
    """
    Write a single params.json that contains anchors, knobs, derived, and meta.
    Returns the file path.
    """
    pj = os.path.join(run_dir, "params.json")
    payload = cfg.to_params_dict()
    os.makedirs(run_dir, exist_ok=True)
    with open(pj, "w") as f:
        json.dump(payload, f, indent=2)
    log.success("params.json written: {}", pj)
    return pj


# --- Compatibility shim for config resolution ---
def resolve(config_path: Optional[str] = None) -> FrozenConfig:
    """
    Compatibility shim:
      - If a TOML path is provided, delegate to resolve_toml().
      - Otherwise, fall back to environment-driven settings.
    """
    if config_path:
        return resolve_toml(config_path)
    return _resolve_from_settings()
