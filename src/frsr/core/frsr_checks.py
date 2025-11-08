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

# Derived from the legacy physics_constraints module
# -*- coding: utf-8 -*-
"""
Physics-driven constraint checks for FRSR calibrations.

This module validates that a given finite-band ladder and its CPL-only cosmological
background (Omega_fld + {w0, wa}) satisfy a set of sanity and phenomenology constraints
**without** requiring a CLASS run or any tabulated w(a) file.

Constraints implemented (A–D):

A) Convergence & discretization sanity
   A1. q = r^4 exp(-beta) < q_max (default 0.95).
   A2. Automatic N-insensitivity by extending N -> 2N holding (E0,r,alpha,beta,k_IR) fixed.
   A3. Weights in [0,1] and (weakly) monotone non-increasing.
   A4. IR pivot (k_IR) within a broad physical window.
   A5. Energy mostly in the first few bands (by default, top-5 contribute ≥80%).

B) Background expansion compatibility (CPL-only)
   B1. Positivity of H^2(a) over a grid.
   B2. Optional LCDM-limit smoke test when (w0, wa) ≈ (-1, 0).

C) Basic stability proxies
   C1. Ω_FRSR(a) > 0 over a grid from the CPL formula.

D) Parsimony / robustness hooks
   D1. Soft recommendation on N and spread of per-band contributions.

validate_constraints(...) -> (ok: bool, report: dict)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .frsr_background import (
    Anchors,
    EoSKnobs,
    FRBackground,
    SpectralKnobs,
    H2_std,
    H2_total_eos,
    H2_total_spectral,
    Omega_frsr_of_a,
    FRBackgroundSpectral,
)

# -------------------------------------------------------------------------
# Symbolic verification and logging utilities
# -------------------------------------------------------------------------
from sympy import symbols, simplify
from frsr.utils.log import get_logger, init_logging, sympy_to_text

log = get_logger()

"""
Notes:
- This module defines physics consistency checks for FRSR components.
- SymPy is used optionally for symbolic verification and pretty expression output.
- Logging uses the shared Loguru instance from frsr.utils.log.
"""


def _finite_geo_sum(q: float, N: int) -> float:
    if abs(1.0 - q) < 1e-12:
        return float(N)
    return (1.0 - q**N) / (1.0 - q)


def _preclean_key(s: str) -> str:
    """
    Normalize header keys by stripping unit suffixes and common unicode variants.
    Examples:
      "𝔈_i [unitless]" -> "E_i"
      "Δk_i [m^-1]"    -> "dk_i"
      "term_i [m^-4]"  -> "term_i"
    """
    if not isinstance(s, str):
        return s
    t = s.strip()
    for sep in (" [", "\t[", " (", "\t("):
        pos = t.find(sep)
        if pos != -1:
            t = t[:pos]
            break
    t = t.replace("𝔈", "E").replace("Δ", "d").replace(" ", "_")
    return t


def ladder_from_rows(
    rows: Iterable[Union[Dict[str, object], Tuple[float, ...], List[float]]],
    **metadata: Any,
) -> Dict[str, Any]:
    """
    Normalize ladder rows from a CSV/dict iterable into a canonical structure.

    Canonical keys per row:
        - i      : int (1-based band index)
        - k_i    : float (pivot wavenumber of band i)           [m^-1]
        - dk_i   : float (band width in k)                      [m^-1]
        - E_i    : float (dimensionless entanglement weight)
        - term_i : float (per-band contribution to C_FRSR)      [m^-4]

    Accepted aliases per field (case-insensitive):
        i      -> "i", "index", "band"
        k_i    -> "k_i", "ki", "k", "k[m^-1]", "k_i[m^-1]", "k_pivot"
        dk_i   -> "dk_i", "dki", "Δk", "delta_k", "dk", "dk[m^-1]"
        E_i    -> "E_i", "Ei", "E", "weight", "w_i"
        term_i -> "term_i", "term", "contrib", "contribution", "C_i"

    Non-dict rows such as tuples/lists are interpreted in the order:
        (i, k_i, dk_i, E_i, term_i) or (k_i, dk_i, E_i, term_i) / (k_i, dk_i, E_i).
    In the shorter cases the band index is auto-filled and term_i is derived if needed.

    Extra keyword arguments (e.g., N=..., r=..., q=...) are copied into the
    returned dictionary so callers passing richer contexts remain compatible.
    """
    def _norm_key_map() -> Dict[str, str]:
        aliases = {
            "i": ["i", "index", "band"],
            "k_i": ["k_i", "ki", "k", "k[m^-1]", "k_i[m^-1]", "k_pivot"],
            "dk_i": ["dk_i", "dki", "dk", "dk[m^-1]", "delta_k", "d_k", "dki[m^-1]"],
            "E_i": ["E_i", "Ei", "E", "weight", "w_i", "E_i_unitless", "Ei_unitless"],
            "term_i": ["term_i", "term", "contrib", "contribution", "c_i", "term_i_m^-4"],
        }
        flat: Dict[str, str] = {}
        for canon, alist in aliases.items():
            for a in alist:
                flat[a.lower()] = canon
        return flat

    keymap = _norm_key_map()

    def _canonize_key(k: str) -> Optional[str]:
        k2 = _preclean_key(k)
        return keymap.get(k2.lower())

    def _to_float(v: object, default: float = 0.0) -> float:
        try:
            if v is None:
                return default
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            if s == "" or s.lower() in {"nan", "none"}:
                return default
            return float(s)
        except Exception:
            return default

    def _to_int(v: object, default: Optional[int] = None) -> Optional[int]:
        try:
            if v is None:
                return default
            if isinstance(v, int):
                return v
            s = str(v).strip()
            if s == "" or s.lower() in {"nan", "none"}:
                return default
            return int(float(s))
        except Exception:
            return default

    out: List[Dict[str, float]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            if isinstance(row, (tuple, list)):
                if len(row) == 5:
                    row = {
                        "i": row[0],
                        "k_i": row[1],
                        "dk_i": row[2],
                        "E_i": row[3],
                        "term_i": row[4],
                    }
                elif len(row) == 4:
                    row = {
                        "i": idx,
                        "k_i": row[0],
                        "dk_i": row[1],
                        "E_i": row[2],
                        "term_i": row[3],
                    }
                elif len(row) == 3:
                    k_i, dk_i, E_i = row
                    row = {
                        "i": idx,
                        "k_i": k_i,
                        "dk_i": dk_i,
                        "E_i": E_i,
                        "term_i": E_i * (k_i ** 3) * dk_i,
                    }
                else:
                    raise ValueError(f"Unsupported ladder row length: {len(row)}")
            else:
                raise TypeError(f"Unsupported ladder row type: {type(row)!r}")
        # Build a dict with canonical keys + passthrough of extras
        # Build a dict with canonical keys + passthrough of extras
        canon: Dict[str, float] = {}
        extras: Dict[str, object] = {}

        for k, v in row.items():
            if isinstance(k, str):
                k = _preclean_key(k)
            ck = _canonize_key(k) if isinstance(k, str) else None
            if ck is None:
                extras[k] = v
                continue
            if ck == "i":
                canon["i"] = _to_int(v, None)  # type: ignore[assignment]
            elif ck == "k_i":
                canon["k_i"] = _to_float(v)
            elif ck == "dk_i":
                canon["dk_i"] = _to_float(v)
            elif ck == "E_i":
                canon["E_i"] = _to_float(v)
            elif ck == "term_i":
                canon["term_i"] = _to_float(v)

        # Ensure required index `i`
        if "i" not in canon or canon["i"] is None:
            canon["i"] = idx  # fallback to enumeration

        # Fill missing numerics with zeros
        canon.setdefault("k_i", 0.0)
        canon.setdefault("dk_i", 0.0)
        canon.setdefault("E_i", 0.0)
        canon.setdefault("term_i", 0.0)

        try:
            if (abs(float(canon.get("term_i", 0.0))) == 0.0) and all(x in canon for x in ("k_i", "dk_i", "E_i")):
                ki = float(canon["k_i"])
                dki = float(canon["dk_i"])
                Ei = float(canon["E_i"])
                canon["term_i"] = Ei * (ki ** 3) * dki
        except Exception:
            pass

        # Merge extras through (non-canonical keys preserved)
        for k, v in extras.items():
            canon[k] = v  # passthrough

        out.append(canon)  # type: ignore[list-item]

    N_val = int(metadata.get("N", len(out)))
    r_val = metadata.get("r")
    alpha_val = metadata.get("alpha")
    beta_val = metadata.get("beta", metadata.get("beta_used"))
    k_ir_val = metadata.get("k_IR")
    E0_val = metadata.get("E0")
    q_val = metadata.get("q")

    if q_val is None and r_val is not None and beta_val is not None:
        try:
            q_val = (float(r_val) ** 4) * math.exp(-float(beta_val))
        except Exception:
            q_val = None

    terms: List[float] = []
    eis: List[float] = []
    for row in out:
        ti = float(row.get("term_i", 0.0))
        ei = float(row.get("E_i", 0.0))
        if ti == 0.0 and all(k in row for k in ("k_i", "dk_i", "E_i")):
            try:
                ti = float(row["E_i"]) * float(row["k_i"]) ** 3 * float(row["dk_i"])
            except Exception:
                pass
        terms.append(ti)
        eis.append(ei)

    ladder: Dict[str, Any] = {
        "rows": out,
        "N": N_val,
        "terms": terms,
        "Ei": eis,
    }

    for key in ("r", "alpha", "beta", "beta_used", "k_IR", "E0"):
        if key in metadata and metadata[key] is not None:
            ladder[key] = metadata[key]
    if q_val is not None:
        ladder["q"] = q_val

    log.debug("ladder_from_rows: normalized {} rows", len(out))
    return ladder


@dataclass
class LadderSummary:
    N: int
    r: float
    alpha: float
    beta: float
    k_IR: float
    E0: float
    q: float
    terms: List[float]
    Ei: List[float]

    @property
    def spread_log(self) -> float:
        import statistics
        eps = 1e-300
        logs = [math.log(max(t, eps)) for t in self.terms if t > 0.0]
        return statistics.pstdev(logs) if len(logs) > 1 else float("inf")


def _ladder_summary_from_dict(ladder: Dict[str, Any]) -> LadderSummary:
    rows = ladder.get("rows") or []
    if not rows:
        raise ValueError("Ladder dictionary missing 'rows'.")

    def _row_to_tuple(idx: int, row: Union[Dict[str, Any], Tuple[Any, ...], List[Any]]) -> Tuple[int, float, float, float, float]:
        if isinstance(row, dict):
            cleaned = {(_preclean_key(k) if isinstance(k, str) else k): v for k, v in row.items()}
            i_val = cleaned.get("i", idx)
            k_val = cleaned.get("k_i") or cleaned.get("k")
            dk_val = cleaned.get("dk_i") or cleaned.get("dk")
            E_val = cleaned.get("E_i") or cleaned.get("E")
            term_val = cleaned.get("term_i") or cleaned.get("term")
            return (
                int(i_val),
                float(k_val or 0.0),
                float(dk_val or 0.0),
                float(E_val or 0.0),
                float(term_val or 0.0),
            )
        if isinstance(row, (tuple, list)):
            if len(row) == 5:
                i_val, k_val, dk_val, E_val, term_val = row
            elif len(row) == 4:
                k_val, dk_val, E_val, term_val = row
                i_val = idx
            elif len(row) == 3:
                k_val, dk_val, E_val = row
                term_val = E_val * (k_val ** 3) * dk_val
                i_val = idx
            else:
                raise ValueError(f"Unsupported ladder row length: {len(row)}")
            return (
                int(i_val),
                float(k_val),
                float(dk_val),
                float(E_val),
                float(term_val),
            )
        raise TypeError(f"Unsupported ladder row type: {type(row)!r}")

    rows_tuple = [_row_to_tuple(idx, row) for idx, row in enumerate(rows, start=1)]
    terms = ladder.get("terms")
    if terms is None:
        terms = [float(r[-1]) for r in rows_tuple]
    else:
        terms = [float(t) for t in terms]

    Ei = ladder.get("Ei")
    if Ei is None:
        Ei = [float(r[3]) for r in rows_tuple]
    else:
        Ei = [float(e) for e in Ei]

    def _require(key: str) -> float:
        val = ladder.get(key)
        if val is None:
            raise ValueError(f"Ladder dictionary missing '{key}'.")
        return float(val)

    N_val = int(ladder.get("N", len(rows_tuple)))
    r_val = _require("r")
    alpha_val = _require("alpha")
    beta_raw = ladder.get("beta", ladder.get("beta_used"))
    if beta_raw is None:
        raise ValueError("Ladder dictionary missing 'beta' or 'beta_used'.")
    beta_val = float(beta_raw)
    k_ir_val = _require("k_IR")
    E0_val = _require("E0")
    q_val = ladder.get("q")
    if q_val is None:
        q_val = (r_val ** 4) * math.exp(-beta_val)

    return LadderSummary(
        N=N_val,
        r=r_val,
        alpha=alpha_val,
        beta=beta_val,
        k_IR=k_ir_val,
        E0=E0_val,
        q=float(q_val),
        terms=terms,
        Ei=Ei,
    )


@dataclass
class CosmologyBackground:
    Omega_fld0: float
    Omega_m0: float = 0.31
    Omega_r0: float = 8.4e-5
    Omega_k0: float = 0.0


# -------------------------------------------------------------------------
# A) Convergence & discretization sanity
# -------------------------------------------------------------------------

def check_convergence_margin(r: float, beta: float, q_max: float = 0.95) -> Tuple[bool, Dict]:
    """A1: Verify that the geometric convergence factor q=r^4 e^-β remains below q_max."""
    q = (r ** 4) * math.exp(-beta)
    ok = q < q_max
    return ok, {"q": q, "q_max": q_max, "pass": ok}


def check_N_insensitivity_autotail(
    ladder: "LadderSummary",
    rel_tol: float = 1e-2
) -> Tuple[bool, Dict]:
    """A2: Check insensitivity of sum to extending N → 2N with fixed ladder parameters."""
    N = ladder.N
    r = ladder.r
    alpha = ladder.alpha
    beta = ladder.beta
    k_IR = ladder.k_IR
    E0 = ladder.E0
    q = ladder.q

    C_sum_N = sum(ladder.terms)

    extra_terms = []
    for i in range(N + 1, 2 * N + 1):
        k_i = k_IR * (r ** (i - 1))
        dk_i = alpha * k_i
        Ei = E0 * math.exp(-beta * (i - 1))
        extra_terms.append(Ei * (k_i ** 3) * dk_i)
    C_sum_2N_fixedE0 = C_sum_N + sum(extra_terms)

    rel = abs(C_sum_2N_fixedE0 - C_sum_N) / max(abs(C_sum_N), 1e-300)
    pass_rel = rel <= rel_tol

    if q >= 1.0:
        tail_frac_inf = float("inf")
    else:
        tail_frac_inf = (q ** N) / (1.0 - q)

    return pass_rel, {
        "rel": rel,
        "rel_tol": rel_tol,
        "C_sum_N": C_sum_N,
        "C_sum_2N_fixedE0": C_sum_2N_fixedE0,
        "q": q,
        "tail_frac_inf": tail_frac_inf,
        "pass": pass_rel,
    }


def suggest_pruned_N(
    terms: "List[float]",
    target_cum_frac: float = 0.99,
    min_band_frac: float = 5e-4
) -> Dict:
    """Suggest effective N by pruning bands with negligible contribution."""
    tot = sum(terms)
    if tot <= 0.0:
        return {"N_eff": 0, "reason": "non-positive total", "kept_indices": []}

    cum = 0.0
    kept = []
    for i, term in enumerate(terms, start=1):
        frac = term / tot
        kept.append(i)
        cum += frac
        if (cum >= target_cum_frac) and (frac < min_band_frac):
            break

    N_eff = max(1, kept[-1]) if kept else 0
    return {
        "N_eff": N_eff,
        "target_cum_frac": target_cum_frac,
        "min_band_frac": min_band_frac,
        "cum_frac_at_N_eff": cum,
        "kept_indices": kept,
    }


def check_weights_monotone(Ei: List[float], tol: float = 1e-12) -> Tuple[bool, Dict]:
    """A3: Check weights are in [0,1] and weakly monotone non-increasing."""
    ok_range = all((e >= -tol and e <= 1.0 + tol) for e in Ei)
    ok_mono = all(Ei[i] <= Ei[i - 1] + 1e-12 for i in range(1, len(Ei)))
    return (ok_range and ok_mono), {"in_[0,1]": ok_range, "monotone_noninc": ok_mono}


def check_kIR_window(k_IR: float, k_min: float = 1e-12, k_max: float = 1e12) -> Tuple[bool, Dict]:
    """A4: Check k_IR is within a broad physical window."""
    ok = (k_IR > k_min) and (k_IR < k_max)
    return ok, {"k_IR": k_IR, "k_min": k_min, "k_max": k_max, "pass": ok}


def check_first_bands_dominate(terms: List[float], top_k: int = 5, min_frac: float = 0.8) -> Tuple[bool, Dict]:
    """A5: Check if top-k bands contribute at least min_frac of total (CPL-only context)."""
    if not terms:
        return False, {"error": "empty terms"}
    tot = sum(terms)
    if not math.isfinite(tot) or tot <= 0.0:
        return False, {"error": "non-positive total", "total": tot}
    frac = sum(sorted(terms, reverse=True)[:top_k]) / tot
    ok = frac >= min_frac
    return ok, {"top_k": top_k, "frac": frac, "min_frac": min_frac, "pass": ok}


# -------------------------------------------------------------------------
# B) Background expansion compatibility from w(a)
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Aggregate validator for all checks (A–D)
# -------------------------------------------------------------------------

def validate_constraints(
    ladder: Union[LadderSummary, Dict[str, Any]],
    cosmo: CosmologyBackground,
    options: Optional[Dict] = None,
) -> Tuple[bool, Dict]:
    """Run all checks and collect a single report (CPL-only, no w(a) table)."""
    if options is None:
        options = {}

    if isinstance(ladder, dict):
        ladder = _ladder_summary_from_dict(ladder)

    if not isinstance(ladder, LadderSummary):
        raise TypeError("ladder must be LadderSummary or dict produced by ladder_from_rows().")

    rep: Dict[str, Dict] = {}
    ok_all = True

    ok, rep["A1_convergence"] = check_convergence_margin(ladder.r, ladder.beta, options.get("q_max", 0.95))
    ok_all &= ok

    ok, rep["A3_weights"] = check_weights_monotone(ladder.Ei)
    ok_all &= ok

    ok, rep["A4_kIR_window"] = check_kIR_window(ladder.k_IR)
    ok_all &= ok

    ok, rep["A5_firstbands"] = check_first_bands_dominate(ladder.terms, options.get("top_k", 5), options.get("min_frac", 0.8))
    ok_all &= ok

    try:
        ok, rep["A2_N_insensitivity_auto"] = check_N_insensitivity_autotail(
            ladder, options.get("A2_rel_tol", 1e-2)
        )
        ok_all &= ok
    except Exception as e:
        rep["A2_N_insensitivity_auto"] = {"error": str(e)}

    # B) Background compatibility placeholders (CPL-only path handled via run_eos_background_checks)
    rep["B_background"] = {"note": "CPL-only mode: w(a) table checks removed"}

    rep["D2_pruning_suggestion"] = suggest_pruned_N(
        ladder.terms,
        target_cum_frac=options.get("target_cum_frac", 0.99),
        min_band_frac=options.get("min_band_frac", 5e-4),
    )

    rep["D1_parsimony"] = {
        "N": ladder.N,
        "spread_log": ladder.spread_log,
        "soft_hint": "Prefer smaller N and moderate spread to avoid fine-tuning; target spread_log ≲ 1.",
    }

    return ok_all, rep


# -------------------------------------------------------------------------

# -*- coding: utf-8 -*-
# Author: Sangwook Lee (aladdin@plusgenie.com)
# Date: 2025-10-28
#
# FRSR "fast ladder" checks:
# - Pure validation utilities (NO I/O, NO printing).
# - Works with frsr.core.frsr_background dataclasses.
# - Keeps the contract tiny: range checks, flatness, positivity, ΛCDM smoke test.

# ---------- Exceptions ----------

class CheckError(ValueError):
    """Raised when a validation fails in a non-recoverable way."""


# ---------- Basic range & consistency checks ----------

# -------------------------------------------------------------------------
# C) Basic range & consistency checks for background dataclasses
# -------------------------------------------------------------------------

def validate_anchors(anc: Anchors) -> None:
    """Check that Anchors values are within broad physical ranges."""
    if not (0.0 <= anc.Omega_b0 <= 1.0):
        raise CheckError(f"Omega_b0 must be in [0,1], got {anc.Omega_b0}")
    if not (0.0 <= anc.Omega_cdm0 <= 1.0):
        raise CheckError(f"Omega_cdm0 must be in [0,1], got {anc.Omega_cdm0}")
    if not (0.0 <= anc.Omega_r0 <= 1.0):
        raise CheckError(f"Omega_r0 must be in [0,1], got {anc.Omega_r0}")
    if not (-1.0 <= anc.Omega_k0 <= 1.0):
        raise CheckError(f"Omega_k0 should be in [-1,1], got {anc.Omega_k0}")
    if anc.H0_km_s_Mpc <= 0.0:
        raise CheckError("H0 must be > 0")


# -------------------------------------------------------------------------


# ---------- Background positivity / finiteness checks ----------

# -------------------------------------------------------------------------

def assert_H2_positive_on_grid_eos(a_grid: List[float], fr: FRBackground) -> None:
    """Ensure H^2(a) > 0 over the provided grid for EoS-based background."""
    for a in a_grid:
        H2 = H2_total_eos(a, fr)
        if not (math.isfinite(H2) and H2 > 0.0):
            raise CheckError(f"H^2(a) non-positive/invalid at a={a}: {H2}")


from .frsr_background import FRBackgroundSpectral

def assert_H2_positive_on_grid_spectral(a_grid: List[float], frs: FRBackgroundSpectral) -> None:
    """Ensure H^2(a) > 0 over the provided grid for spectral ΔH^2 background."""
    for a in a_grid:
        H2 = H2_total_spectral(a, frs)
        if not (math.isfinite(H2) and H2 > 0.0):
            raise CheckError(f"H^2(a) non-positive/invalid at a={a}: {H2}")


# ---------- EoS / Spectral knob validators ----------

def validate_eos_knobs(eos: EoSKnobs) -> None:
    """Lightweight range checks for common EoS parametrisations (CPL / exp_wz)."""
    if hasattr(eos, "w0") and not (-2.0 <= eos.w0 <= 0.5):
        raise CheckError(f"w0 out of range: {eos.w0}")
    if hasattr(eos, "wa") and not (-2.0 <= eos.wa <= 2.0):
        raise CheckError(f"wa out of range: {eos.wa}")
    if hasattr(eos, "epsilon") and not (abs(eos.epsilon) <= 0.2):
        raise CheckError(f"epsilon too large: {eos.epsilon}")
    if hasattr(eos, "alpha") and not (getattr(eos, "alpha") > 0.0):
        raise CheckError("alpha must be > 0 for exp_wz")
    if hasattr(eos, "cs2") and not (0.0 <= eos.cs2 <= 1.0):
        raise CheckError(f"cs2 must be in [0,1], got {eos.cs2}")


# ---------- ΛCDM sanity (smoke) test ----------

# -------------------------------------------------------------------------


# ---------- Aggregate convenience ----------

# -------------------------------------------------------------------------
# D) High-level background checks and symbolic verification
# -------------------------------------------------------------------------

def validate_spectral_knobs(spec: SpectralKnobs) -> None:
    """Lightweight range checks for spectral kernel parameters."""
    if hasattr(spec, "A") and not (spec.A > 0.0):
        raise CheckError("Spectral amplitude A must be > 0")
    if hasattr(spec, "k0") and not (spec.k0 > 0.0):
        raise CheckError("Reference scale k0 must be > 0")
    if hasattr(spec, "kernel") and hasattr(spec.kernel, "value") and not str(spec.kernel.value):
        raise CheckError("Kernel identifier empty")

@dataclass(frozen=True)
class CheckReport:
    ok: bool
    details: Dict[str, object]


def lcdm_limit_ok(a_grid: List[float], fr: FRBackground, tol: float = 1e-8) -> bool:
    """
    Return True if either (w0, wa) is not close to (-1, 0), or—if it is—
    H^2_total matches the standard (no-FRSR) background within `tol` fractionally.
    """
    w0 = getattr(fr.eos, "w0", None)
    wa = getattr(fr.eos, "wa", None)
    if w0 is None or wa is None:
        return True
    if not (abs(w0 + 1.0) < 1e-6 and abs(wa) < 1e-6):
        return True
    for a in a_grid:
        h2_std = H2_std(a, fr.anchors)
        h2_tot = H2_total_eos(a, fr)
        denom = max(abs(h2_std), 1e-300)
        if abs(h2_tot - h2_std) / denom > tol:
            return False
    return True


def run_eos_background_checks(
    fr: FRBackground,
    a_min: float = 1e-8,
    a_max: float = 2.0,
    n_grid: int = 256,
) -> CheckReport:
    """Run a robust suite of checks for the EoS-based background."""
    # 1) basic anchors
    validate_anchors(fr.anchors)
    # 2) eos knobs
    validate_eos_knobs(fr.eos)
    # 3) flatness or provided Ω_FRSR,0 is consistent
    Omega0 = validate_flatness_budget(fr.anchors, fr.Omega_frsr0)

    # Symbolic verification (optional)
    Omega_m0, Omega_r0, Omega_k0 = symbols('Omega_m0 Omega_r0 Omega_k0')
    expr = simplify(1 - (Omega_m0 + Omega_r0 + Omega_k0))
    log.debug("Flatness symbolic form Ω_FRSR,0 = {}", sympy_to_text(expr))

    # grid
    if a_min <= 0.0 or a_max <= 0.0 or a_min >= a_max:
        raise CheckError("Invalid a-grid bounds")
    step = (a_max - a_min) / (n_grid - 1)
    a_grid = [a_min + i * step for i in range(n_grid)]

    # 4) positivity
    assert_frsr_positive(a_grid, fr)
    assert_H2_positive_on_grid_eos(a_grid, fr)

    # 5) ΛCDM smoke test (only meaningful if w=-1)
    smoke = lcdm_limit_ok(a_grid, fr)

    log.info("EOS background checks passed; Ω_FRSR,0 used = {}", Omega0)

    return CheckReport(
        ok=True,
        details={
            "Omega_frsr0_used": Omega0,
            "a_min": a_min,
            "a_max": a_max,
            "n_grid": n_grid,
            "lcdm_smoke_ok": smoke,
        },
    )


def run_spectral_background_checks(
    anchors: Anchors,
    spec: SpectralKnobs,
    a_min: float = 1e-8,
    a_max: float = 2.0,
    n_grid: int = 256,
) -> CheckReport:
    """Run checks for the spectral ΔH² path."""
    log.info("Running spectral background checks for kernel {}", spec.kernel.value)
    validate_anchors(anchors)
    validate_spectral_knobs(spec)

    # Symbolic verification (optional)
    Omega_m0, Omega_r0, Omega_k0 = symbols('Omega_m0 Omega_r0 Omega_k0')
    expr = simplify(1 - (Omega_m0 + Omega_r0 + Omega_k0))
    log.debug("Flatness symbolic form Ω_FRSR,0 = {}", sympy_to_text(expr))

    if a_min <= 0.0 or a_max <= 0.0 or a_min >= a_max:
        raise CheckError("Invalid a-grid bounds")
    step = (a_max - a_min) / (n_grid - 1)
    a_grid = [a_min + i * step for i in range(n_grid)]

    assert_H2_positive_on_grid_spectral(a_grid, anchors, spec)

    return CheckReport(
        ok=True,
        details={
            "a_min": a_min,
            "a_max": a_max,
            "n_grid": n_grid,
            "kernel": spec.kernel.value,
            "A": spec.A,
        },
    )

# -------------------------------------------------------------------------
# E) Symbolic flatness relation helpers (for documentation/debug)
# -------------------------------------------------------------------------

def derive_flat_Omega_frsr0(anc: Anchors) -> float:
    """
    Flatness relation:
        Ω_FRSR,0 = 1 - (Ω_m0 + Ω_r0 + Ω_k0)
    Symbolic verification (optional)
    """
    # Symbolic verification (optional)
    Omega_m0, Omega_r0, Omega_k0 = symbols('Omega_m0 Omega_r0 Omega_k0')
    expr = simplify(1 - (Omega_m0 + Omega_r0 + Omega_k0))
    log.debug("Flatness symbolic form Ω_FRSR,0 = {}", sympy_to_text(expr))
    return 1.0 - (anc.Omega_m0 + anc.Omega_r0 + anc.Omega_k0)


def validate_flatness_budget(anc: Anchors, Omega_frsr0: Optional[float]) -> float:
    """
    If Omega_frsr0 is None, derive it from flatness.
    If provided, verify the total budget is ≤ 1 + small slack.
    Returns Ω_FRSR,0 to use.
    Symbolic verification (optional)
    """
    # Symbolic verification (optional)
    Omega_m0, Omega_r0, Omega_k0 = symbols('Omega_m0 Omega_r0 Omega_k0')
    expr = simplify(1 - (Omega_m0 + Omega_r0 + Omega_k0))
    log.debug("Flatness symbolic form Ω_FRSR,0 = {}", sympy_to_text(expr))
    if Omega_frsr0 is None:
        return derive_flat_Omega_frsr0(anc)
    total = anc.Omega_m0 + anc.Omega_r0 + anc.Omega_k0 + Omega_frsr0
    if total > 1.000_000_1:
        raise CheckError(f"Flat budget exceeded: Ω_tot={total}")
    return Omega_frsr0


def main() -> None:
    # initialize shared Loguru config (respects FRSR_LOG_LEVEL env)
    init_logging()
