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
#
# Unified FRSR physics background utilities. This module provides:
#   - Ladder calibration helpers (compute_C_FRSR, solve_E0, etc.)
#   - Fast-ladder background dataclasses and analytic helpers for H(a)
#   - Lightweight sanity checks for background quantities

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import pstdev
# (no new external deps)
import numpy as np
from typing import Iterable, List, Optional, Tuple
from typing import Sequence
# (no new external deps)
# ---- Physical constants (SI) ----
G = 6.674_30e-11          # m^3 kg^-1 s^-2
c = 2.997_924_58e8        # m s^-1
hbar = 1.054_571_817e-34  # J s

# ---------------------------------------------------------------------------
# Ladder calibration helpers (legacy physics_core)
# ---------------------------------------------------------------------------

def kmps_per_Mpc_to_Hz(H_km_s_Mpc: float) -> float:
    """Convert H₀ (km/s/Mpc) to s⁻¹."""
    Mpc_m = 3.085_677_581_491_367e22
    return (H_km_s_Mpc * 1_000.0) / Mpc_m


def compute_C_FRSR(Lambda_obs: float, lambda_bare: float, kappa: float) -> float:
    """Compute 𝓒_FRSR appearing in Σ 𝔈_i k_i³ Δk_i = 𝓒_FRSR (units: m⁻⁴)."""
    return (math.pi * (c ** 3) / (2.0 * G * hbar)) * (Lambda_obs - lambda_bare) / kappa


def q_from(r: float, beta: float) -> float:
    """Return q = r⁴ · exp(-β)."""
    return (r ** 4) * math.exp(-beta)


def sum_geo(q: float, N: int) -> float:
    """Finite geometric sum Σ_{i=1..N} q^{i-1}, well behaved near q≈1."""
    if abs(1.0 - q) < 1e-12:
        return float(N)
    return (1.0 - (q ** N)) / (1.0 - q)


def solve_E0(
    C_FRSR: float,
    N: int,
    r: float,
    alpha: float,
    beta: float,
    k_IR: float,
) -> Tuple[float, float]:
    """Solve for 𝔈₀ (unitless), returning (E0, q)."""
    q = q_from(r, beta)
    S = sum_geo(q, N)
    E0 = C_FRSR / (alpha * (k_IR ** 4) * S)
    return E0, q


def solve_kIR_for_E0_target(
    C_FRSR: float,
    N: int,
    r: float,
    alpha: float,
    beta: float,
    E0_target: float,
) -> Tuple[float, float]:
    """Solve k_IR (m⁻¹) so that 𝔈₀ = E0_target ∈ (0,1]; returns (k_IR, q)."""
    q = q_from(r, beta)
    S = sum_geo(q, N)
    denom = alpha * E0_target * S
    if denom <= 0.0:
        raise ValueError("Non-positive denominator when solving k_IR; check alpha, E0_target, S.")
    k4 = C_FRSR / denom
    if k4 <= 0.0:
        raise ValueError("Computed k_IR⁴ ≤ 0; check inputs.")
    return (k4 ** 0.25), q


def build_ladder(
    N: int,
    r: float,
    alpha: float,
    beta: float,
    k_IR: float,
    E0: float,
) -> List[Tuple[int, float, float, float, float]]:
    """Return rows (i, k_i, Δk_i, 𝔈_i, term_i) with term_i = 𝔈_i k_i³ Δk_i."""
    rows: List[Tuple[int, float, float, float, float]] = []
    for i in range(1, N + 1):
        k_i = k_IR * (r ** (i - 1))
        dk_i = alpha * k_i
        Ei = E0 * math.exp(-beta * (i - 1))
        term_i = Ei * (k_i ** 3) * dk_i
        rows.append((i, k_i, dk_i, Ei, term_i))
    return rows


def spread_metric(terms: List[float]) -> float:
    """Log-spread (population stdev of logs). Lower is better."""
    eps = 1e-300
    logs = [math.log(max(t, eps)) for t in terms if t > 0.0]
    if len(logs) <= 1:
        return float("inf")
    return pstdev(logs)


def within_unit_interval(vals: List[float]) -> bool:
    """Check all values in [0,1]."""
    return all((0.0 <= v <= 1.0) for v in vals)


def sane_kir(kir: float) -> bool:
    """Loose window to avoid absurd solutions."""
    return (kir > 1e-12) and (kir < 1e12)


def adjust_beta(r_val: float, beta_in: float, q_cap_val: float) -> Tuple[float, float]:
    """Ensure q < 1 by minimally increasing β if needed; returns (beta_eff, q)."""
    q_tmp = q_from(r_val, beta_in)
    if q_tmp >= 1.0:
        q_cap_eff = q_cap_val if (0.0 < q_cap_val < 1.0) else 0.9
        beta_eff = math.log((r_val ** 4) / q_cap_eff)
        return beta_eff, q_from(r_val, beta_eff)
    return beta_in, q_tmp


def parse_list(arg: str, cast=float):
    """Utility to parse comma-separated lists with a given caster."""
    return [cast(x.strip()) for x in arg.split(",") if x.strip()]


class EoSModel(str, Enum):
    """
    Equation-of-state family (runtime).

    Policy (v2.1):
    • **CPL only** in pipelines. Dev variants removed to reduce surface area.
    """
    CPL = "CPL"            # w(a) = w0 + wa(1-a)


@dataclass(frozen=True)
class Anchors:
    """Present-epoch anchors (dimensionless Ω's; H0 in km/s/Mpc)."""
    H0_km_s_Mpc: float
    Omega_b0: float
    Omega_cdm0: float
    Omega_r0: float
    Omega_k0: float = 0.0

    @property
    def H0_SI(self) -> float:
        return kmps_per_Mpc_to_Hz(self.H0_km_s_Mpc)

    @property
    def Omega_m0(self) -> float:
        return self.Omega_b0 + self.Omega_cdm0

    @property
    def rho_crit0(self) -> float:
        """Critical density today (kg m⁻³)."""
        return 3.0 * (self.H0_SI ** 2) / (8.0 * math.pi * G)


@dataclass(frozen=True)
class EoSKnobs:
    """
    Parameters for CPL EoS used by pipelines.

    Provide (w0, wa) already mapped from spectral parameters via `map_kernel_to_cpl`.
    """
    model: EoSModel = EoSModel.CPL
    w0: float = -1.0
    wa: float = 0.0


@dataclass(frozen=True)
class FRBackground:
    """
    Background definition for the fast ladder when using an EoS-based FRSR term.
    Provide Omega_frsr0 if not deriving it from flatness.
    """
    anchors: Anchors
    eos: EoSKnobs
    Omega_frsr0: Optional[float] = None   # if None, derive via flatness


class Kernel(str, Enum):
    NONE = "NONE"
    EXP = "EXP"       # ΔH^2/H0^2 = A / (1 + xi * (a + 1/sc)); if sc=0 → A/(1 + xi * a)
    POWER2 = "POWER2" # ΔH^2/H0^2 = A / (1 + s0 * a)  (β fixed to 2)


@dataclass(frozen=True)
class SpectralKnobs:
    """
    Minimal spectral-kernel parametrization for fast ladder:
    - A ≥ 0 sets present-day amplitude of ΔH^2/H0^2 up to the kernel factor.
    - xi or s0 are positive scale parameters (dimensionless in a-units).
    - sc is an optional micro-coherence filter scale with 𝔈(s)=exp(-s/sc); set sc=0 to disable.

    Amplitude mapping (v2.1, Eq. A.1), explicitly documented for 1:1 code↔paper:
      A = (8*pi*G / (3*H0^2)) * kappa * rho0 * E0
      where:
      • H0 is in SI (s⁻¹), ρ0 in kg·m⁻³ (or energy density / c²), κ and E0 are dimensionless.
    """
    kernel: Kernel = Kernel.NONE
    A: float = 0.0
    xi: float = 0.0     # EXP kernel scale (dimensionless a-scale)
    s0: float = 0.0     # POWER2 kernel scale (β=2 fixed; dimensionless a-scale)
    sc: float = 0.0     # optional micro-coherence filter scale; 0 disables


@dataclass(frozen=True)
class FRBackgroundSpectral:
    """Background definition when using a ΔH^2 spectral kernel (no Ω_FRSR(a))."""
    anchors: Anchors
    spec: SpectralKnobs


def amplitude_from_physical(H0_SI: float, kappa: float, rho0: float, E0: float) -> float:
    """
    Compute the **dimensionless** spectral amplitude A from physical inputs.
    v2.1 (Eq. A.1):  A = (8πG / (3 H0^2)) * kappa * rho0 * E0
    Units:
    • H0_SI : s⁻¹
    • kappa : dimensionless coupling
    • rho0  : kg·m⁻³ (mass density; energy density/c² equivalently)
    • E0    : dimensionless base entanglement weight
    """
    if H0_SI <= 0.0:
        raise ValueError("H0_SI must be > 0.")
    return (8.0 * math.pi * G / (3.0 * (H0_SI ** 2))) * float(kappa) * float(rho0) * float(E0)


def map_kernel_to_cpl(spec: SpectralKnobs) -> tuple[float, float]:
    """
    Closed-form, single-source mapping from spectral kernel slope to CPL (w0, wa).

    FRSR v2.1 (§8.5) result:
      ΔH²/H0² ∝ 1 / (1 + L a),  with
        • POWER2 (β=2 locked): L = s0
        • EXP (with optional micro-filter 𝔈(s)=exp(−s/sc)):
              L = ξ_eff = (ξ * sc) / (ξ + sc)   if sc > 0
              L = ξ                              if sc == 0 or unset

    Effective EoS:
      w_eff(a) = -1 + (1/3) * (L a) / (1 + L a)

    CPL identification at a=1:
      w0 = -1 + L / [3 (1 + L)]
      wa = -    L / [3 (1 + L)^2]

    Inputs
    ------
    spec : SpectralKnobs
        Kernel ∈ {EXP, POWER2}; provide xi or s0 accordingly. sc applies only to EXP.

    Returns
    -------
    (w0, wa) : tuple[float, float]
        CPL params for CLASS/MontePython (w0_fld, wa_fld).

    Notes
    -----
    Amplitude A does not affect (w0, wa); it sets Ω_fld,0 only.
    This is the canonical mapping for FRSR→CPL.
    """
    # Determine slope L from kernel dials
    if spec.kernel == Kernel.POWER2:
        if spec.s0 <= 0.0:
            raise ValueError("POWER2 kernel requires s0 > 0.")
        L = float(spec.s0)
    elif spec.kernel == Kernel.EXP:
        if spec.xi <= 0.0:
            raise ValueError("EXP kernel requires xi > 0.")
        if spec.sc and spec.sc > 0.0:
            L = float((spec.xi * spec.sc) / (spec.xi + spec.sc))
        else:
            L = float(spec.xi)
    else:
        raise ValueError(f"Unsupported kernel for CPL mapping: {spec.kernel}")

    denom = 1.0 + L
    w0 = -1.0 + (L / (3.0 * denom))
    wa = - L / (3.0 * (denom ** 2))
    return w0, wa

def map_epsilon_alpha_to_cpl(epsilon: float, alpha: float) -> Tuple[float, float]:
    """
    Legacy proxy mapping from (epsilon, alpha) to (w0, wa).
    Deprecated: use map_kernel_to_cpl as the canonical mapping for FRSR→CPL.
    w0 = -1 + epsilon
    wa = -(alpha * epsilon)
    Returns:
    (w0, wa) as floats.
    """
    w0 = -1.0 + float(epsilon)
    wa = -float(alpha) * float(epsilon)
    return w0, wa


def w_proxy_z_from_eps_alpha(z: "np.ndarray | Sequence[float]", epsilon: float, alpha: float) -> "np.ndarray":
    """
    Analytic CPL proxy expressed in **redshift** z for quick sanity checks.

    Model:
        (w+1)(z) = ε · exp(-α z)  ⇒  w(z) = -1 + ε · exp(-α z)

    Inputs (physical meaning & units):
    • z (array-like, dimensionless): redshift samples where you want w(z).
    • epsilon ε (float, dimensionless): present‑day deviation from ΛCDM at z=0; w(0) = -1 + ε.
    • alpha α (float, dimensionless, >0): decay‑rate of the deviation with redshift.

    Output:
    • numpy.ndarray (dimensionless): equation‑of‑state w(z) evaluated at the provided z grid.

    Usage:
    • Used by MontePython hooks for monotonicity and high‑z‑negligibility checks.
    • Single‑sourced here to avoid duplicate physics elsewhere.
    """
    arr = np.asarray(z, dtype=float)
    return -1.0 + float(epsilon) * np.exp(-float(alpha) * arr)


def w_frsr(a: float, eos: EoSKnobs) -> float:
    """
    Equation of state w(a) as a function of scale factor a (dimensionless).

    Runtime policy (v2.1): **CPL only**. Provide (w0, wa) already mapped
    via `map_kernel_to_cpl`.
    """
    arr = np.asarray(a, dtype=float)
    if np.any(arr <= 0.0):
        raise ValueError("Scale factor a must be > 0.")
    if eos.model != EoSModel.CPL:
        raise ValueError(f"Unsupported EoS model: {eos.model}; runtime is CPL-only.")
    out = eos.w0 + eos.wa * (1.0 - arr)
    return float(out) if np.ndim(a) == 0 else out


def rho_frsr_over_rho0(a: float, eos: EoSKnobs) -> float:
    """
    Return the dimensionless ratio ρ_FRSR(a) / ρ_FRSR,0 for the CPL runtime model.

    Uses the closed-form integral of the CPL equation of state:
        ρ(a)/ρ0 = a^{-3(1+w0+wa)} · exp[3 wa (a - 1)].

    Args:
        a: Scale factor (dimensionless, > 0).
        eos: CPL equation-of-state knobs (w0, wa).
    """
    if a <= 0.0:
        raise ValueError("Scale factor a must be > 0.")
    if eos.model != EoSModel.CPL:
        raise ValueError(f"Unsupported EoS model: {eos.model}; runtime is CPL-only.")
    expo = -3.0 * (1.0 + eos.w0 + eos.wa)
    return (a ** expo) * math.exp(3.0 * eos.wa * (a - 1.0))


def derive_Omega_frsr0_flat(anchors: Anchors) -> float:
    """
    Flatness relation for the present‑day density parameter:
    \[
        \Omega_{\text{FRSR},0} \equiv \frac{\rho_{\text{FRSR},0}}{\rho_{\text{crit},0}}
        = 1 - (\Omega_{m0} + \Omega_{r0} + \Omega_{k0}).
    \]

    Notation and units:
    • \(\Omega_x \equiv \rho_x / \rho_{\text{crit}}\) is **dimensionless**.
    • \(\rho_{\text{crit},0} = 3H_0^2/(8\pi G)\) (SI: kg·m⁻³; equivalently J·m⁻³ using \(c^2\)).
    """
    return 1.0 - (anchors.Omega_m0 + anchors.Omega_r0 + anchors.Omega_k0)


def Omega_frsr_of_a(a: float, fr: FRBackground) -> float:
    """
    Compute the FRSR density parameter at scale factor \(a\):
    \[
        \Omega_{\text{FRSR}}(a) = \Omega_{\text{FRSR},0}\,\frac{\rho(a)}{\rho_0}.
    \]
    All \(\Omega\)'s are **dimensionless**; \(a\) is dimensionless with \(a{=}1\) today.
    """
    Omega0 = fr.Omega_frsr0 if fr.Omega_frsr0 is not None else derive_Omega_frsr0_flat(fr.anchors)
    return Omega0 * rho_frsr_over_rho0(a, fr.eos)


def H2_std(a: float, A: Anchors) -> float:
    r"""
    Standard background (without FRSR) for the Hubble rate squared:
    \[
        H^2(a) = H_0^2 \Big(\Omega_{r0}a^{-4} + \Omega_{m0}a^{-3} + \Omega_{k0}a^{-2}\Big).
    \]

    Notation and units:
    • \(H_0\): present‑day Hubble parameter (SI: s⁻¹).
    • \(H^2(a)\): returned in SI **s⁻²**.
    • \(a\): dimensionless scale factor.
    • \(\Omega_i\): dimensionless density parameters.
    """
    if a <= 0.0:
        raise ValueError("Scale factor a must be > 0.")
    H0 = A.H0_SI
    return (H0 ** 2) * (A.Omega_r0 / (a ** 4) + A.Omega_m0 / (a ** 3) + A.Omega_k0 / (a ** 2))


def H2_total_eos(a: float, fr: FRBackground) -> float:
    r"""
    Total \(H^2(a)\) in SI (s⁻²) when the FRSR sector is modeled via an EoS:
    \[
        H^2(a) = H_0^2 \left(\Omega_{r0}a^{-4} + \Omega_{m0}a^{-3} + \Omega_{k0}a^{-2}
        + \Omega_{\text{FRSR}}(a)\right),
    \]
    with \(\Omega_{\text{FRSR}}(a)\) built from \(\rho(a)/\rho_0\) using the chosen \(w(a)\).
    """
    H0 = fr.anchors.H0_SI
    Omega_frsr_a = Omega_frsr_of_a(a, fr)
    return (H0 ** 2) * (
        fr.anchors.Omega_r0 / (a ** 4)
        + fr.anchors.Omega_m0 / (a ** 3)
        + fr.anchors.Omega_k0 / (a ** 2)
        + Omega_frsr_a
    )


def H_of_a_eos(a: float, fr: FRBackground) -> float:
    r"""
    Hubble rate \(H(a)\) for the EoS‑based FRSR mode.
    Returns \(H(a)=\sqrt{H^2(a)}\) in SI **s⁻¹**.
    """
    return math.sqrt(H2_total_eos(a, fr))


def delta_H2_over_H0sq(a: float, spec: SpectralKnobs) -> float:
    r"""
    Dimensionless spectral kernel \(\Delta H^2/H_0^2\).

    Kernels:
    • EXP:     \\(\Delta H^2/H_0^2 = A / (1 + \\xi (a + 1/s_c))\\) with \\(\\xi&gt;0\\);
               reduces to \(A/(1+\xi a)\) when \(s_c=0\).
    • POWER2:  \(\Delta H^2/H_0^2 = A / (1 + s_0 a)\) with \(s_0&gt;0\) (effective exponent −1).
    
    Returns a **dimensionless** quantity; \(a\) is dimensionless.
    """
    if spec.kernel == Kernel.NONE or spec.A == 0.0:
        return 0.0
    if a <= 0.0:
        raise ValueError("Scale factor a must be > 0.")

    if spec.kernel == Kernel.EXP:
        if spec.xi <= 0.0:
            raise ValueError("EXP kernel requires xi > 0.")
        sc_term = (1.0 / spec.sc) if (spec.sc and spec.sc > 0.0) else 0.0
        den = 1.0 + spec.xi * (a + sc_term)
        return spec.A / den

    if spec.kernel == Kernel.POWER2:
        if spec.s0 <= 0.0:
            raise ValueError("POWER2 kernel requires s0 > 0.")
        return spec.A / (1.0 + spec.s0 * a)

    raise ValueError(f"Unknown kernel: {spec.kernel}")


def H2_total_spectral(a: float, frs: FRBackgroundSpectral) -> float:
    r"""
    Total \(H^2(a)\) including a spectral \(\Delta H^2\) contribution:
    \[
        H^2(a) = H^2_{\text{std}}(a) + H_0^2\,\Delta H^2/H_0^2.
    \]
    Returned in SI **s⁻²**.
    """
    H0 = frs.anchors.H0_SI
    return H2_std(a, frs.anchors) + (H0 ** 2) * delta_H2_over_H0sq(a, frs.spec)


def H_of_a_spectral(a: float, frs: FRBackgroundSpectral) -> float:
    r"""
    Hubble rate \(H(a)\) for the spectral‑kernel mode.
    Returns \(H(a)=\sqrt{H^2(a)}\) in SI **s⁻¹**.
    """
    return math.sqrt(H2_total_spectral(a, frs))


# ---------------------------------------------------------------------------
# Lightweight background sanity checks
# ---------------------------------------------------------------------------

def assert_background_ok(anchors: Anchors) -> None:
    r"""
    Sanity checks for present‑day background anchors.

    Expectations:
    • \(0 \le \Omega_{b0}, \Omega_{cdm0}, \Omega_{r0} \le 1\) (dimensionless).
    • \(\Omega_{k0}\) within a reasonable range (dimensionless).
    • \(H_0 &gt; 0\) (SI: s⁻¹ when converted via km/s/Mpc → s⁻¹).
    """
    for name, val in [
        ("Omega_b0", anchors.Omega_b0),
        ("Omega_cdm0", anchors.Omega_cdm0),
        ("Omega_r0", anchors.Omega_r0),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {val}")
    if not (-1.0 <= anchors.Omega_k0 <= 1.0):
        raise ValueError(f"Omega_k0 is out of a reasonable range: {anchors.Omega_k0}")
    if anchors.H0_km_s_Mpc <= 0.0:
        raise ValueError("H0 must be > 0.")


def assert_frsr_positive(a_grid: Iterable[float], fr: FRBackground) -> None:
    r"""
    Ensure \(\Omega_{\text{FRSR}}(a) &gt; 0\) on a supplied grid of scale factors \(a\).
    Both \(\Omega_{\text{FRSR}}\) and \(a\) are **dimensionless**.
    """
    for a in a_grid:
        if Omega_frsr_of_a(a, fr) <= 0.0:
            raise ValueError(f"Ω_FRSR(a) must be > 0 for all a; failed at a={a}")
