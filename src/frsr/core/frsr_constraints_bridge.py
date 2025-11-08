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
Thin wrappers around local physics constraint utilities.
CPL-only mode: no tabulated w(a) files are used anymore.
This bridge computes the ladder summary, runs constraints, and writes a report.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

from .frsr_checks import (
    ladder_from_rows,
    CosmologyBackground,
    validate_constraints,
    suggest_pruned_N,
)
from frsr.utils.log import get_logger

__all__ = [
    "ladder_from_rows",
    "CosmologyBackground",
    "validate_constraints",
    "suggest_pruned_N",
    "run_constraints_and_write",
]

log = get_logger()


def run_constraints_and_write(
    run_dir: str,
    rows: list[tuple],
    N: int,
    r: float,
    alpha: float,
    beta_used: float,
    k_IR: float,
    E0: float,
    q: float,
    Omega_de0: float,
    options: Optional[Dict] = None,
) -> Tuple[bool, Dict]:
    """Create ladder summary, run checks, save JSON; returns (ok, report).
    CPL-only: no w_table is read; constraints operate on the ladder and cosmology background only.
    """
    log.debug(
        "Running constraints (N={}, r={}, alpha={}, beta_used={}, k_IR={}, E0={}, q={}, Omega_de0={})",
        N, r, alpha, beta_used, k_IR, E0, q, Omega_de0
    )
    ladder = ladder_from_rows(
        N=N,
        r=r,
        alpha=alpha,
        beta=beta_used,
        k_IR=k_IR,
        E0=E0,
        rows=rows,
        q=q,
    )
    cosmo = CosmologyBackground(Omega_fld0=Omega_de0)
    ok_constraints, constraints_report = validate_constraints(
        ladder, None, cosmo, options=options or {}
    )
    constraints_path = f"{run_dir}/constraints_report.json"
    with open(constraints_path, "w") as f:
        json.dump(dict(ok=ok_constraints, report=constraints_report), f, indent=2)
    log.success("Constraints report written: {}", constraints_path)
    return ok_constraints, constraints_report
