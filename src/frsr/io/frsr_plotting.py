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
from typing import List, Tuple, Optional
import os
import sys


def maybe_plot(rows: List[Tuple[int, float, float, float, float]], out_png: Optional[str], run_dir: str, logy: bool) -> None:
    if out_png is None:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib not available; skipping plot.", file=sys.stderr)
        return

    i_vals = [r[0] for r in rows]
    k_vals = [r[1] for r in rows]
    Ei_vals = [r[3] for r in rows]
    terms = [r[4] for r in rows]

    base = os.path.splitext(os.path.basename(out_png))[0]
    p_Ei = os.path.join(run_dir, f"{base}_Ei.png")
    p_ki = os.path.join(run_dir, f"{base}_ki.png")
    p_term = os.path.join(run_dir, f"{base}_term.png")
    p_panel = os.path.join(run_dir, f"{base}_panel.png")

    # Plot 1: Ei
    plt.figure()
    plt.plot(i_vals, Ei_vals, marker="o")
    plt.xlabel("Band index i")
    plt.ylabel(r"Entanglement weight $\mathcal{E}_i$")
    plt.title("FRSR: Entanglement weights per band")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(p_Ei, dpi=160)

    # Plot 2: k_i
    plt.figure()
    plt.plot(i_vals, k_vals, marker="s")
    plt.xlabel("Band index i")
    plt.ylabel(r"$k_i$ (m$^{-1}$)")
    plt.title("FRSR: Central momenta per band")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(p_ki, dpi=160)

    # Plot 3: term_i
    plt.figure()
    plt.plot(i_vals, terms, marker="^")
    plt.xlabel("Band index i")
    plt.ylabel(r"$\mathrm{term}_i = \mathcal{E}_i\, k_i^3\, \Delta k_i$")
    if logy:
        plt.yscale("log")
    plt.title("FRSR: Contribution per band")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(p_term, dpi=160)

    # Combined panel
    import matplotlib.pyplot as plt2

    fig, axs = plt2.subplots(1, 3, figsize=(12, 3.5))
    axs[0].plot(i_vals, Ei_vals, marker="o")
    axs[0].set_title(r"$\mathcal{E}_i$")
    axs[0].set_xlabel("i")
    axs[0].set_ylabel(r"$\mathcal{E}_i$")
    axs[0].grid(True)

    axs[1].plot(i_vals, k_vals, marker="s")
    axs[1].set_title("kᵢ (m⁻¹)")
    axs[1].set_xlabel("i")
    axs[1].grid(True)

    axs[2].plot(i_vals, terms, marker="^")
    axs[2].set_title(r"$\mathrm{term}_i$")
    axs[2].set_xlabel("i")
    axs[2].grid(True)
    if logy:
        axs[2].set_yscale("log")

    fig.suptitle("FRSR ladder summary", y=1.02)
    fig.tight_layout()
    fig.savefig(p_panel, dpi=180)

    print(f"[ok] Plots saved:\n  {p_Ei}\n  {p_ki}\n  {p_term}\n  {p_panel}")
