# -*- coding: utf-8 -*-
#
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
#
# Module: frsr.utils.log
# Purpose: Single, shared logging + SymPy printing initializer for all modules.

from __future__ import annotations

import os
import sys
from typing import Optional

from loguru import logger

_INITIALIZED = False


def init_logging(
    level: Optional[str] = None,
    *,
    colorize: bool = True,
    backtrace: bool = False,
    diagnose: bool = False,
    init_sympy: bool = True,
) -> logger.__class__:
    """
    Initialize one Loguru sink globally and (optionally) SymPy printing.

    Log level priority:
      1) `level` arg
      2) env FRSR_LOG_LEVEL
      3) env FRSR_DEBUG -> DEBUG
      4) default INFO

    Env knobs:
      - FRSR_LOG_LEVEL: INFO|DEBUG|WARNING|ERROR|CRITICAL
      - FRSR_DEBUG: if set (any non-empty), forces DEBUG (unless `level` provided)
      - FRSR_SYMPY_LATEX: "1" to prefer LaTeX printing
      - FRSR_SYMPY_UNICODE: "0" to disable unicode pretty printing
    """
    # -------- choose level --------
    env_level = os.getenv("FRSR_LOG_LEVEL")
    if level:
        level_final = str(level).upper()
    elif env_level:
        level_final = env_level.upper()
    elif os.getenv("FRSR_DEBUG"):
        level_final = "DEBUG"
    else:
        level_final = "INFO"

    # -------- reset + add sink --------
    logger.remove()
    logger.add(
        sys.stderr,
        level=level_final,
        colorize=colorize,
        backtrace=backtrace,
        diagnose=diagnose,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <7}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # -------- optional SymPy pretty/latex printing --------
    if init_sympy:
        try:
            import sympy as sp  # type: ignore

            use_latex = os.getenv("FRSR_SYMPY_LATEX", "0") == "1"
            use_unicode = os.getenv("FRSR_SYMPY_UNICODE", "1") != "0"

            if use_latex:
                # Use LaTeX strings (no external renderer hook here)
                sp.init_printing(use_latex=True, latex_mode="plain")
            else:
                sp.init_printing(use_unicode=use_unicode, pretty_print=True)

            logger.debug(
                "SymPy printing initialized (latex={}, unicode={})",
                use_latex,
                use_unicode,
            )
        except Exception as e:  # pragma: no cover
            # Never fail the app because of printing setup.
            logger.debug("SymPy printing init skipped: {}", e)

    logger.info("Logging initialized at level {}", level_final)
    global _INITIALIZED
    _INITIALIZED = True
    return logger


def set_level(level: str) -> None:
    """Re-initialize logging with a different level."""
    init_logging(level=level)


def get_logger(level: Optional[str] = None) -> logger.__class__:
    """
    Convenience accessor so callers can do:

        from frsr.utils.log import init_logging, get_logger
        init_logging()  # once, in entrypoint
        log = get_logger()
        log.info("Hello")
    """
    global _INITIALIZED
    if level:
        init_logging(level=level)
    elif not _INITIALIZED:
        init_logging()
    return logger


# --- helpers to format SymPy expressions for logs --------------------------------

def sympy_to_text(expr) -> str:
    """
    Best-effort textual representation of a SymPy expression for logs.
    Honors FRSR_SYMPY_LATEX=1 to prefer LaTeX (wrapped in $...$).
    Falls back to sstr()/str().
    """
    try:
        import sympy as sp  # type: ignore
        if os.getenv("FRSR_SYMPY_LATEX", "0") == "1":
            from sympy import latex  # type: ignore
            return f"${latex(sp.simplify(expr))}$"
        else:
            from sympy.printing import sstr  # type: ignore
            return sstr(sp.simplify(expr))
    except Exception:  # pragma: no cover
        return str(expr)
