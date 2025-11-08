import os, pathlib, textwrap, tempfile

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent


@pytest.fixture
def tmp_run(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture
def sample_toml(tmp_path):
    p = tmp_path / "frsr.sample.toml"
    p.write_text(textwrap.dedent(
        """\
        [frsr.anchors]
        H0 = 67.4
        Omega_b0 = 0.048
        Omega_cdm0 = 0.262
        Omega_r0 = 5e-5
        Omega_k0 = 0.0

        [frsr.model]
        family = "CPL"
        w0 = -0.98
        wa = 0.00
        cs2 = 1.0

        [class.background]
        # leave Omega_fld empty → loader can infer residual hint or user sets later
    """
    ))
    return p
