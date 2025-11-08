import os
from frsr.io.frsr_loader import resolve_toml


def test_resolve_toml_minimal(sample_toml):
    cfg = resolve_toml(str(sample_toml))
    p = cfg.to_params_dict()

    assert p["mode"] in ("CPL", "Cpl", "cpl")
    assert p["anchors"]["H0"] == 67.4
    assert abs(p["derived"]["h"] - 0.674) < 1e-9

    assert "model" in p["eos"]
    if "w0" in p["eos"]:
        assert p["eos"]["w0"] == -0.98
    if "wa" in p["eos"]:
        assert p["eos"]["wa"] == 0.0

    # class_params populated with sane defaults
    for k in ("Omega_b", "Omega_cdm", "Omega_r", "Omega_k", "h"):
        assert k in p["class_params"]
