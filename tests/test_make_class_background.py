import json, os, subprocess, sys


def test_make_ini_from_cli(tmp_run):
    out_ini = tmp_run / "frsr_background.ini"
    cmd = [
        sys.executable, "-m", "frsr.io.frsr_make_class_background",
        "--w0", "-0.98", "--wa", "0.02",
        "--h", "0.674", "--omega-b", "0.048", "--omega-cdm", "0.262", "--omega-fld", "0.69",
        "--out", str(out_ini),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    txt = out_ini.read_text()
    assert "use_tabulated_w  = no" in txt
    assert "w0_fld" in txt and "wa_fld" in txt
    assert "Omega_fld        = 0.69" in txt
    import re
    # No double counting: ensure Omega_Lambda is not set as a key (comments may mention it)
    assert re.search(r'^\s*Omega_Lambda\s*=', txt, re.M) is None
