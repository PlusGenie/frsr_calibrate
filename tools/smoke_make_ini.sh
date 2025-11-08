#!/usr/bin/env bash
set -euo pipefail
VENV_PY="${PYTHON:-python}"
RUN_DIR="runs/smoke_tools"
mkdir -p "$RUN_DIR"

$VENV_PY -m frsr.io.frsr_make_class_background \
  --w0 -0.98 --wa 0.02 \
  --h 0.674 --omega-b 0.048 --omega-cdm 0.262 --omega-fld 0.69 \
  --out "$RUN_DIR/frsr_background.ini"

echo "[ok] ini written: $RUN_DIR/frsr_background.ini"
grep -q "w0_fld" "$RUN_DIR/frsr_background.ini" && echo "[ok] CPL present"

#!/usr/bin/env bash
set -euo pipefail
VENV_PY="${PYTHON:-python}"
RUN_DIR="runs/smoke_tools"
mkdir -p "$RUN_DIR"

echo "==[1/5] CLASS ini from CPL (frsr_make_class_background)=="
$VENV_PY -m frsr.io.frsr_make_class_background \
  --w0 -0.98 --wa 0.02 \
  --h 0.674 --omega-b 0.048 --omega-cdm 0.262 --omega-fld 0.69 \
  --out "$RUN_DIR/frsr_background.ini"
grep -q "w0_fld" "$RUN_DIR/frsr_background.ini"
echo "[ok] ini written: $RUN_DIR/frsr_background.ini"
echo "[ok] CPL present in ini"

echo
echo "==[2/5] POWER2 → CPL param (s0=1.0)=="
P2_PARAM="$RUN_DIR/frsr_power2.param"
$VENV_PY -m frsr.utils.frsr_mp_hook make-param \
  --out "$P2_PARAM" \
  --h 0.674 --omega-b 0.048 --omega-cdm 0.262 \
  --kernel power2 --s0 1.0
grep -E "w0_fld|wa_fld" "$P2_PARAM"
echo "[ok] POWER2 param generated: $P2_PARAM"

echo
echo "==[3/5] EXP → CPL param (xi=0.8, sc=0.0)=="
EXP_SC0_PARAM="$RUN_DIR/frsr_exp_sc0.param"
$VENV_PY -m frsr.utils.frsr_mp_hook make-param \
  --out "$EXP_SC0_PARAM" \
  --h 0.674 --omega-b 0.048 --omega-cdm 0.262 \
  --kernel exp --xi 0.8 --sc 0.0
grep -E "w0_fld|wa_fld" "$EXP_SC0_PARAM"
echo "[ok] EXP (sc=0.0) param generated: $EXP_SC0_PARAM"

echo
echo "==[4/5] EXP → CPL param (xi=0.8, sc=0.2) and divergence check=="
EXP_SC02_PARAM="$RUN_DIR/frsr_exp_sc02.param"
$VENV_PY -m frsr.utils.frsr_mp_hook make-param \
  --out "$EXP_SC02_PARAM" \
  --h 0.674 --omega-b 0.048 --omega-cdm 0.262 \
  --kernel exp --xi 0.8 --sc 0.2
echo "[sc=0.0]";  grep -E "w0_fld|wa_fld" "$EXP_SC0_PARAM"  | sed 's/^/  /'
echo "[sc=0.2]";  grep -E "w0_fld|wa_fld" "$EXP_SC02_PARAM" | sed 's/^/  /'
if diff -q <(grep -E "w0_fld|wa_fld" "$EXP_SC0_PARAM") <(grep -E "w0_fld|wa_fld" "$EXP_SC02_PARAM") > /dev/null; then
  echo "[fail] EXP sc=0.0 and sc=0.2 produced identical CPL; expected divergence" >&2
  exit 1
else
  echo "[ok] EXP sc switch affects (w0, wa)"
fi

echo
echo "==[5/5] Ladder sanity (kernel → (w0, wa))=="
LADDER_CSV="$RUN_DIR/frsr_ladder.csv"
$VENV_PY - "$LADDER_CSV" <<'PY'
import sys
from pathlib import Path
from frsr.core.frsr_background import map_kernel_to_cpl, SpectralKnobs, Kernel
out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
pairs = [
    ("power2_s0_1.0", SpectralKnobs(kernel=Kernel.POWER2, s0=1.0)),
    ("exp_xi_0.8_sc_0.0", SpectralKnobs(kernel=Kernel.EXP, xi=0.8, sc=0.0)),
    ("exp_xi_0.8_sc_0.2", SpectralKnobs(kernel=Kernel.EXP, xi=0.8, sc=0.2)),
]
with open(out, "w") as f:
    f.write("name,w0,wa\n")
    for name,spec in pairs:
        w0,wa = map_kernel_to_cpl(spec)
        f.write(f"{name},{w0},{wa}\n")
print(out.read_text(), end="")
PY
test_lines=$(wc -l < "$LADDER_CSV")
if [ "$test_lines" -ne 4 ]; then
  echo "[fail] ladder rows != 3 (plus header)" >&2
  exit 1
fi
echo "[ok] ladder generated: $LADDER_CSV"

echo
echo "All smoke sentinels PASSED."