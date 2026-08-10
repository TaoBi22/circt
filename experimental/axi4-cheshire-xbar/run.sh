#!/usr/bin/env bash
#
# Standalone check of the generated Cheshire AXI crossbar, in six tiers:
#   1. generate     gen_cheshire_xbar.py emits the AXI4-dialect network
#   2. lower        circt-opt --lower-axi4-to-hw
#   3. emit         --lower-seq-to-sv --canonicalize --export-verilog
#   4. structural   the emitted SV has the module and a baked axi_xbar wrapper
#   5. elaborate    verilator --lint-only against the real PULP axi_xbar
#   6. simulate     the generated routing testbench
#
# Path overrides (env): CIRCT_OPT, AXI_ROOT, COMMON_CELLS_ROOT, TECH_CELLS_ROOT.

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
code="$(cd "$repo/.." && pwd)"

CIRCT_OPT="${CIRCT_OPT:-$repo/build/bin/circt-opt}"
AXI_ROOT="${AXI_ROOT:-$code/axi}"
COMMON_CELLS_ROOT="${COMMON_CELLS_ROOT:-$code/common_cells}"
TECH_CELLS_ROOT="${TECH_CELLS_ROOT:-$code/tech_cells_generic}"

build="$here/build"
rm -rf "$build"; mkdir -p "$build"
mlir="$build/cheshire_axi_xbar.mlir"
low="$build/cheshire_axi_xbar.lowered.mlir"
sv="$build/cheshire_axi_xbar.sv"

red=$'\033[31m'; grn=$'\033[32m'; ylw=$'\033[33m'; rst=$'\033[0m'
pass() { echo "  ${grn}PASS${rst} $*"; }
fail() { echo "  ${red}FAIL${rst} $*"; failures=$((failures + 1)); }
skip() { echo "  ${ylw}SKIP${rst} $*"; }
failures=0

[[ -x "$CIRCT_OPT" ]] || { echo "${red}circt-opt not found at $CIRCT_OPT${rst}" >&2; exit 1; }

python3 "$here/gen_cheshire_xbar.py" > "$mlir" || { fail "generate"; exit 1; }
pass "generate ($(grep -c '^    in %slv' "$mlir") managers, $(grep -c '^    out mst' "$mlir") subordinates)"

if ! "$CIRCT_OPT" "$mlir" --lower-axi4-to-hw=pulp-mapping=true -o "$low" 2>"$build/lower.log"; then
  fail "lower"; sed 's/^/      /' "$build/lower.log"; exit 1
fi
pass "lower"

if ! "$CIRCT_OPT" "$low" --lower-seq-to-sv --canonicalize --export-verilog -o /dev/null >"$sv" 2>"$build/ev.log"; then
  fail "export-verilog"; sed 's/^/      /' "$build/ev.log"; exit 1
fi
pass "export-verilog"

ok=1
for needle in "module cheshire_axi_xbar(" "axi_xbar #(" "NoSlvPorts:         14" "NoMstPorts:         7" "NoAddrRules:        9"; do
  grep -qF "$needle" "$sv" || { ok=0; fail "structural: missing \"$needle\""; }
done
grep -qE "axi4\." "$low" && { ok=0; fail "structural: abstract axi4 ops survived"; }
[[ $ok -eq 1 ]] && pass "structural (14 slv / 7 mst / 9 rules)"

elaborate=1
command -v verilator >/dev/null 2>&1 || elaborate=0
for d in "$AXI_ROOT" "$COMMON_CELLS_ROOT"; do [[ -d "$d" ]] || elaborate=0; done
if [[ $elaborate -eq 0 ]]; then
  skip "elaborate (need verilator + $AXI_ROOT + $COMMON_CELLS_ROOT)"
else
  log="$build/verilator.lint.log"
  verilator --lint-only -sv --top-module cheshire_axi_xbar \
    +incdir+"$AXI_ROOT/include" +incdir+"$COMMON_CELLS_ROOT/include" \
    -y "$AXI_ROOT/src" -y "$COMMON_CELLS_ROOT/src" -y "$TECH_CELLS_ROOT/src" \
    -Wno-fatal \
    "$AXI_ROOT/src/axi_pkg.sv" "$COMMON_CELLS_ROOT/src/cf_math_pkg.sv" \
    "$sv" >"$log" 2>&1
  if [[ $? -ne 0 ]]; then
    fail "elaborate (verilator error, see $log)"; grep -m3 '%Error' "$log" | sed 's/^/      /'
  elif grep -E "^%(Error|Warning)[^:]*: .*/cheshire_axi_xbar\.sv:" "$log" >/dev/null; then
    fail "elaborate: verilator flagged the generated SV"
    grep -E "^%(Error|Warning)[^:]*: .*/cheshire_axi_xbar\.sv:" "$log" | sed 's/^/      /'
  else
    pass "elaborate against real axi_xbar ($(grep -c '%Warning' "$log") PULP-internal warnings)"
  fi
fi

tb="$build/tb_cheshire_xbar.sv"
python3 "$here/gen_cheshire_xbar.py" --tb > "$tb"
if [[ $elaborate -eq 1 ]]; then
  simdir="$build/sim"; mkdir -p "$simdir"
  if verilator --cc --exe --main --timing --top-module tb_cheshire_xbar -Wno-fatal \
       +incdir+"$AXI_ROOT/include" +incdir+"$COMMON_CELLS_ROOT/include" \
       -y "$AXI_ROOT/src" -y "$COMMON_CELLS_ROOT/src" -y "$TECH_CELLS_ROOT/src" \
       -Mdir "$simdir/obj" \
       "$tb" "$sv" "$AXI_ROOT/src/axi_pkg.sv" "$COMMON_CELLS_ROOT/src/cf_math_pkg.sv" \
       --build >"$build/sim.build.log" 2>&1; then
    (cd "$simdir" && "$simdir/obj/Vtb_cheshire_xbar") >"$build/sim.run.log" 2>&1
    if grep -q "^PASS:" "$build/sim.run.log" && ! grep -q "FAIL" "$build/sim.run.log"; then
      pass "simulate ($(grep -m1 '^PASS:' "$build/sim.run.log" | sed 's/^PASS: //'))"
    else
      fail "simulate: routing check failed (see $build/sim.run.log)"
      grep -E "FAIL|FAILED" "$build/sim.run.log" | sed 's/^/      /'
    fi
  else
    fail "simulate: verilator build failed (see $build/sim.build.log)"; tail -12 "$build/sim.build.log" | sed 's/^/      /'
  fi
else
  skip "simulate (needs verilator + PULP checkouts)"
fi

echo
[[ $failures -eq 0 ]] && { echo "${grn}all checks passed${rst}"; exit 0; }
echo "${red}$failures check(s) failed${rst}"; exit 1
