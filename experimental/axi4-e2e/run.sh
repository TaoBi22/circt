#!/usr/bin/env bash
#
# End-to-end check for the AXI4 SV lowering (experimental, not upstreamed):
# takes an AXI4-dialect design all the way to emitted SystemVerilog and
# elaborates the whole stack against the real PULP axi_xbar.
#
# For each design in designs/*.mlir (a network inside an hw.module @AXITop):
#   1. lower       --lower-axi4-to-hw
#   2. emit        --lower-seq-to-sv --export-verilog -> the full design (top
#                  module + xbar wrappers)
#   3. structural  assert the emitted SV has the wrapper pieces we expect (Tier 1)
#   4. elaborate   verilator --lint-only with AXITop as top, resolving the real
#                  axi_xbar + common_cells by library search (Tier 2, skipped if
#                  the checkouts / verilator are absent). One run covers the whole
#                  stack: AXITop glue -> wrapper -> axi_xbar -> common_cells. The
#                  designs carry their own manager/subordinate modules, so the
#                  emitted SV is self-contained.
#   5. simulate    if a matching sim/tb_axitop_$name.sv testbench exists for
#                  this design, build and run it against the real axi_xbar,
#                  dumping a waveform of the AXI4 traffic (Tier 3, same
#                  availability gate as Tier 2). Designs without a matching
#                  testbench (currently mixed_fanout) are skipped.
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

red=$'\033[31m'; grn=$'\033[32m'; ylw=$'\033[33m'; rst=$'\033[0m'
pass() { echo "  ${grn}PASS${rst} $*"; }
fail() { echo "  ${red}FAIL${rst} $*"; failures=$((failures + 1)); }
skip() { echo "  ${ylw}SKIP${rst} $*"; }
failures=0

if [[ ! -x "$CIRCT_OPT" ]]; then
  echo "${red}circt-opt not found at $CIRCT_OPT${rst} (build it, or set CIRCT_OPT)" >&2
  exit 1
fi

# Tier 2 is available only if verilator and the PULP checkouts are present.
elaborate=1
command -v verilator >/dev/null 2>&1 || elaborate=0
for d in "$AXI_ROOT" "$COMMON_CELLS_ROOT"; do [[ -d "$d" ]] || elaborate=0; done
if [[ $elaborate -eq 0 ]]; then
  echo "${ylw}verilator and/or PULP checkouts not found - elaboration (Tier 2) skipped${rst}"
  echo "  expected verilator on PATH, and $AXI_ROOT + $COMMON_CELLS_ROOT"
  echo "  (clone pulp-platform/{axi@v0.39.10,common_cells@v1.39.0} as siblings of circt)"
fi

for design in "$here"/designs/*.mlir; do
  name="$(basename "$design" .mlir)"
  echo "== $name =="
  low="$build/$name.lowered.mlir"
  sv="$build/$name.sv"

  if ! "$CIRCT_OPT" "$design" --lower-axi4-to-hw -o "$low" 2>"$build/$name.lower.log"; then
    fail "lower $name"; sed 's/^/      /' "$build/$name.lower.log"; continue
  fi
  pass "lower $name"

  if ! "$CIRCT_OPT" "$low" --lower-seq-to-sv --export-verilog -o /dev/null >"$sv" 2>"$build/$name.ev.log"; then
    fail "export-verilog $name"; sed 's/^/      /' "$build/$name.ev.log"; continue
  fi
  pass "export-verilog $name"

  # By convention each design's network lives in hw.module @AXITop; the other
  # modules are the trivial manager/subordinate stubs.
  top=AXITop
  grep -q "hw.module @$top" "$design" || { fail "$name: no hw.module @$top"; continue; }

  # Structural (Tier 1): the emitted SV carries the top module and, for each xbar,
  # a wrapper instantiating axi_xbar with a baked Cfg + address map.
  ok=1
  for needle in "module $top(" "axi_xbar #(" "axi_pkg::xbar_cfg_t Cfg" "AddrMap = '{"; do
    grep -qF "$needle" "$sv" || { ok=0; fail "structural $name: missing \"$needle\""; }
  done
  grep -q "axi4\." "$low" && { ok=0; fail "structural $name: abstract axi4 ops survived lowering"; }
  [[ $ok -eq 1 ]] && pass "structural $name"

  if [[ $elaborate -eq 0 ]]; then
    skip "elaborate $name"; continue
  fi

  log="$build/$name.verilator.log"
  verilator --lint-only -sv --top-module "$top" \
    +incdir+"$AXI_ROOT/include" +incdir+"$COMMON_CELLS_ROOT/include" \
    -y "$AXI_ROOT/src" -y "$COMMON_CELLS_ROOT/src" -y "$TECH_CELLS_ROOT/src" \
    -Wno-fatal \
    "$AXI_ROOT/src/axi_pkg.sv" "$COMMON_CELLS_ROOT/src/cf_math_pkg.sv" \
    "$sv" >"$log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    fail "elaborate $name (verilator rc=$rc, see $log)"
    grep -m3 '%Error' "$log" | sed 's/^/      /'
    continue
  fi
  # Flag only diagnostics whose primary location is our generated SV; the "... note:
  # In file included from" trace lines name it merely because it is the top file.
  if grep -E "^%(Error|Warning)[^:]*: .*/$name\.sv:" "$log" >/dev/null; then
    fail "elaborate $name: verilator flagged the generated SV"
    grep -E "^%(Error|Warning)[^:]*: .*/$name\.sv:" "$log" | sed 's/^/      /'
    continue
  fi
  warns="$(grep -c '%Warning' "$log")"
  pass "elaborate $name ($top) against real axi_xbar (${warns} PULP-internal warnings)"

  # Tier 3 (simulate): if a matching hand-written testbench exists for this
  # design, build and run it against the real axi_xbar, dumping a waveform of
  # the AXI4 traffic. See sim/tb_axitop_single.sv / sim/tb_axitop_multi.sv.
  tb="$here/sim/tb_axitop_$name.sv"
  if [[ -f "$tb" ]]; then
    top_tb="tb_axitop_$name"
    simdir="$build/$name.sim"
    mkdir -p "$simdir"
    if verilator --cc --exe --main --timing --trace-vcd --top-module "$top_tb" \
         -Wno-fatal \
         +incdir+"$AXI_ROOT/include" +incdir+"$COMMON_CELLS_ROOT/include" \
         -y "$AXI_ROOT/src" -y "$COMMON_CELLS_ROOT/src" -y "$TECH_CELLS_ROOT/src" \
         -Mdir "$simdir/obj" \
         "$tb" "$sv" "$AXI_ROOT/src/axi_pkg.sv" "$COMMON_CELLS_ROOT/src/cf_math_pkg.sv" \
         --build >"$build/$name.simulate.log" 2>&1; then
      if (cd "$simdir" && "$simdir/obj/V$top_tb") >"$build/$name.simulate.run.log" 2>&1; then
        pass "simulate $name (waveform: $simdir/$top_tb.vcd)"
      else
        fail "simulate $name (dut mismatch/timeout, see $build/$name.simulate.run.log)"
        tail -5 "$build/$name.simulate.run.log" | sed 's/^/      /'
      fi
    else
      fail "simulate $name: verilator build failed (see $build/$name.simulate.log)"
    fi
  else
    skip "simulate $name (no sim/tb_axitop_$name.sv)"
  fi
done

echo
if [[ $failures -eq 0 ]]; then
  echo "${grn}all checks passed${rst}"; exit 0
fi
echo "${red}$failures check(s) failed${rst}"; exit 1
