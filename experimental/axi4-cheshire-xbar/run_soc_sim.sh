#!/usr/bin/env bash
#
# Drop the generated Cheshire crossbar into a Chimera checkout and run the
# Verilator full-SoC regression:
#   1. generate    the AXI4-dialect model, at the sim build's widths
#   2. lower       circt-opt -> cheshire/hw/cheshire_axi_xbar.sv
#   3. integrate   apply cheshire-xbar-integration.patch (idempotent)
#   4. make chim-setup
#   5. make sim-vlt-build
#   6. make $SIM_TARGET
#
# Usage: ./run_soc_sim.sh [CHIMERA_ROOT]
# Env: CHIMERA_ROOT, CIRCT_OPT, XBAR_MST_ID, XBAR_EXT_MST, SIM_TARGET, FULLTEST,
#      VENV, RISCV_TOOLCHAIN_BIN, SKIP_ENV_SETUP.

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHIMERA="${1:-${CHIMERA_ROOT:-$HOME/Work/chimera}}"
CIRCT_OPT="${CIRCT_OPT:-$here/../../build/bin/circt-opt}"
export XBAR_MST_ID="${XBAR_MST_ID:-3}"
export XBAR_EXT_MST="${XBAR_EXT_MST:-12}"
SIM_TARGET="${SIM_TARGET:-sim-vlt-test}"
FULLTEST="${FULLTEST:-0}"   # set to 1 for the full JTAG/debug test set

red=$'\033[31m'; grn=$'\033[32m'; ylw=$'\033[33m'; rst=$'\033[0m'
info() { echo "${grn}==>${rst} $*"; }
warn() { echo "${ylw}warning:${rst} $*" >&2; }
die()  { echo "${red}error:${rst} $*" >&2; exit 1; }

[[ -d "$CHIMERA" ]] || die "Chimera root not found: $CHIMERA"
CHS="$CHIMERA/cheshire"
[[ -e "$CHS/hw/cheshire_soc.sv" ]] || die "no cheshire checkout at $CHS (run bender first?)"
[[ -x "$CIRCT_OPT" ]] || die "circt-opt not found/executable: $CIRCT_OPT"

if [[ -z "${SKIP_ENV_SETUP:-}" ]]; then
  # Chimera's build needs a GNU userland, morty, a venv and a RISC-V GCC.
  for pkg in gnu-sed coreutils grep gawk findutils; do
    gnubin="$(brew --prefix "$pkg" 2>/dev/null)/libexec/gnubin"
    [[ -d "$gnubin" ]] && PATH="$gnubin:$PATH"
  done
  [[ -d "$HOME/.cargo/bin" ]] && PATH="$HOME/.cargo/bin:$PATH"
  if ! command -v riscv32-corev-elf-gcc >/dev/null 2>&1 &&
     ! command -v riscv32-unknown-elf-gcc >/dev/null 2>&1; then
    rvbin="${RISCV_TOOLCHAIN_BIN:-$(ls -d "$HOME"/Documents/Tools/corev-openhw-gcc-*/bin 2>/dev/null | head -1)}"
    [[ -n "$rvbin" && -d "$rvbin" ]] && PATH="$rvbin:$PATH"
  fi
  export PATH
  VENV="${VENV:-$HOME/Work/venv}"
  # shellcheck disable=SC1091
  [[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"
fi

for t in flock sed; do
  command -v "$t" >/dev/null 2>&1 || warn "$t not on PATH - chim-setup may fail"
done
command -v morty >/dev/null 2>&1 || warn "morty not on PATH - chim-setup may fail (cargo install morty)"
if ! command -v riscv32-corev-elf-gcc >/dev/null 2>&1 &&
   ! command -v riscv32-unknown-elf-gcc >/dev/null 2>&1; then
  warn "no riscv32 GCC on PATH - the SDK build ($SIM_TARGET) will fail (set RISCV_TOOLCHAIN_BIN)"
fi

sv="$CHS/hw/cheshire_axi_xbar.sv"
mlir="$(mktemp)"
trap 'rm -f "$mlir"' EXIT

info "generating AXI4-dialect model (mst-id $XBAR_MST_ID, $((XBAR_EXT_MST + 2)) slave ports)"
python3 "$here/gen_cheshire_xbar.py" > "$mlir" || die "generate failed"

info "lowering + export-verilog -> $sv"
"$CIRCT_OPT" "$mlir" --lower-axi4-to-hw=pulp-mapping=true --lower-seq-to-sv \
  --canonicalize --export-verilog -o /dev/null > "$sv" 2>/dev/null ||
  die "circt-opt lowering failed"

info "applying integration patch to $CHS"
patch="$here/cheshire-xbar-integration.patch"
[[ -f "$patch" ]] || die "integration patch not found: $patch"
if git -C "$CHS" apply --reverse --check "$patch" >/dev/null 2>&1; then
  info "  integration patch already applied"
elif git -C "$CHS" apply --check "$patch" >/dev/null 2>&1; then
  git -C "$CHS" apply "$patch" && info "  applied cheshire-xbar-integration.patch"
else
  die "integration patch does not apply cleanly to $CHS (cheshire rev drift, or another patch already applied?)"
fi

info "make chim-setup"
make -C "$CHIMERA" chim-setup || die "chim-setup failed"

info "make sim-vlt-build FULLTEST=$FULLTEST (Verilator elaboration + compile)"
make -C "$CHIMERA" FULLTEST="$FULLTEST" sim-vlt-build || die "sim-vlt-build failed"

if [[ "$SIM_TARGET" == "sim-vlt-build" ]]; then
  info "SIM_TARGET=sim-vlt-build - stopping after elaboration"
  exit 0
fi

info "make $SIM_TARGET FULLTEST=$FULLTEST"
make -C "$CHIMERA" FULLTEST="$FULLTEST" "$SIM_TARGET"; rc=$?

echo
grep -E "tests passed|tests failed out of|The following tests FAILED" \
  "$CHIMERA"/target/sim/*.log 2>/dev/null | tail -5 || true
[[ $rc -eq 0 ]] && info "${grn}done${rst}" || die "$SIM_TARGET failed (rc=$rc)"
