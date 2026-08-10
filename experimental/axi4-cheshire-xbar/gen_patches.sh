#!/usr/bin/env bash
#
# Emit the two self-contained patches for a machine that only clones Chimera:
# the structural edit plus the generated SV, baked at each flow's AXI id widths.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIRCT_OPT="${CIRCT_OPT:-$here/../../build/bin/circt-opt}"
export XBAR_EXT_MST="${XBAR_EXT_MST:-12}"
integration="$here/cheshire-xbar-integration.patch"

[[ -x "$CIRCT_OPT" ]] || { echo "circt-opt not found: $CIRCT_OPT" >&2; exit 1; }
[[ -f "$integration" ]] || { echo "missing structural patch: $integration" >&2; exit 1; }

emit() {  # $1=variant  $2=mst-id
  local variant="$1" mstid="$2"
  local out="$here/chimera-cheshire-axi4-xbar-$variant.patch"
  local tmp; tmp="$(mktemp -d)"; mkdir "$tmp/hw"
  XBAR_MST_ID="$mstid" python3 "$here/gen_cheshire_xbar.py" \
    | "$CIRCT_OPT" --lower-axi4-to-hw=pulp-mapping=true --lower-seq-to-sv \
        --canonicalize --export-verilog -o /dev/null 2>/dev/null \
        > "$tmp/hw/cheshire_axi_xbar.sv"
  # Wrap the generated file as a new-file diff, via a throwaway repo.
  ( cd "$tmp" && git init -q && git add hw/cheshire_axi_xbar.sv \
      && git -c core.autocrlf=false diff --cached ) > "$tmp/sv.diff"
  cat "$integration" "$tmp/sv.diff" > "$out"
  local slv=$((mstid + 4))
  echo "wrote $(basename "$out")  (mst-id $mstid / slv-id $slv, $((XBAR_EXT_MST + 2)) slave ports)"
  rm -rf "$tmp"
}

emit vlt 3
emit vcs 6
