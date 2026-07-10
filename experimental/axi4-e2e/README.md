# AXI4 → SystemVerilog end-to-end check

Experimental, **not upstreamed**. Exercises the AXI4 lowering on branch
`axi4-dialect-experimental`: takes an AXI4-dialect design all the way to emitted
SystemVerilog and elaborates the whole stack against the *real* PULP `axi_xbar`.

## What it does

Each `designs/*.mlir` is a network inside `hw.module @AXITop(in %clk_i : i1)`,
plus trivial manager/subordinate stub modules (outputs tied to 0) so the emitted
SystemVerilog is self-contained. For each design, `run.sh`:

1. **lower** — `circt-opt --lower-axi4-to-hw`.
2. **emit** — `circt-opt --export-verilog` → the full design: `AXITop`, the stubs,
   and one `axi_xbar` wrapper per crossbar.
3. **structural** (Tier 1) — assert the emitted SV has the top module and, per
   crossbar, a wrapper instantiating `axi_xbar` with a baked `Cfg` + address map;
   assert no abstract `axi4.` ops survived lowering.
4. **elaborate** (Tier 2) — `verilator --lint-only` with `AXITop` as top,
   resolving the real `axi_xbar` + `common_cells` by library search. One run
   covers the whole stack: `AXITop` glue → wrapper → `axi_xbar` → `common_cells`.
   Skipped if the PULP checkouts or verilator are absent.

Run it:

```sh
./run.sh
```

## Dependencies

- `circt-opt` at `../../build/bin/` (override with `CIRCT_OPT`).
- `verilator` on `PATH` (tested with 5.044).
- PULP checkouts as siblings of `circt` (override with `AXI_ROOT` /
  `COMMON_CELLS_ROOT` / `TECH_CELLS_ROOT`), pinned to the versions in axi's
  `Bender.yml`:
  - `pulp-platform/axi` @ `v0.39.10`
  - `pulp-platform/common_cells` @ `v1.39.0`
  - `pulp-platform/tech_cells_generic` @ `v0.2.2`

```sh
cd .. # sibling of circt
git clone --branch v0.39.10 https://github.com/pulp-platform/axi.git
git clone --branch v1.39.0  https://github.com/pulp-platform/common_cells.git
git clone --branch v0.2.2   https://github.com/pulp-platform/tech_cells_generic.git
```

## Known stopgaps validated as-is

The wrapper ties `rst_ni = 1'b1`, hardcodes user width 1, and defaults
`MaxMstTrans`/`MaxSlvTrans`/`LatencyMode` — see the branch's commit history.
Elaboration confirms these bind; it does not exercise reset behavior.
