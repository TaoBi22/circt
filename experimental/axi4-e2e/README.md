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
5. **simulate** (Tier 3, `single` design only) — builds and runs
   `sim/tb_axitop.sv` against the real `axi_xbar`, dumping a waveform of an
   actual AXI4 read completing end to end. Same skip condition as Tier 2.

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

## Tier 3: simulate

`designs/single.mlir`'s `mgr_module` issues a single AXI4 read to address 0;
`sub_module` is a tiny 4-word ROM (only word 0 is ever fetched). `run.sh`
builds `sim/tb_axitop.sv` with verilator (`--trace-vcd`), runs it, and
self-checks that the read completes with the expected ROM word. The waveform
lands at `build/single.sim/tb_axitop.vcd` — open it in gtkwave/surfer to see
the AR/R handshakes flow through the real `axi_xbar`. `multi`/`mixed_fanout`
don't get this treatment: their manager/subordinate modules are independent
stub copies untouched by this, and two `axi4.node` references to the same
symbol always lower to identical hardware, so there's no way to give the two
managers in those designs distinct target addresses without extending the
dialect.

## Known stopgaps validated as-is

The wrapper ties `rst_ni = 1'b1`, hardcodes user width 1, and defaults
`MaxMstTrans`/`MaxSlvTrans`/`LatencyMode` — see the branch's commit history.
Elaboration confirms these bind; it does not exercise reset behavior.

`single.mlir`'s `mgr_module`/`sub_module` registers likewise never reset
(their `seq.compreg` reset inputs are tied permanently false) — they rely on
the simulator's zero-initialized register state at time 0, same stopgap as
`rst_ni` above. `mgr_module`'s `done` output is sticky and never clears.
