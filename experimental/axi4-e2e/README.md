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
5. **simulate** (Tier 3, if a matching `sim/tb_axitop_$name.sv` exists) —
   builds and runs it against the real `axi_xbar`, dumping a waveform of
   actual AXI4 burst traffic completing end to end. Same skip condition as
   Tier 2; designs without a matching testbench (currently `mixed_fanout`)
   are skipped automatically.

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

Each design with a matching `sim/tb_axitop_$name.sv` gets built and run
against the real `axi_xbar`; designs without one (currently `mixed_fanout`)
are skipped automatically — their manager/subordinate modules are still the
original stub copies, and two `axi4.node` references to the same symbol
always lower to identical hardware, so there's no way to give the two
managers in that design distinct target addresses without extending the
dialect.

- **`single`**: `designs/single.mlir`'s `mgr_module` issues a single 4-beat
  AXI4 INCR burst read starting at address 0; `sub_module` is a tiny 4-word
  ROM that streams all four words back across the burst. `run.sh` builds
  `sim/tb_axitop_single.sv` with verilator (`--trace-vcd`), runs it, and
  self-checks that the burst completes with the four expected ROM words.
  Waveform: `build/single.sim/tb_axitop_single.vcd`.

- **`multi`**: `designs/multi.mlir` has 2 managers (`mgr_module_a`,
  `mgr_module_b`) issuing concurrent 4-beat AXI4 INCR burst reads through the
  *same* shared crossbar to 2 different subordinates (`sub_module5_a` at
  address 0, `sub_module5_b` at address 4096), each with its own distinct
  4-word ROM. This proves the real `axi_xbar`'s address-based routing,
  inter-manager arbitration, and downstream id-widening (4 → 5 bits, so the
  xbar can disambiguate which manager an in-flight R response belongs to)
  actually work end to end, not just single-flow correctness. `run.sh` builds
  `sim/tb_axitop_multi.sv`, runs it, and self-checks that both bursts
  complete — they may finish on different cycles depending on xbar-internal
  arbitration/pipeline registers — with their respective, distinct ROM
  contents. Waveform: `build/multi.sim/tb_axitop_multi.vcd`.

Open the waveforms in gtkwave/surfer to see the AR/R handshakes (including
the multi-beat `rlast` sequencing, and for `multi`, both managers' traffic
sharing the crossbar) flow through the real `axi_xbar`.

Fragility note (both testbenches): the hierarchical `dut.<instance>.done` /
`.beatN` references rely on `AXI4ToHW.cpp`'s node-instance naming
(`<module>_<counter>`, one shared counter across managers/subordinates/xbars,
assigned in `axi4.node` textual/program order — see `NetworkLowering`'s
`instanceCounter` in `lib/Conversion/AXI4ToHW/AXI4ToHW.cpp`). Reordering the
`axi4.node` ops in a design (or inserting new ones before them) silently
shifts every instance name after that point; the testbench then fails at
elaboration with an unresolved hierarchical reference, not at lowering time.

## Known stopgaps validated as-is

The wrapper ties `rst_ni = 1'b1`, hardcodes user width 1, and defaults
`MaxMstTrans`/`MaxSlvTrans`/`LatencyMode` — see the branch's commit history.
Elaboration confirms these bind; it does not exercise reset behavior.

`single.mlir`'s `mgr_module`/`sub_module` registers likewise never reset
(their `seq.compreg` reset inputs are tied permanently false) — they rely on
the simulator's zero-initialized register state at time 0, same stopgap as
`rst_ni` above. `mgr_module`'s `done` output is sticky and never clears.

`multi.mlir`'s `mgr_module_a`/`mgr_module_b`/`sub_module5_a`/`sub_module5_b`
carry the same never-reset stopgap. Running two concurrent instances of this
pattern doesn't introduce any new cross-instance risk: each manager/
subordinate pair's FSM state is private to that instance, so there's no
shared mutable state for the two concurrent bursts to race on outside of the
real `axi_xbar`'s own (already-vendored, out-of-scope) internal
arbitration/id-tracking logic.
