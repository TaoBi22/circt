# AXI4 → SystemVerilog end-to-end check

Experimental, **not upstreamed**. Takes an AXI4-dialect design all the way to
emitted SystemVerilog and elaborates the whole stack against the *real* PULP
`axi_xbar`.

## What it does

Each `designs/*.mlir` is a network inside
`hw.module @AXITop(in %clk_i : !seq.clock, in %rst_ni : i1)`, plus the
manager/subordinate endpoint modules it instantiates, so the emitted
SystemVerilog is self-contained. An endpoint declares one `!axi4.port` and
bridges it to its own FSM with `axi4.channel_structs_to_port` (manager) or
`axi4.port_to_channel_structs` (subordinate); the lowering materializes the
mirror-image bridge at the module boundary and the pair cancels, so no `axi4`
op survives. For each design, `run.sh`:

1. **lower** — `circt-opt --lower-axi4-to-hw=pulp-mapping=true`.
2. **emit** — `circt-opt --lower-seq-to-sv --canonicalize --export-verilog` → the
   full design: `AXITop`, the endpoints, and one `axi_xbar` wrapper per
   crossbar.
3. **structural** (Tier 1) — assert the emitted SV has the top module and, per
   crossbar, a wrapper instantiating `axi_xbar` with a baked `Cfg` + address map;
   assert no `axi4` op survived lowering.
4. **elaborate** (Tier 2) — `verilator --lint-only` with `AXITop` as top,
   resolving the real `axi_xbar` + `common_cells` by library search. One run
   covers the whole stack: `AXITop` glue → wrapper → `axi_xbar` → `common_cells`.
   Skipped if the PULP checkouts or verilator are absent.
5. **simulate** (Tier 3, if a matching `sim/tb_axitop_$name.sv` exists) —
   builds and runs it against the real `axi_xbar`, dumping a waveform of
   actual AXI4 burst traffic completing end to end. Same skip condition as
   Tier 2; all three designs currently have a matching testbench.

Run it:

```sh
./run.sh
```

## Dependencies

- `circt-opt` at `../../build/bin/` (override with `CIRCT_OPT`).
- `verilator` on `PATH` (tested with 5.051).
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
against the real `axi_xbar`; a design without one would be skipped
automatically (no design is currently in that state).

- **`single`**: `designs/single.mlir`'s `mgr_module` runs a 7-phase
  read-write-read sequence: a 4-beat AXI4 INCR read burst from address 0, a
  2-beat INCR write burst overwriting words 1 and 2 (address 8) with new
  data, then a second 4-beat read burst re-reading all 4 words.
  `sub_module` is a real 4-word read/write RAM (not a ROM) whose starting
  contents are seeded by reset. `run.sh` builds
  `sim/tb_axitop_single.sv` with verilator (`--trace-vcd`), runs it, and
  self-checks both the pre-write read (all 4 original words) and the
  post-write read (words 1 and 2 changed, 0 and 3 unchanged) — a genuine
  read-after-write check through the real crossbar, not just a read-only
  smoke test. Waveform: `build/single.sim/tb_axitop_single.vcd`.

- **`multi`**: `designs/multi.mlir` has 2 managers (`mgr_module_a`,
  `mgr_module_b`), each running the same 7-phase read-write-read sequence as
  `single.mlir`'s `mgr_module` — a 4-beat AXI4 INCR read burst, a 2-beat INCR
  write burst overwriting words 1 and 2, then a second 4-beat read burst —
  concurrently, through the *same* shared crossbar, against 2 different
  subordinates (`sub_module5_a` at address 0, `sub_module5_b` at address
  4096), each a real 4-word read/write RAM (not a ROM) seeded by reset with
  its own distinct starting words. This proves the real
  `axi_xbar`'s address-based routing, inter-manager arbitration, and
  downstream id-widening (4 → 5 bits, so the xbar can disambiguate which
  manager an in-flight R response belongs to) hold up under concurrent
  read-after-write traffic, not just single-flow correctness. `run.sh` builds
  `sim/tb_axitop_multi.sv`, runs it, and self-checks both managers' pre-write
  and post-write reads — they may finish on different cycles depending on
  xbar-internal arbitration/pipeline registers — against their respective,
  distinct expected words. Waveform: `build/multi.sim/tb_axitop_multi.vcd`.

- **`mixed_fanout`**: `designs/mixed_fanout.mlir` has 1 manager (`mgr_module`)
  running the same 7-phase read-write-read sequence as `single.mlir`'s
  `mgr_module` — a 4-beat AXI4 INCR read burst, a 2-beat INCR write burst
  overwriting words 1 and 2, then a second 4-beat read burst — *sequentially
  twice*: first through one crossbar (`xbar1`) direct to `sub_module_a` at
  address 0, then through a chained second crossbar (`xbar2`) to
  `sub_module_b` at address 4096, each subordinate a real 4-word read/write
  RAM (not a ROM) seeded by reset with its own distinct starting words. This
  proves both fan-out paths of the real `axi_xbar` support
  read-after-write consistency, not just single-flow correctness. `run.sh`
  builds `sim/tb_axitop_mixed_fanout.sv`, runs it, and self-checks both
  sequential paths' pre-write and post-write reads against their respective,
  distinct expected words. Waveform:
  `build/mixed_fanout.sim/tb_axitop_mixed_fanout.vcd`.

Open the waveforms in gtkwave/surfer to see the AR/R handshakes (including
the multi-beat `rlast` sequencing, and for `multi`/`mixed_fanout`, traffic
sharing the crossbar) flow through the real `axi_xbar`.

## Reset

`AXITop` takes a real active-low `rst_ni` and passes it to every endpoint
instance and to `axi4.xbar`, and each testbench asserts it for a few
cycles at startup — long enough for the real `axi_xbar` to reset too. The
manager/subordinate FSM registers reset to their initial state, and each
`sub_module`'s 4 RAM words are seeded to their identifiable
`0xCAFEF00D...`/etc. starting values straight from the `seq.compreg` reset
operand (no `seq.initial` needed).

## ID widths

A crossbar widens its downstream ID by `clog2` of its *upstream manager* count,
so `mixed_fanout`'s two single-manager crossbars do not widen at all (4 → 4)
where `multi`'s two-manager crossbar does (4 → 5). `checkPulpSupported` rejects
anything else, because PULP's `axi_mux` asserts on the width it is handed.

## Known stopgaps validated as-is

Each `mgr_module`'s `done` output is sticky and never clears, and every
`sub_module` ignores `wstrb` (full-word writes only, no partial-strobe
support).
