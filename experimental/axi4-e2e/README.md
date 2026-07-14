# AXI4 → SystemVerilog end-to-end check

Experimental, **not upstreamed**. Exercises the AXI4 lowering on branch
`axi4-dialect-experimental`: takes an AXI4-dialect design all the way to emitted
SystemVerilog and elaborates the whole stack against the *real* PULP `axi_xbar`.

## What it does

Each `designs/*.mlir` is a network inside `hw.module @AXITop(in %clk_i : i1)`,
plus trivial manager/subordinate stub modules (outputs tied to 0) so the emitted
SystemVerilog is self-contained. For each design, `run.sh`:

1. **lower** — `circt-opt --lower-axi4-to-hw`.
2. **emit** — `circt-opt --lower-seq-to-sv --canonicalize --export-verilog` → the
   full design: `AXITop`, the stubs, and one `axi_xbar` wrapper per crossbar.
   `--canonicalize` is required once any module uses `seq.initial`-preloaded
   registers (see `single`'s Tier 3 entry below) — `--lower-seq-to-sv` alone
   leaves a dead, unused `builtin.unrealized_conversion_cast` behind that a bare
   `--export-verilog` refuses to emit.
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
   Tier 2; all three designs currently have a matching testbench.

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
against the real `axi_xbar`; a design without one would be skipped
automatically (no design is currently in that state).

- **`single`**: `designs/single.mlir`'s `mgr_module` runs a 7-phase
  read-write-read sequence: a 4-beat AXI4 INCR read burst from address 0, a
  2-beat INCR write burst overwriting words 1 and 2 (address 8) with new
  data, then a second 4-beat read burst re-reading all 4 words.
  `sub_module` is a real 4-word read/write RAM (not a ROM) preloaded via
  `seq.initial` (see "Known stopgaps" below for why that's necessary — the
  `seq.compreg` reset operand alone can't do it here). `run.sh` builds
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
  4096), each a real 4-word read/write RAM (not a ROM) preloaded via
  `seq.initial` with its own distinct starting words. This proves the real
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
  RAM (not a ROM) preloaded via `seq.initial` with its own distinct starting
  words. This proves both fan-out paths of the real `axi_xbar` support
  read-after-write consistency, not just single-flow correctness. `run.sh`
  builds `sim/tb_axitop_mixed_fanout.sv`, runs it, and self-checks both
  sequential paths' pre-write and post-write reads against their respective,
  distinct expected words. Waveform:
  `build/mixed_fanout.sim/tb_axitop_mixed_fanout.vcd`.

Open the waveforms in gtkwave/surfer to see the AR/R handshakes (including
the multi-beat `rlast` sequencing, and for `multi`/`mixed_fanout`, traffic
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
(their `seq.compreg` reset inputs are tied permanently false). `mgr_module`'s
`done` output is sticky and never clears. Unlike the other designs,
`sub_module`'s 4 RAM word registers can't rely on the simulator's
zero-initialized state to get their identifiable starting values (they need
to start as the *specific*, distinct `0xCAFEF00D...`/etc. words, not just
"some deterministic value") — the `seq.compreg` reset operand is dead code
here (reset is never asserted, same as every other register in this
harness), so each word register is preloaded via a separate `seq.initial`
region and the compreg's `initial` clause instead. Don't confuse the two:
the `reset`/`resetValue` operand and the `initial`/`initialValue` operand on
`seq.compreg` are independent — only the latter actually seeds a register's
simulation-time value when reset is never asserted. `sub_module` also
ignores `wstrb` entirely (full-word writes only, no partial-strobe support).

`multi.mlir`'s `mgr_module_a`/`mgr_module_b`/`sub_module5_a`/`sub_module5_b`
carry the same never-reset stopgap. Running two concurrent instances of this
pattern doesn't introduce any new cross-instance risk: each manager/
subordinate pair's FSM state is private to that instance, so there's no
shared mutable state for the two concurrent read-write-read sequences to race
on outside of the real `axi_xbar`'s own (already-vendored, out-of-scope)
internal arbitration/id-tracking logic. Like `single.mlir`'s `sub_module`,
`sub_module5_a`/`sub_module5_b`'s 4 RAM word registers are preloaded via a
separate `seq.initial` region and the compreg's `initial` clause per word
(the `reset`/`resetValue` operand is dead code here too, since reset is never
asserted) — see `single.mlir`'s stopgap note above for why that's the only
mechanism that actually seeds a register's simulation-time value. Both
subordinates also ignore `wstrb` (full-word writes only, no partial-strobe
support).

`mixed_fanout.mlir`'s `mgr_module`/`sub_module_a`/`sub_module_b` carry the
same never-reset stopgap. `mgr_module`'s two-sequence sequencer is a single
14-phase FSM (`phase_q`) driving both read-write-read sequences one after
another, not two independent instances, so unlike `multi.mlir` there
genuinely is one piece of shared state across the two sequences — but it's a
plain sequential handoff (sub_module_b's sequence only ever starts after
sub_module_a's final `rlast`), not a concurrency hazard. Like `single.mlir`'s
`sub_module`, `sub_module_a`/`sub_module_b`'s 4 RAM word registers are
preloaded via a separate `seq.initial` region and the compreg's `initial`
clause per word (the `reset`/`resetValue` operand is dead code here too,
since reset is never asserted) — see `single.mlir`'s stopgap note above for
why that's the only mechanism that actually seeds a register's
simulation-time value. Both subordinates also ignore `wstrb` (full-word
writes only, no partial-strobe support).
