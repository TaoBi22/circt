# Cheshire AXI crossbar from the AXI4 dialect

Experimental, **not upstreamed**. Models Chimera's Cheshire main AXI crossbar in
the AXI4 dialect, lowers it to SystemVerilog, and checks it against the real
PULP `axi_xbar` — a worked example of generating a real SoC interconnect.

The network is the module boundary plus one `axi4.xbar`: every endpoint is a
top-level `!axi4.port`, which the lowering explodes into a payload, a valid and
a ready per channel. Those signals connect straight to the existing
`axi_in_req`/`axi_out_req` buses in `cheshire_soc.sv`, because the channel
payload structs share their layout with PULP's `axi_pkg` channel structs.

## Config (chimera `SelectedCfg=0`, `AXISIM=1`)

Resolved from `cheshire_pkg`/`chimera_pkg`: addr 32, data 32, user 8; **14 slave
ports, 7 master ports, 9 address rules**; `CUT_ALL_PORTS`, full connectivity,
`ATOPs=1`. Slave ports: `[0]=dbg, [1]=dma, [2..13]=`external (the last is the
`AXISIM` preload master; `XBAR_EXT_MST=11` drops it). Master ports: `[0]=dbg,
[1]=reg_demux, [2]=dma, [3..6]=`Dormouse/MemIsl/Cluster0/Cluster1. (This build
has no SPM port.)

Master-port ID width is 3 under `FEWERAXIBITS` and 6 otherwise; the slave side
is the crossbar's widening, mst-id + `clog2(14)`. See the table under
*Config-specific widths* below.

### Where this diverges from Cheshire's `AxiXbarCfg`

Two knobs come out of the dialect rather than being dictated, so the generated
`Cfg` is not identical to the hand-written one:

| | generated (`XBAR_MST_ID=3`) | Cheshire |
|---|---|---|
| `MaxSlvTrans` | 8 | 24 |
| `MaxMstTrans` | 112 | 24 |
| `LatencyMode` | `CUT_ALL_AX` | `CUT_ALL_PORTS` |

`MaxSlvTrans` follows the slave ports' `outstanding_writes`/`outstanding_reads`,
which `!axi4.port` caps at the number of IDs the port has — 8 for a 3-bit ID, so
24 is not expressible there. `MaxMstTrans` is the sum over the slave ports
reaching each master port, which is what `verify-axi4-networks` wants to see.
Both only size flow-control capacity and counters, so the crossbar routes
identically; `CUT_ALL_AX` likewise just inserts fewer register slices than
`CUT_ALL_PORTS`. Worth revisiting for the VCS variant, where the same rule gives
`MaxMstTrans` 336.

## Run

```sh
./run.sh
```

Tiers (each a PASS/FAIL/SKIP line):

1. **generate** — `gen_cheshire_xbar.py` emits the AXI4-dialect network.
2. **lower** — `circt-opt --lower-axi4-to-hw=pulp-mapping=true`.
3. **export-verilog** — `--lower-seq-to-sv --canonicalize --export-verilog`.
4. **structural** — the emitted SV has the module + an `axi_xbar` wrapper baked
   with `NoSlvPorts=14 / NoMstPorts=7 / NoAddrRules=9`, and no `axi4` op
   survived the lowering.
5. **elaborate** — `verilator --lint-only` against the real PULP `axi_xbar` +
   `common_cells` (skipped if verilator / the checkouts are absent).
6. **simulate** — `gen_cheshire_xbar.py --tb` emits a routing testbench: one
   master issues a single-beat read to an address in each master port's window;
   the check asserts the AR reaches that master port and no other, through the
   real `axi_xbar`. The testbench holds PULP `axi_req_t`/`axi_rsp_t` structs and
   connects them field by field, so it also checks the struct layouts the
   `cheshire_soc.sv` instantiation relies on.

## Full-SoC flow

`run_soc_sim.sh` drops the generated crossbar into a Chimera checkout in place of
the hand-written `axi_xbar` and runs the Verilator full-SoC regression:

```sh
./run_soc_sim.sh [CHIMERA_ROOT]      # default ~/Work/chimera
```

Steps: generate → lower/export-verilog → `cheshire/hw/cheshire_axi_xbar.sv`;
apply `cheshire-xbar-integration.patch` (Bender.yml source + the `cheshire_soc.sv`
instance swap, idempotent); `make chim-setup` → `sim-vlt-build` → `sim-vlt-test`.
Set `SIM_TARGET=sim-vlt-build` to stop after elaboration.

### Config-specific widths (important)

The generated module bakes concrete AXI id widths, so it is **simulator-specific**.
`FEWERAXIBITS` is defined only for `PERFORMANCE_MODEL && VERILATOR`, which shrinks
the master id width:

| Target | mst-id / slv-id | `XBAR_MST_ID` |
|---|---|---|
| Verilator (`sim-vlt-*`, perf model) | 3 / 7 | 3 |
| VCS / synthesis / non-perf Verilator | 6 / 10 | 6 |

Both use 14 slave ports (`AXISIM`); the `cheshire_soc.sv` instance connections
name no widths, so only the baked `cheshire_axi_xbar.sv` differs.

### Remote clone-and-apply

For a machine that only clones Chimera (no circt-opt), `gen_patches.sh` emits two
self-contained patches (structural edit + the generated SV):

    chimera-cheshire-axi4-xbar-vlt.patch   # 3/7, for sim-vlt-*
    chimera-cheshire-axi4-xbar-vcs.patch   # 6/10, for sim-vcs-*

Then on the remote, after `bender checkout`:

    git -C "$(bender path cheshire)" apply /path/to/chimera-cheshire-axi4-xbar-<vlt|vcs>.patch
    make sim-<vlt|vcs>-test

## Dependencies

- `circt-opt` at `../../build/bin/` (override with `CIRCT_OPT`).
- `verilator` on `PATH` (tested with 5.051) — Tiers 5-6 only.
- PULP checkouts as siblings of `circt` (override `AXI_ROOT` / `COMMON_CELLS_ROOT`
  / `TECH_CELLS_ROOT`), pinned per axi's `Bender.yml`: `axi@v0.39.10`,
  `common_cells@v1.39.0`.
- `run_soc_sim.sh` additionally needs the Chimera build prerequisites: a GNU
  userland (`gnu-sed`/`coreutils`/`grep`/`gawk`/`findutils`), `flock`, `morty`,
  a python venv with Chimera's `requirements.txt` (`VENV`, default `~/Work/venv`),
  and a RISC-V GCC (`riscv32-corev-elf-gcc`/`riscv32-unknown-elf-gcc`;
  `RISCV_TOOLCHAIN_BIN`). It auto-detects these; pass `SKIP_ENV_SETUP=1` to manage
  the environment yourself.

## Files

- `gen_cheshire_xbar.py` — single source of truth. No args → the AXI4-dialect
  MLIR model; `--tb` → the matching SystemVerilog routing testbench; `--inst` →
  the `cheshire_soc.sv` instantiation (port/index mapping) used to derive the
  integration patch. All three share one table of the fifteen signals a port
  explodes into and where each lives in PULP's req/resp pair.
- `run.sh` — the 6-tier standalone check above; writes to `build/` (gitignored).
- `run_soc_sim.sh` — the full-SoC flow above; `CHIMERA_ROOT` is arg 1.
- `gen_patches.sh` — regenerates the two combined patches below (deterministic,
  checkout-free: generator + circt-opt + the structural patch).
- `cheshire-xbar-integration.patch` — the width-independent structural edit
  (`cheshire_soc.sv` swap + `Bender.yml` source); used by `run_soc_sim.sh`.
- `chimera-cheshire-axi4-xbar-{vlt,vcs}.patch` — standalone combined patches
  (structural edit + baked SV) for remote clone-and-apply.
