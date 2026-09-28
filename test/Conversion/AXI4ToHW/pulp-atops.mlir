// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true --split-input-file | FileCheck %s

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!mem = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>
!periph = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>

hw.module.extern @Core(out axi : !mgr {pulp.atops})
hw.module.extern @Dma(out axi : !mgr)
hw.module.extern @Mem(in %axi : !mem {pulp.atops})
hw.module.extern @Periph(in %axi : !periph)

// A crossbar atomics reach carries atop on every face, and has PULP's support
// for atomics enabled
// CHECK-LABEL: hw.module.extern @axi_xbar_2u2d_a32_d64_i4_o5_atop(
// CHECK-SAME:    in %mgr0_aw_atop : i6,
// CHECK-SAME:    in %mgr1_aw_atop : i6,
// CHECK-SAME:    out sub0_aw_atop : i6,
// CHECK-SAME:    out sub1_aw_atop : i6)
// CHECK:      sv.verbatim.source @axi_xbar_2u2d_a32_d64_i4_o5_atop.sv
// CHECK-SAME:   input  axi_pkg::atop_t mgr0_aw_atop,\0A
// CHECK-SAME:   output axi_pkg::atop_t sub0_aw_atop,\0A
// CHECK-SAME:   region: mgr0_aw.region, atop: mgr0_aw_atop, user: '0};\0A
// CHECK-SAME:   assign sub0_aw_atop = mst_req[0].aw.atop;\0A
// CHECK-SAME:   .ATOPs         (1'b1),\0A

// A cut only the unmarked manager reaches carries no atop
// CHECK-LABEL: hw.module.extern @axi_cut_a32_d64_i5(
// CHECK-NOT:     atop
// CHECK-SAME:  )

// A marked endpoint gets atop alongside its AW channel
// CHECK-LABEL: hw.module.extern @Core(
// CHECK-SAME:    out axi_aw_atop : i6)
// CHECK-LABEL: hw.module.extern @Mem(
// CHECK-SAME:    in %axi_aw_atop : i6,

// Only the marked manager drives atop into the crossbar, and the marked
// subordinate takes it back out
// CHECK-LABEL: hw.module @Mixed(
// CHECK:         %c0_i6 = hw.constant 0 : i6
// CHECK:         hw.instance "xbar0" @axi_xbar_2u2d_a32_d64_i4_o5_atop(
// CHECK-SAME:      mgr0_aw_atop: %core.axi_aw_atop: i6,
// CHECK-SAME:      mgr1_aw_atop: %c0_i6: i6,
// CHECK:         hw.instance "mem" @Mem(
// CHECK-SAME:      axi_aw_atop: %xbar0.sub0_aw_atop: i6
// CHECK:         hw.instance "periph" @Periph(
// CHECK-NOT:       atop
// CHECK-SAME:    )
hw.module @Mixed(in %clk : !seq.clock, in %rst_ni : i1) {
  %c = hw.instance "core" @Core() -> (axi: !mgr)
  %d = hw.instance "dma" @Dma() -> (axi: !mgr)
  %mem, %periph = axi4.xbar %clk, %rst_ni mgrs %c, %d {PULP_CONFIG_Connectivity = [[true, false], [true, true]]} : (!mgr, !mgr) -> (!mem, !periph)
  %cut = axi4.cut %clk, %rst_ni, %periph : !periph
  hw.instance "mem" @Mem(axi: %mem: !mem) -> ()
  hw.instance "periph" @Periph(axi: %cut: !periph) -> ()
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Marked ports of the module the network is described in carry atop across it
// CHECK-LABEL: hw.module @Boundary(
// CHECK-SAME:    in %core_aw_atop : i6,
// CHECK-SAME:    out mem_aw_atop : i6
// CHECK:         %cut0.sub0_aw_atop = hw.instance "cut0" @axi_cut_a32_d64_i4_atop(
// CHECK-SAME:      mgr0_aw_atop: %core_aw_atop: i6
// CHECK:         hw.output
// CHECK-SAME:      %cut0.sub0_aw_atop
hw.module @Boundary(in %clk : !seq.clock, in %rst_ni : i1, in %core : !port {pulp.atops}, out mem : !port {pulp.atops}) {
  %cut = axi4.cut %clk, %rst_ni, %core : !port
  hw.output %cut : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// A demux atomics reach has AtopSupport enabled
// CHECK:      sv.verbatim.source @axi_demux_1d_a32_d64_i4_atop.sv
// CHECK-SAME:   .AtopSupport (1'b1),\0A
hw.module @Demux(in %clk : !seq.clock, in %rst_ni : i1, in %core : !port {pulp.atops}, out mem : !port {pulp.atops}) {
  %demuxed = axi4.demux %clk, %rst_ni, %core : (!port) -> (!port)
  hw.output %demuxed : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Config agreeing with whether atomics reach a component is allowed
// CHECK:      sv.verbatim.source @axi_xbar_1u1d_a32_d64_i4_o4_atop.sv
// CHECK-SAME:   .ATOPs         (1'b1),\0A
hw.module @AgreeingConfig(in %clk : !seq.clock, in %rst_ni : i1, in %core : !port {pulp.atops}, out mem : !port {pulp.atops}) {
  %s = axi4.xbar %clk, %rst_ni mgrs %core {PULP_CONFIG_ATOPs = true} : (!port) -> (!port)
  hw.output %s : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Config can enable support for atomics where none reach
// CHECK:      sv.verbatim.source @axi_demux_1d_a32_d64_i4.sv
// CHECK-SAME:   .AtopSupport (1'b1),\0A
hw.module @EnablingConfig(in %clk : !seq.clock, in %rst_ni : i1, in %core : !port, out mem : !port) {
  %demuxed = axi4.demux %clk, %rst_ni, %core {PULP_CONFIG_AtopSupport = true} : (!port) -> (!port)
  hw.output %demuxed : !port
}
