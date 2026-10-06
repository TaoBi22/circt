// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true --split-input-file | FileCheck %s

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module.extern @Manager(out axi : !port)
hw.module.extern @Subordinate(in %axi : !port)

// A PULP_CONFIG_ attribute overrides a parameter the wrapper sets, and adds one
// it does not. An i1 is a bit, any other integer is decimal, and a string is
// verbatim.
// CHECK:      sv.verbatim.source @axi_demux_1d_a32_d64_i4_usr0.sv
// CHECK-SAME:   axi_demux #(\0A
// CHECK-SAME:     .AxiIdWidth  (4),\0A
// CHECK-SAME:     .MaxTrans    (5),\0A
// CHECK-SAME:     .AxiLookBits (2),\0A
// CHECK-SAME:     .UniqueIds   (1'b1),\0A
// CHECK-SAME:     .SpillAr     (1'b0),\0A
// CHECK-SAME:     .SpillW      (DoSpill)\0A
// CHECK-SAME:   ) i_demux (\0A

// CHECK-LABEL: hw.module @Demux(
// CHECK-NOT:     PULP_CONFIG_
// CHECK:         hw.instance "demux0" @axi_demux_1d_a32_d64_i4_usr0(
hw.module @Demux(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !port)
  %s = axi4.demux %clk, %rst_ni, %m upstream_concurrent_per_id 4 {PULP_CONFIG_AxiLookBits = 2 : i32, PULP_CONFIG_UniqueIds = true, PULP_CONFIG_SpillAr = false, PULP_CONFIG_SpillW = "DoSpill"} : (!port) -> (!port)
  hw.instance "sub" @Subordinate(axi: %s: !port) -> ()
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module.extern @Manager(out axi : !port)
hw.module.extern @Subordinate(in %axi : !sub)

// The crossbar's config naming a field of xbar_cfg_t sets it in Cfg, and any
// other config is a parameter of axi_xbar
// CHECK:      sv.verbatim.source @axi_xbar_1u1d_a32_d64_i4_o4_usr0.sv
// CHECK-SAME:   LatencyMode:        axi_pkg::NO_LATENCY,\0A
// CHECK-SAME:   NoAddrRules:        1,\0A
// CHECK-SAME:   PipelineStages:     2,\0A
// CHECK-SAME:   default:            '0\0A
// CHECK-SAME:   axi_xbar #(\0A
// CHECK-SAME:     .ATOPs         (1'b0),\0A
// CHECK-SAME:     .rule_t        (rule_t)\0A
// CHECK-SAME:   ) i_xbar (\0A
hw.module @Xbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !port)
  %s = axi4.xbar %clk, %rst_ni mgrs %m upstream_concurrent_per_id 4 {PULP_CONFIG_LatencyMode = "axi_pkg::NO_LATENCY", PULP_CONFIG_PipelineStages = 2 : i32, PULP_CONFIG_ATOPs = false} : (!port) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module.extern @Manager(out axi : !port)
hw.module.extern @Subordinate(in %axi : !port)

// Components with the same ports share a wrapper only if their config matches
// too
// CHECK:      hw.module.extern @axi_cut_a32_d64_i4_usr0(
// CHECK:      sv.verbatim.source @axi_cut_a32_d64_i4_usr0.sv
// CHECK-SAME:   .Bypass     (1'b0),\0A
// CHECK:      hw.module.extern @axi_cut_a32_d64_i4_usr0_0(
// CHECK:      sv.verbatim.source @axi_cut_a32_d64_i4_usr0_0.sv
// CHECK-SAME:   .Bypass     (1'b1),\0A

// CHECK-LABEL: hw.module @Cuts(
// CHECK:         hw.instance "cut0" @axi_cut_a32_d64_i4_usr0(
// CHECK:         hw.instance "cut1" @axi_cut_a32_d64_i4_usr0(
// CHECK:         hw.instance "cut2" @axi_cut_a32_d64_i4_usr0_0(
hw.module @Cuts(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !port)
  %a = axi4.cut %clk, %rst_ni, %m : !port
  %b = axi4.cut %clk, %rst_ni, %a : !port
  %c = axi4.cut %clk, %rst_ni, %b {PULP_CONFIG_Bypass = true} : !port
  hw.instance "sub" @Subordinate(axi: %c: !port) -> ()
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module.extern @Manager(out axi : !mgr)
hw.module.extern @Low(in %axi : !lo)
hw.module.extern @High(in %axi : !hi)

// A matrix of booleans is a packed bit array, row 0 and column 0 lowest
// CHECK:      sv.verbatim.source @axi_xbar_2u2d_a32_d64_i4_o5_usr0.sv
// CHECK-SAME:   .Connectivity  ({2'b10, 2'b11}),\0A
hw.module @Connectivity(in %clk : !seq.clock, in %rst_ni : i1) {
  %m0 = hw.instance "mgr0" @Manager() -> (axi: !mgr)
  %m1 = hw.instance "mgr1" @Manager() -> (axi: !mgr)
  %lo, %hi = axi4.xbar %clk, %rst_ni mgrs %m0, %m1 upstream_concurrent_per_id 4 {PULP_CONFIG_Connectivity = [[true, true], [false, true]]} : (!mgr, !mgr) -> (!lo, !hi)
  hw.instance "lo" @Low(axi: %lo: !lo) -> ()
  hw.instance "hi" @High(axi: %hi: !hi) -> ()
}
