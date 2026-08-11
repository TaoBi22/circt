// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s --implicit-check-not=axi4.

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !port)
hw.module.extern @Subordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !port)

// A crossing takes a clock per side, and the single reset it may not cross
// CHECK-LABEL: hw.module.extern @axi_cdc_a32_d64_i4(
// CHECK-SAME:    in %src_clk_i : !seq.clock, in %dst_clk_i : !seq.clock,
// CHECK-SAME:    in %rst_ni : i1,
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %mgr0_rready : i1,
// CHECK-SAME:    in %sub0_awready : i1,
// CHECK-SAME:    in %sub0_rvalid : i1,
// CHECK-SAME:    out mgr0_awready : i1,
// CHECK-SAME:    out mgr0_rvalid : i1,
// CHECK-SAME:    out sub0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    out sub0_rready : i1)

// CHECK-LABEL: hw.module @Crossing(
hw.module @Crossing(in %clk : !seq.clock, in %other_clk : !seq.clock,
                    in %rst_ni : i1) {
  // CHECK: %mgr.axi_aw, {{.*}} = hw.instance "mgr" @Manager(
  // CHECK-SAME: axi_awready: %cdc0.mgr0_awready: i1
  %m = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !port)

  // Each side's clock reaches the port of its own domain
  // CHECK: %cdc0.mgr0_awready, {{.*}} = hw.instance "cdc0" @axi_cdc_a32_d64_i4(
  // CHECK-SAME: src_clk_i: %clk: !seq.clock, dst_clk_i: %other_clk: !seq.clock
  // CHECK-SAME: rst_ni: %rst_ni: i1
  // CHECK-SAME: mgr0_aw: %mgr.axi_aw:
  // CHECK-SAME: sub0_awready: %sub.axi_awready: i1
  %cdc = axi4.cdc from %clk to %other_clk, %rst_ni, %m : !port

  // CHECK: hw.instance "sub" @Subordinate(
  // CHECK-SAME: clk: %other_clk: !seq.clock
  // CHECK-SAME: axi_aw: %cdc0.sub0_aw:
  hw.instance "sub" @Subordinate(clk: %other_clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %cdc: !port) -> ()
}
