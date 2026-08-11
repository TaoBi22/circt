// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s --implicit-check-not=axi4.

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !port)
hw.module.extern @Subordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !port)

// One external module per cut shape, with the upstream port group named after
// the manager it faces and the downstream one after the subordinate
// CHECK-LABEL: hw.module.extern @axi_cut_a32_d64_i4(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %mgr0_rready : i1,
// CHECK-SAME:    in %sub0_awready : i1,
// CHECK-SAME:    in %sub0_rvalid : i1,
// CHECK-SAME:    out mgr0_awready : i1,
// CHECK-SAME:    out mgr0_rvalid : i1,
// CHECK-SAME:    out sub0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    out sub0_rready : i1)

// CHECK-LABEL: hw.module @Cuts(
hw.module @Cuts(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %mgr.axi_aw, {{.*}} = hw.instance "mgr" @Manager(
  // CHECK-SAME: axi_awready: %cut0.mgr0_awready: i1
  %m = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !port)

  // Cuts of the same shape share the module, and count up their instances. Two
  // back to back wire straight to each other, with no bridge left between.
  // CHECK: %cut0.mgr0_awready, {{.*}} = hw.instance "cut0" @axi_cut_a32_d64_i4(
  // CHECK-SAME: clk_i: %clk: !seq.clock, rst_ni: %rst_ni: i1
  // CHECK-SAME: mgr0_aw: %mgr.axi_aw:
  // CHECK-SAME: sub0_awready: %cut1.mgr0_awready: i1
  %first = axi4.cut %clk, %rst_ni, %m : !port

  // CHECK: %cut1.mgr0_awready, {{.*}} = hw.instance "cut1" @axi_cut_a32_d64_i4(
  // CHECK-SAME: mgr0_aw: %cut0.sub0_aw:
  // CHECK-SAME: sub0_awready: %sub.axi_awready: i1
  %second = axi4.cut %clk, %rst_ni, %first : !port

  // CHECK: hw.instance "sub" @Subordinate(
  // CHECK-SAME: axi_aw: %cut1.sub0_aw:
  hw.instance "sub" @Subordinate(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %second: !port) -> ()
}
