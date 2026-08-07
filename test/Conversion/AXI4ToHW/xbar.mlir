// RUN: circt-opt %s --lower-axi4-to-hw --split-input-file | FileCheck %s --implicit-check-not=axi4.

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfffffff, burst_specs = <<incr, len = 16>>>, <base = 0x10000000, last = 0x10000fff, burst_specs = <<fixed, len = 4>>>, <base = 0x20000000, last = 0x20000fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow_mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfffffff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 2, outstanding_reads = 2>
!sub_a = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfffffff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 6, outstanding_reads = 6>
!sub_b = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x10000000, last = 0x10000fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub_c = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x20000000, last = 0x20000fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !mgr)
hw.module.extern @SubordinateA(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !sub_a)
hw.module.extern @SubordinateB(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !sub_b)
hw.module.extern @ManagerAndSubordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi_sub : !sub_c, out axi_mgr : !narrow_mgr)

// One external module per crossbar, with a port group per endpoint named
// based on port index
// CHECK-LABEL: hw.module.extern @axi_xbar_2u3d_a32_d64_i4_o5(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %mgr0_rready : i1,
// CHECK-SAME:    in %mgr1_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %mgr1_rready : i1,
// CHECK-SAME:    in %sub0_awready : i1,
// CHECK-SAME:    in %sub2_rvalid : i1,
// CHECK-SAME:    out mgr0_awready : i1,
// CHECK-SAME:    out mgr1_rvalid : i1,
// CHECK-SAME:    out sub0_aw : !hw.struct<id: i5, addr: i32,
// CHECK-SAME:    out sub2_rready : i1)

// Check endpoint wires to the group of its own index, in both directions
// CHECK-LABEL: hw.module @ManyEndpoints(
hw.module @ManyEndpoints(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %mgr.axi_aw, {{.*}} = hw.instance "mgr" @Manager(
  // CHECK-SAME: axi_awready: %xbar0.mgr0_awready: i1
  %m = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !mgr)

  // An endpoint can sit on both sides of the same crossbar
  // CHECK: %both.axi_sub_awready, {{.*}} = hw.instance "both" @ManagerAndSubordinate(
  // CHECK-SAME: axi_sub_aw: %xbar0.sub2_aw:
  // CHECK-SAME: axi_mgr_awready: %xbar0.mgr1_awready: i1
  %b = hw.instance "both" @ManagerAndSubordinate(
      clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi_sub: %xbar#2: !sub_c)
    -> (axi_mgr: !narrow_mgr)

  // CHECK: %xbar0.mgr0_awready, {{.*}} = hw.instance "xbar0" @axi_xbar_2u3d_a32_d64_i4_o5(
  // CHECK-SAME: clk_i: %clk: !seq.clock, rst_ni: %rst_ni: i1
  // CHECK-SAME: mgr0_aw: %mgr.axi_aw:
  // CHECK-SAME: mgr1_aw: %both.axi_mgr_aw:
  // CHECK-SAME: sub0_awready: %sub_a.axi_awready: i1
  // CHECK-SAME: sub1_awready: %sub_b.axi_awready: i1
  // CHECK-SAME: sub2_awready: %both.axi_sub_awready: i1
  %xbar:3 = axi4.xbar %clk, %rst_ni mgrs %m, %b
    : (!mgr, !narrow_mgr) -> (!sub_a, !sub_b, !sub_c)

  // CHECK: hw.instance "sub_a" @SubordinateA(
  // CHECK-SAME: axi_aw: %xbar0.sub0_aw:
  hw.instance "sub_a" @SubordinateA(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %xbar#0: !sub_a) -> ()
  // CHECK: hw.instance "sub_b" @SubordinateB(
  // CHECK-SAME: axi_aw: %xbar0.sub1_aw:
  hw.instance "sub_b" @SubordinateB(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %xbar#1: !sub_b) -> ()
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !mgr)
hw.module.extern @Subordinate(in %axi : !sub)

// A single manager needs no extra ID bits, but still gets a crossbar
// CHECK-LABEL: hw.module.extern @axi_xbar_1u1d_a32_d64_i4_o5(
// CHECK-LABEL: hw.module @SingleManager(
// CHECK:         hw.instance "xbar0" @axi_xbar_1u1d_a32_d64_i4_o5(
hw.module @SingleManager(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  %s = axi4.xbar %clk, %rst_ni mgrs %m : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!wide_sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 6, read_id_width = 6, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
// The same widths, but a different address map
!high_mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x8000, last = 0x8fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!high_sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x8000, last = 0x8fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !mgr)
hw.module.extern @Subordinate(in %axi : !sub)
hw.module.extern @WideSubordinate(in %axi : !wide_sub)
hw.module.extern @HighManager(out axi : !high_mgr)
hw.module.extern @HighSubordinate(in %axi : !high_sub)

// Only crossbars whose ports match exactly share a module. Differing widths
// give a different name, and anything else the name does not describe - the
// address map here - collides and takes a suffix.
// CHECK: hw.module.extern @axi_xbar_1u1d_a32_d64_i4_o5(
// CHECK: hw.module.extern @axi_xbar_1u1d_a32_d64_i4_o6(
// CHECK: hw.module.extern @axi_xbar_1u1d_a32_d64_i4_o5_0(
// CHECK-LABEL: hw.module @Sharing(
// CHECK:         hw.instance "xbar0" @axi_xbar_1u1d_a32_d64_i4_o5(
// CHECK:         hw.instance "xbar1" @axi_xbar_1u1d_a32_d64_i4_o5(
// CHECK:         hw.instance "xbar2" @axi_xbar_1u1d_a32_d64_i4_o6(
// CHECK:         hw.instance "xbar3" @axi_xbar_1u1d_a32_d64_i4_o5_0(
hw.module @Sharing(in %clk : !seq.clock, in %rst_ni : i1) {
  %m0 = hw.instance "mgr0" @Manager() -> (axi: !mgr)
  %s0 = axi4.xbar %clk, %rst_ni mgrs %m0 : (!mgr) -> (!sub)
  hw.instance "sub0" @Subordinate(axi: %s0: !sub) -> ()

  %m1 = hw.instance "mgr1" @Manager() -> (axi: !mgr)
  %s1 = axi4.xbar %clk, %rst_ni mgrs %m1 : (!mgr) -> (!sub)
  hw.instance "sub1" @Subordinate(axi: %s1: !sub) -> ()

  %m2 = hw.instance "mgr2" @Manager() -> (axi: !mgr)
  %s2 = axi4.xbar %clk, %rst_ni mgrs %m2 : (!mgr) -> (!wide_sub)
  hw.instance "sub2" @WideSubordinate(axi: %s2: !wide_sub) -> ()

  %m3 = hw.instance "mgr3" @HighManager() -> (axi: !high_mgr)
  %s3 = axi4.xbar %clk, %rst_ni mgrs %m3 : (!high_mgr) -> (!high_sub)
  hw.instance "sub3" @HighSubordinate(axi: %s3: !high_sub) -> ()
}
