// RUN: circt-opt %s --lower-axi4-to-hw --split-input-file | FileCheck %s --implicit-check-not=axi4.

!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !wide_ids)
hw.module.extern @Subordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !narrow_ids)

// The module name carries both ID widths, and its two faces carry the payload
// structs of their own side
// CHECK-LABEL: hw.module.extern @axi_iw_converter_a32_d64_i4to2(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %sub0_r : !hw.struct<id: i2, data: i64,
// CHECK-SAME:    out mgr0_r : !hw.struct<id: i4, data: i64,
// CHECK-SAME:    out sub0_aw : !hw.struct<id: i2, addr: i32,

// CHECK-LABEL: hw.module @Narrowing(
hw.module @Narrowing(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %mgr.axi_aw, {{.*}} = hw.instance "mgr" @Manager(
  // CHECK-SAME: axi_awready: %iw_converter0.mgr0_awready: i1
  %m = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !wide_ids)

  // CHECK: %iw_converter0.mgr0_awready, {{.*}} = hw.instance "iw_converter0" @axi_iw_converter_a32_d64_i4to2(
  // CHECK-SAME: clk_i: %clk: !seq.clock, rst_ni: %rst_ni: i1
  // CHECK-SAME: mgr0_aw: %mgr.axi_aw:
  // CHECK-SAME: sub0_awready: %sub.axi_awready: i1
  %iwc = axi4.id_width_converter %clk, %rst_ni, %m : (!wide_ids) -> !narrow_ids

  // CHECK: hw.instance "sub" @Subordinate(
  // CHECK-SAME: axi_aw: %iw_converter0.sub0_aw:
  hw.instance "sub" @Subordinate(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %iwc: !narrow_ids) -> ()
}

// -----

!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !narrow_ids)
hw.module.extern @Subordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !wide_ids)

// Widening is the same component with the widths the other way round
// CHECK-LABEL: hw.module.extern @axi_iw_converter_a32_d64_i2to4(
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i2, addr: i32,
// CHECK-SAME:    out sub0_aw : !hw.struct<id: i4, addr: i32,

// CHECK-LABEL: hw.module @Widening(
hw.module @Widening(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !narrow_ids)
  // CHECK: hw.instance "iw_converter0" @axi_iw_converter_a32_d64_i2to4(
  %iwc = axi4.id_width_converter %clk, %rst_ni, %m : (!narrow_ids) -> !wide_ids
  hw.instance "sub" @Subordinate(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %iwc: !wide_ids) -> ()
}
