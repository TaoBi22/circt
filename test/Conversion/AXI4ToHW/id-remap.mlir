// RUN: circt-opt %s --lower-axi4-to-hw --split-input-file | FileCheck %s --implicit-check-not=axi4.

!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!fewer_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !wide_ids)
hw.module.extern @Subordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !narrow_ids)
hw.module.extern @SmallSubordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !fewer_ids)

// The module name carries both ID widths and the IDs tracked, and its two faces
// carry the payload structs of their own side
// CHECK-LABEL: hw.module.extern @axi_id_remap_a32_d64_i4to2_u4_usr0(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %sub0_r : !hw.struct<id: i2, data: i64,
// CHECK-SAME:    out mgr0_r : !hw.struct<id: i4, data: i64,
// CHECK-SAME:    out sub0_aw : !hw.struct<id: i2, addr: i32,

// Remappers tracking different numbers of IDs differ in behaviour, not ports
// CHECK-LABEL: hw.module.extern @axi_id_remap_a32_d64_i4to2_u2_usr0(

// CHECK-LABEL: hw.module @Remapping(
hw.module @Remapping(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %mgr.axi_aw, {{.*}} = hw.instance "mgr" @Manager(
  // CHECK-SAME: axi_awready: %id_remap0.mgr0_awready: i1
  %m = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !wide_ids)

  // CHECK: %id_remap0.mgr0_awready, {{.*}} = hw.instance "id_remap0" @axi_id_remap_a32_d64_i4to2_u4_usr0(
  // CHECK-SAME: clk_i: %clk: !seq.clock, rst_ni: %rst_ni: i1
  // CHECK-SAME: mgr0_aw: %mgr.axi_aw:
  // CHECK-SAME: sub0_awready: %sub.axi_awready: i1
  %remap = axi4.id_remap %clk, %rst_ni, %m max_unique_ids = 4, concurrent_per_id = 4 : (!wide_ids) -> !narrow_ids

  // CHECK: hw.instance "sub" @Subordinate(
  // CHECK-SAME: axi_aw: %id_remap0.sub0_aw:
  hw.instance "sub" @Subordinate(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %remap: !narrow_ids) -> ()

  %m2 = hw.instance "mgr2" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !wide_ids)
  // CHECK: hw.instance "id_remap1" @axi_id_remap_a32_d64_i4to2_u2_usr0(
  %small = axi4.id_remap %clk, %rst_ni, %m2 max_unique_ids = 2, concurrent_per_id = 4 : (!wide_ids) -> !fewer_ids
  hw.instance "small" @SmallSubordinate(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %small: !fewer_ids) -> ()
}
