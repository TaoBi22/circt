// RUN: circt-opt %s --lower-axi4-to-hw --split-input-file | FileCheck %s --implicit-check-not=axi4.

!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !wide)
hw.module.extern @Subordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !thin)

// The module name carries both data widths, and its two faces carry the payload
// structs of their own side
// CHECK-LABEL: hw.module.extern @axi_dw_converter_a32_d64to32_i4(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mgr0_w : !hw.struct<data: i64, strb: i8,
// CHECK-SAME:    in %sub0_r : !hw.struct<id: i4, data: i32,
// CHECK-SAME:    out mgr0_r : !hw.struct<id: i4, data: i64,
// CHECK-SAME:    out sub0_w : !hw.struct<data: i32, strb: i4,

// CHECK-LABEL: hw.module @Narrowing(
hw.module @Narrowing(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %mgr.axi_aw, {{.*}} = hw.instance "mgr" @Manager(
  // CHECK-SAME: axi_awready: %dw_converter0.mgr0_awready: i1
  %m = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !wide)

  // CHECK: %dw_converter0.mgr0_awready, {{.*}} = hw.instance "dw_converter0" @axi_dw_converter_a32_d64to32_i4(
  // CHECK-SAME: clk_i: %clk: !seq.clock, rst_ni: %rst_ni: i1
  // CHECK-SAME: mgr0_aw: %mgr.axi_aw:
  // CHECK-SAME: sub0_awready: %sub.axi_awready: i1
  %dwc = axi4.data_width_converter %clk, %rst_ni, %m : (!wide) -> !thin

  // CHECK: hw.instance "sub" @Subordinate(
  // CHECK-SAME: axi_aw: %dw_converter0.sub0_aw:
  hw.instance "sub" @Subordinate(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %dwc: !thin) -> ()
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !port)
hw.module.extern @Subordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !port)

// A converter that keeps the data width has the same ports as a cut, so the
// kind is part of what makes two components share a module
// CHECK: hw.module.extern @axi_cut_a32_d64_i4(
// CHECK: hw.module.extern @axi_dw_converter_a32_d64to64_i4(

// CHECK-LABEL: hw.module @SameShapeAsACut(
hw.module @SameShapeAsACut(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !port)
  // CHECK: hw.instance "cut0" @axi_cut_a32_d64_i4(
  %cut = axi4.cut %clk, %rst_ni, %m : !port
  // CHECK: hw.instance "dw_converter0" @axi_dw_converter_a32_d64to64_i4(
  %dwc = axi4.data_width_converter %clk, %rst_ni, %cut : (!port) -> !port
  hw.instance "sub" @Subordinate(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %dwc: !port) -> ()
}
