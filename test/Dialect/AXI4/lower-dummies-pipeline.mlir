// RUN: circt-opt %s --lower-axi4-dummies-to-axi --verify-axi4-networks \
// RUN:   --lower-axi4-to-hw=pulp-mapping=true | FileCheck %s --implicit-check-not=axi4.

// A structural description reaches RTL through the whole pipeline: two managers
// of different data widths reach a memory and a peripheral through a crossbar,
// so the network carries a converter of every kind.

// Each endpoint keeps its declared widths at the boundary, with its own ID
// width, and its name.
// CHECK-LABEL: hw.module @System(
// CHECK-SAME:    in %core_aw : !hw.struct<id: i2, addr: i32,
// CHECK-SAME:    in %core_w : !hw.struct<data: i64, strb: i8,
// CHECK-SAME:    in %debug_w : !hw.struct<data: i32, strb: i4,
// CHECK-SAME:    in %mem_r : !hw.struct<id: i4, data: i64,
// CHECK-SAME:    in %periph_r : !hw.struct<id: i3, data: i32,
// CHECK-SAME:    out mem_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    out periph_w : !hw.struct<data: i32, strb: i4,

// The crossbar tags requests with the manager they came from, so the memory's
// wider IDs are converted onto, and the peripheral's narrower data width is
// converted down. The narrower manager is widened before the crossbar routes.
// CHECK: hw.instance "dw_converter0" @axi_dw_converter_a32_d32to64_i2
// CHECK: hw.instance "xbar0" @axi_xbar_2u2d_a32_d64_i2_o3
// CHECK: hw.instance "iw_converter0" @axi_iw_converter_a32_d64_i3to4
// CHECK: hw.instance "dw_converter1" @axi_dw_converter_a32_d64to32_i3

hw.module @System(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 16, outstanding_reads = 16
  %periph_access = axi4.dummies.ext_subordinate "periph" %clk, %rst_ni, %xbar window <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 4>>> addr_width = 32, data_width = 32, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %core_access -> %mem_access with <<incr, len = 16>>
  axi4.dummies.accesses %core_access -> %periph_access with <<incr, len = 2>>
  axi4.dummies.accesses %debug_access -> %mem_access with <<incr, len = 8>>
}
