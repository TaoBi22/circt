// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s --implicit-check-not=axi4.

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// A demux carries every width through, so all of its faces explode into the
// same payload structs; the name counts the downstream ports it routes to
// CHECK-LABEL: hw.module.extern @axi_demux_2d_a32_d64_i4(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %mgr0_rready : i1,
// CHECK-SAME:    in %sub0_awready : i1,
// CHECK-SAME:    in %sub1_rvalid : i1,
// CHECK-SAME:    out mgr0_awready : i1,
// CHECK-SAME:    out mgr0_rvalid : i1,
// CHECK-SAME:    out sub0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    out sub1_rready : i1)

// CHECK-LABEL: hw.module @Demux(
// CHECK-SAME:    in %upstream_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %hi_rvalid : i1,
// CHECK-SAME:    out upstream_awready : i1,
// CHECK-SAME:    out hi_rready : i1)
hw.module @Demux(in %clk : !seq.clock, in %rst_ni : i1, in %upstream : !mgr,
                 out lo : !lo, out hi : !hi) {
  // CHECK: %demux0.mgr0_awready, {{.*}} = hw.instance "demux0" @axi_demux_2d_a32_d64_i4(
  // CHECK-SAME: clk_i: %clk: !seq.clock, rst_ni: %rst_ni: i1
  // CHECK-SAME: mgr0_aw: %upstream_aw:
  // CHECK-SAME: sub0_awready: %lo_awready: i1
  // CHECK-SAME: sub1_awready: %hi_awready: i1
  %a, %b = axi4.demux %clk, %rst_ni, %upstream : (!mgr) -> (!lo, !hi)

  // CHECK: hw.output %demux0.mgr0_awready,
  // CHECK-SAME: %demux0.sub0_aw,
  hw.output %a, %b : !lo, !hi
}
