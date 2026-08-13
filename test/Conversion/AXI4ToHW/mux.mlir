// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s --implicit-check-not=axi4.

!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>

// A mux tags its managers' transactions, so its downstream face carries the
// wider IDs the name ends in; the name counts the managers it arbitrates
// CHECK-LABEL: hw.module.extern @axi_mux_2u_a32_d64_i4_o5(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %mgr1_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %mgr1_rready : i1,
// CHECK-SAME:    in %sub0_awready : i1,
// CHECK-SAME:    in %sub0_rvalid : i1,
// CHECK-SAME:    out mgr0_awready : i1,
// CHECK-SAME:    out mgr1_rvalid : i1,
// CHECK-SAME:    out sub0_aw : !hw.struct<id: i5, addr: i32,
// CHECK-SAME:    out sub0_rready : i1)

// CHECK-LABEL: hw.module @Mux(
// CHECK-SAME:    in %a_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %downstream_rvalid : i1,
// CHECK-SAME:    out a_awready : i1,
// CHECK-SAME:    out downstream_rready : i1)
hw.module @Mux(in %clk : !seq.clock, in %rst_ni : i1, in %a : !lo, in %b : !hi,
               out downstream : !sub) {
  // CHECK: %mux0.mgr0_awready, {{.*}} = hw.instance "mux0" @axi_mux_2u_a32_d64_i4_o5(
  // CHECK-SAME: clk_i: %clk: !seq.clock, rst_ni: %rst_ni: i1
  // CHECK-SAME: mgr0_aw: %a_aw:
  // CHECK-SAME: mgr1_aw: %b_aw:
  // CHECK-SAME: sub0_awready: %downstream_awready: i1
  %downstream = axi4.mux %clk, %rst_ni, %a, %b : (!lo, !hi) -> !sub

  // CHECK: hw.output %mux0.mgr0_awready,
  // CHECK-SAME: %mux0.sub0_aw,
  hw.output %downstream : !sub
}
