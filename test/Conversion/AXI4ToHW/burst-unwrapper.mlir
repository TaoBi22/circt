// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s --implicit-check-not=axi4.

!wrapping = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!unwrapped = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>

// An unwrapper changes only the bursts a port carries, so both faces explode
// into the same payload structs
// CHECK-LABEL: hw.module.extern @axi_burst_unwrapper_a32_d64_i4(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %mgr0_rready : i1,
// CHECK-SAME:    in %sub0_awready : i1,
// CHECK-SAME:    in %sub0_rvalid : i1,
// CHECK-SAME:    out mgr0_awready : i1,
// CHECK-SAME:    out mgr0_rvalid : i1,
// CHECK-SAME:    out sub0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    out sub0_rready : i1)

// The enclosing module's own ports explode to mirror the instance's
// CHECK-LABEL: hw.module @BurstUnwrapper(
// CHECK-SAME:    in %upstream_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %downstream_rvalid : i1,
// CHECK-SAME:    out upstream_awready : i1,
// CHECK-SAME:    out downstream_rready : i1)
hw.module @BurstUnwrapper(in %clk : !seq.clock, in %rst_ni : i1,
                          in %upstream : !wrapping, out downstream : !unwrapped) {
  // CHECK: %burst_unwrapper0.mgr0_awready, {{.*}} = hw.instance "burst_unwrapper0" @axi_burst_unwrapper_a32_d64_i4(
  // CHECK-SAME: clk_i: %clk: !seq.clock, rst_ni: %rst_ni: i1
  // CHECK-SAME: mgr0_aw: %upstream_aw:
  // CHECK-SAME: sub0_awready: %downstream_awready: i1
  %unwrapped = axi4.burst_unwrapper %clk, %rst_ni, %upstream : (!wrapping) -> !unwrapped

  // CHECK: hw.output %burst_unwrapper0.mgr0_awready,
  // CHECK-SAME: %burst_unwrapper0.sub0_aw,
  hw.output %unwrapped : !unwrapped
}
