// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s --implicit-check-not=axi4.

!burstty = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!beats = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>>, outstanding_writes = 16, outstanding_reads = 16>

// A splitter changes only the bursts a port carries, so both faces explode into
// the same payload structs
// CHECK-LABEL: hw.module.extern @axi_burst_splitter_a32_d64_i4(
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
// CHECK-LABEL: hw.module @BurstSplitter(
// CHECK-SAME:    in %upstream_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %downstream_rvalid : i1,
// CHECK-SAME:    out upstream_awready : i1,
// CHECK-SAME:    out downstream_rready : i1)
hw.module @BurstSplitter(in %clk : !seq.clock, in %rst_ni : i1,
                         in %upstream : !burstty, out downstream : !beats) {
  // CHECK: %burst_splitter0.mgr0_awready, {{.*}} = hw.instance "burst_splitter0" @axi_burst_splitter_a32_d64_i4(
  // CHECK-SAME: clk_i: %clk: !seq.clock, rst_ni: %rst_ni: i1
  // CHECK-SAME: mgr0_aw: %upstream_aw:
  // CHECK-SAME: sub0_awready: %downstream_awready: i1
  %split = axi4.burst_splitter %clk, %rst_ni, %upstream : (!burstty) -> !beats

  // CHECK: hw.output %burst_splitter0.mgr0_awready,
  // CHECK-SAME: %burst_splitter0.sub0_aw,
  hw.output %split : !beats
}
