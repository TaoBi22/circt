// RUN: circt-opt %s --lower-axi4-dummies-to-axi --optimize-axi4-networks | FileCheck %s
// RUN: circt-opt %s --lower-axi4-dummies-to-axi --optimize-axi4-networks \
// RUN:   --lower-axi4-to-hw=pulp-mapping=true | FileCheck %s --check-prefix=PULP

// An endpoint's PULP atop markers go on the port it becomes, where later
// rewrites of the network leave them alone
// CHECK-LABEL: hw.module @Atomics(
// CHECK-SAME:    in %core : !axi4.port<{{.*}}> {pulp.atops}, in %dma :
// CHECK-SAME:    concurrent_reads_per_id = 1>, out mem :
// CHECK-SAME:    > {pulp.atops}, out periph :
// CHECK-SAME:    > {pulp.atop_filter})

// The PULP lowering carries the core's atomics to the memory, and filters them
// out in front of the peripheral
// PULP-LABEL: hw.module @Atomics(
// PULP-SAME:    in %core_aw_atop : i6
// PULP-SAME:    out mem_aw_atop : i6
// PULP-DAG:     hw.instance "xbar0" @axi_xbar_2u2d_{{.*}}_atop(
// PULP-DAG:     hw.instance "atop_filter0" @axi_atop_filter_
hw.module @Atomics(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1 {pulp.atops}
  %dma, %dma_access = axi4.dummies.ext_manager "dma" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %dma addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 8, outstanding_read_ids = 8, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1 {pulp.atops}
  %periph_access = axi4.dummies.ext_subordinate "periph" %clk, %rst_ni, %xbar windows <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 1>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 8, outstanding_read_ids = 8, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1 {pulp.atop_filter}
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %core_access -> %periph_access with <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 1>>>
  axi4.dummies.accesses %dma_access -> %periph_access with <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 1>>>
}
