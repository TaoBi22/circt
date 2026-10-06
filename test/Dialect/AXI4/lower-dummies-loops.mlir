// RUN: circt-opt %s --lower-axi4-dummies-to-axi --split-input-file | FileCheck %s
// RUN: circt-opt %s --lower-axi4-dummies-to-axi --optimize-axi4-networks --split-input-file | FileCheck %s
// RUN: circt-opt %s --lower-axi4-dummies-to-axi --verify-axi4-networks --lower-axi4-to-hw=pulp-mapping=true --split-input-file | FileCheck %s --check-prefix=PULP

// A quadrant crossbar reaches each cluster's crossbar through an ID remapper,
// and each cluster's crossbar reaches back up to it, so the network loops. A
// crossbar's downstream port carries the windows below it reached without
// coming back through a crossbar, so the quadrant sends only a cluster's own
// memory down to it, and a cluster sends everything else up.
// CHECK-LABEL: hw.module @QuadrantLoops(
// CHECK-SAME:    out soc_out : !axi4.port<{{.*}} write_id_width = 5, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1> {pulp.atops})

// Each upstream port is connected only to the downstream ports its accesses
// are routed through, so no cluster is connected back down to itself
// CHECK:       %[[Q:.+]]:3 = axi4.xbar %clk, %rst_ni mgrs %soc, %{{.+}}#1, %{{.+}}#1 {PULP_CONFIG_Connectivity = {{\[}}[false, true, true], [true, true, false], [true, false, true]]}
// CHECK-SAME:    -> (!axi4.port<{{.*}}>, !axi4.port<{{.*}} write_id_width = 5, {{.*}} windows = <<base = 0x10040000, last = 0x1005ffff, {{.*}}>>>>, concurrent_writes_per_id = 1, {{.*}}>, !axi4.port<{{.*}} write_id_width = 5, {{.*}} windows = <<base = 0x10000000, last = 0x1001ffff, {{.*}}>>>>, concurrent_writes_per_id = 1, {{.*}}>)

// The remappers compact the quadrant's IDs, so the IDs stop growing around the
// loop
// CHECK:       %[[DOWN0:.+]] = axi4.id_remap %clk, %rst_ni, %[[Q]]#2 max_unique_ids = 4 : (!axi4.port<{{.*}} write_id_width = 5, {{.*}}>) -> !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
// CHECK:       %[[DOWN1:.+]] = axi4.id_remap %clk, %rst_ni, %[[Q]]#1 max_unique_ids = 4

// A cluster sends the other cluster's memory and the rest of the SoC up
// CHECK:       %[[C0:.+]]:2 = axi4.xbar %clk, %rst_ni mgrs %core0, %[[DOWN0]] {PULP_CONFIG_Connectivity = {{\[}}[true, true], [true, false]]}
// CHECK-SAME:    -> (!axi4.port<{{.*}} write_id_width = 3, {{.*}} windows = <<base = 0x10000000, last = 0x1001ffff, {{.*}}>>>>, concurrent_writes_per_id = 1, {{.*}}>, !axi4.port<{{.*}} write_id_width = 3, {{.*}} windows = <<base = 0x0, last = 0xfffffff, {{.*}}>>>, <base = 0x10040000, last = 0x1005ffff, {{.*}}>>>, <base = 0x10080000, last = 0xffffffffffff, {{.*}}>>>>, concurrent_writes_per_id = 1, {{.*}}>)
// CHECK:       %[[C1:.+]]:2 = axi4.xbar %clk, %rst_ni mgrs %core1, %[[DOWN1]]
// CHECK:       hw.output %[[C0]]#0, %[[C1]]#0, %[[Q]]#0

// Lowering to PULP traces the atomics through the loops and ends
// PULP-LABEL: hw.module @QuadrantLoops(
// PULP:         hw.instance "xbar0" @axi_xbar_3u3d_a48_d64_i3_o5_usr0_atop(
// PULP:         hw.instance "id_remap0" @axi_id_remap_a48_d64_i5to2_u4_usr0_atop(
// PULP:         hw.instance "xbar1" @axi_xbar_2u2d_a48_d64_i2_o3_usr0_atop(
hw.module @QuadrantLoops(in %clk : !seq.clock, in %rst_ni : i1) {
  %soc, %soc_access = axi4.dummies.ext_manager "soc" %clk, %rst_ni addr_width = 48, data_width = 64, outstanding_writes = 8, outstanding_reads = 8 {pulp.atops}
  %core0, %core0_access = axi4.dummies.ext_manager "core0" %clk, %rst_ni addr_width = 48, data_width = 64, outstanding_writes = 4, outstanding_reads = 4 {pulp.atops}
  %core1, %core1_access = axi4.dummies.ext_manager "core1" %clk, %rst_ni addr_width = 48, data_width = 64, outstanding_writes = 4, outstanding_reads = 4 {pulp.atops}

  %q = axi4.dummies.xbar %clk, %rst_ni mgrs %soc, %c0, %c1 addr_width = 48, data_width = 64
  %down0 = axi4.dummies.id_remap %clk, %rst_ni, %q max_unique_ids = 4
  %down1 = axi4.dummies.id_remap %clk, %rst_ni, %q max_unique_ids = 4
  %c0 = axi4.dummies.xbar %clk, %rst_ni mgrs %core0, %down0 addr_width = 48, data_width = 64
  %c1 = axi4.dummies.xbar %clk, %rst_ni mgrs %core1, %down1 addr_width = 48, data_width = 64

  %tcdm0_access = axi4.dummies.ext_subordinate "tcdm0" %clk, %rst_ni, %c0 windows <<base = 0x10000000, last = 0x1001ffff, burst_specs = <<incr, len = 256>>>> addr_width = 48, data_width = 64, outstanding_writes = 8, outstanding_reads = 8 {pulp.atops}
  %tcdm1_access = axi4.dummies.ext_subordinate "tcdm1" %clk, %rst_ni, %c1 windows <<base = 0x10040000, last = 0x1005ffff, burst_specs = <<incr, len = 256>>>> addr_width = 48, data_width = 64, outstanding_writes = 8, outstanding_reads = 8 {pulp.atops}
  %soc_out_access = axi4.dummies.ext_subordinate "soc_out" %clk, %rst_ni, %q windows <<base = 0x0, last = 0xfffffff, burst_specs = <<incr, len = 256>>>, <base = 0x10080000, last = 0xffffffffffff, burst_specs = <<incr, len = 256>>>> addr_width = 48, data_width = 64, outstanding_writes = 32, outstanding_reads = 32 {pulp.atops}

  axi4.dummies.accesses %soc_access -> %tcdm0_access with <base = 0x10000000, last = 0x1001ffff, burst_specs = <<incr, len = 256>>>
  axi4.dummies.accesses %soc_access -> %tcdm1_access with <base = 0x10040000, last = 0x1005ffff, burst_specs = <<incr, len = 256>>>
  axi4.dummies.accesses %core0_access -> %tcdm0_access with <base = 0x10000000, last = 0x1001ffff, burst_specs = <<incr, len = 256>>>
  axi4.dummies.accesses %core0_access -> %tcdm1_access with <base = 0x10040000, last = 0x1005ffff, burst_specs = <<incr, len = 256>>>
  axi4.dummies.accesses %core0_access -> %soc_out_access with <base = 0x10080000, last = 0xffffffffffff, burst_specs = <<incr, len = 256>>>
  axi4.dummies.accesses %core1_access -> %tcdm0_access with <base = 0x10000000, last = 0x1001ffff, burst_specs = <<incr, len = 256>>>
  axi4.dummies.accesses %core1_access -> %tcdm1_access with <base = 0x10040000, last = 0x1005ffff, burst_specs = <<incr, len = 256>>>
  axi4.dummies.accesses %core1_access -> %soc_out_access with <base = 0x10080000, last = 0xffffffffffff, burst_specs = <<incr, len = 256>>>
}

// -----

// Two crossbars feed each other through a remapper and a cut each way. The
// outer remapper is asked to track more IDs than reach it, so it tracks the 8
// that do, which widens the IDs reaching the inner remapper and the crossbar's
// other manager
// CHECK-LABEL: hw.module @SocLoop(
// CHECK:         %[[WIDENED:.+]] = axi4.id_width_converter %clk, %rst_ni, %m_inter : {{.*}} -> !axi4.port<{{.*}} write_id_width = 3,
// CHECK:         %[[INTER:.+]]:2 = axi4.xbar %clk, %rst_ni mgrs %[[WIDENED]], %{{.+}} {
// CHECK-SAME:      -> (!axi4.port<{{.*}} write_id_width = 4, {{.*}}>, !axi4.port<{{.*}} write_id_width = 4,
// CHECK:         %[[UP:.+]] = axi4.id_remap %clk, %rst_ni, %[[INTER]]#1 max_unique_ids = 4 : (!axi4.port<{{.*}} write_id_width = 4, {{.*}}>) -> !axi4.port<{{.*}} write_id_width = 2, {{.*}} concurrent_writes_per_id = 1,
// CHECK:         %[[UP_CUT:.+]] = axi4.cut %clk, %rst_ni, %[[UP]]
// CHECK:         %[[WIDE:.+]]:2 = axi4.xbar %clk, %rst_ni mgrs %m_wide, %[[UP_CUT]] {
// CHECK-SAME:      -> (!axi4.port<{{.*}} write_id_width = 3, {{.*}}>, !axi4.port<{{.*}} write_id_width = 3,
// CHECK:         %[[DOWN:.+]] = axi4.id_remap %clk, %rst_ni, %[[WIDE]]#1 max_unique_ids = 8 : (!axi4.port<{{.*}} write_id_width = 3, {{.*}}>) -> !axi4.port<{{.*}} write_id_width = 3, {{.*}} concurrent_writes_per_id = 1,
// CHECK:         %[[DOWN_CUT:.+]] = axi4.cut %clk, %rst_ni, %[[DOWN]]
// CHECK:         hw.output %[[INTER]]#0, %[[WIDE]]#0
// PULP-LABEL: hw.module @SocLoop(
// PULP:         hw.instance "id_remap0" @axi_id_remap_a32_d64_i4to2_u4_usr0(
// PULP:         hw.instance "id_remap1" @axi_id_remap_a32_d64_i3to3_u8_usr0(
hw.module @SocLoop(in %clk : !seq.clock, in %rst_ni : i1) {
  %m_inter, %m_inter_access = axi4.dummies.ext_manager "m_inter" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %m_wide, %m_wide_access = axi4.dummies.ext_manager "m_wide" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4

  %inter = axi4.dummies.xbar %clk, %rst_ni mgrs %m_inter, %down_cut addr_width = 32, data_width = 64
  %up = axi4.dummies.id_remap %clk, %rst_ni, %inter max_unique_ids = 4
  %up_cut = axi4.dummies.cut %clk, %rst_ni, %up
  %wide = axi4.dummies.xbar %clk, %rst_ni mgrs %m_wide, %up_cut addr_width = 32, data_width = 64
  %down = axi4.dummies.id_remap %clk, %rst_ni, %wide max_unique_ids = 16
  %down_cut = axi4.dummies.cut %clk, %rst_ni, %down

  %s_inter_access = axi4.dummies.ext_subordinate "s_inter" %clk, %rst_ni, %inter windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 16, outstanding_reads = 16
  %s_wide_access = axi4.dummies.ext_subordinate "s_wide" %clk, %rst_ni, %wide windows <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8

  axi4.dummies.accesses %m_inter_access -> %s_inter_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %m_inter_access -> %s_wide_access with <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %m_wide_access -> %s_inter_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %m_wide_access -> %s_wide_access with <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>
}
