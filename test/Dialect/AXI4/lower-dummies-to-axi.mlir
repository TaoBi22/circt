// RUN: circt-opt %s --lower-axi4-dummies-to-axi --split-input-file | FileCheck %s --implicit-check-not=axi4.dummies
// RUN: circt-opt %s --lower-axi4-dummies-to-axi=user-width=4 --split-input-file | FileCheck %s --check-prefix=USER

// A module with no dummies ops is left alone
// CHECK-LABEL: hw.module @NoDummies(in %clk : !seq.clock, in %rst_ni : i1)
hw.module @NoDummies(in %clk : !seq.clock, in %rst_ni : i1) {
}

// -----

// The endpoints become ports of the module the network is described in, and the
// manager's windows come from the accesses it declares
// CHECK-LABEL: hw.module @PointToPoint(
// CHECK-SAME:    in %clk : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %[[MGR:.+]] : !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>,
// CHECK-SAME:    out subordinate : !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>)
hw.module @PointToPoint(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  // CHECK: hw.output %[[MGR]]
}

// -----

// The endpoint names name the ports
// CHECK-LABEL: hw.module @Named(
// CHECK-SAME:    in %core : !axi4.port<{{.*}}>, out mem : !axi4.port<{{.*}}>)
hw.module @Named(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

// A manager declares the bursts it reaches a subordinate with, which may be
// narrower than the subordinate supports
// CHECK-LABEL: hw.module @NarrowerBursts(
// CHECK-SAME:    windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>
hw.module @NarrowerBursts(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 1, outstanding_reads = 1
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 16>, <incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 1, outstanding_reads = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>
}

// -----

// A subordinate port fronting several windows is reached by an access to each,
// so a manager can declare different bursts in each of them
// CHECK-LABEL: hw.module @PerWindowAccesses(
// CHECK-SAME:    in %core : !axi4.port<{{.*}} windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>, <base = 0x2000, last = 0x2fff, burst_specs = <<fixed, len = 4>>>>
hw.module @PerWindowAccesses(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>, <base = 0x2000, last = 0x2fff, burst_specs = <<fixed, len = 4>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x2000, last = 0x2fff, burst_specs = <<fixed, len = 4>>>
}

// -----

// A manager can declare an access to part of a subordinate's window
// CHECK-LABEL: hw.module @NarrowedWindow(
// CHECK-SAME:    in %core : !axi4.port<{{.*}} windows = <<base = 0x0, last = 0x7ff, burst_specs = <<incr, len = 16>>>>
hw.module @NarrowedWindow(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0x7ff, burst_specs = <<incr, len = 16>>>
}

// -----

// Overlapping accesses reach the union of the bursts they declare
// CHECK-LABEL: hw.module @OverlappingAccesses(
// CHECK-SAME:    in %core : !axi4.port<{{.*}} windows = <<base = 0x0, last = 0x7ff, burst_specs = <<fixed, len = 4>, <incr, len = 16>>>, <base = 0x800, last = 0xfff, burst_specs = <<fixed, len = 4>>>>
hw.module @OverlappingAccesses(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>, <incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0x7ff, burst_specs = <<incr, len = 16>>>
}

// -----

// Two managers reaching two subordinates through a crossbar. Each manager's
// windows are those of the subordinates it declares accesses to, and the
// crossbar widens the IDs to tag which manager a request came from. Only core
// accesses periph, so debug is only connected to mem.
// CHECK-LABEL: hw.module @Crossbar(
// CHECK-SAME:    in %core : !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>, <base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
// CHECK-SAME:    in %debug : !axi4.port<{{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
// CHECK-SAME:    out mem : !axi4.port<{{.*}} write_id_width = 3, read_id_width = 3, {{.*}} windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
// CHECK-SAME:    out periph : !axi4.port<{{.*}} write_id_width = 3, read_id_width = 3, {{.*}} windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
// USER-LABEL: hw.module @Crossbar(
// USER-SAME:    in %core : !axi4.port<{{[^>]*}} user_width = 4,
// USER-SAME:    in %debug : !axi4.port<{{[^>]*}} user_width = 4,
// USER-SAME:    out mem : !axi4.port<{{[^>]*}} user_width = 4,
// USER-SAME:    out periph : !axi4.port<{{[^>]*}} user_width = 4,
hw.module @Crossbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[XBAR:.+]]:2 = axi4.xbar %clk, %rst_ni mgrs %core, %debug {PULP_CONFIG_Connectivity = {{\[}}[true, true], [false, true]]}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  %periph_access = axi4.dummies.ext_subordinate "periph" %clk, %rst_ni, %xbar windows <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %core_access -> %periph_access with <base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>
  axi4.dummies.accesses %debug_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  // The crossbar's results follow its use list, so they run backwards here
  // CHECK: hw.output %[[XBAR]]#1, %[[XBAR]]#0
}

// -----

// A crossbar can reach a subordinate through another crossbar, which routes
// what the managers above can issue to it
// CHECK-LABEL: hw.module @ChainedCrossbars(
// CHECK-SAME:    out mem : !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
hw.module @ChainedCrossbars(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[TOP:.+]] = axi4.xbar %clk, %rst_ni mgrs %core
  %top = axi4.dummies.xbar %clk, %rst_ni mgrs %core addr_width = 32, data_width = 64
  // CHECK: %[[BOTTOM:.+]] = axi4.xbar %clk, %rst_ni mgrs %[[TOP]]
  %bottom = axi4.dummies.xbar %clk, %rst_ni mgrs %top addr_width = 32, data_width = 64
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %bottom windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  // CHECK: hw.output %[[BOTTOM]]
}

// -----

// A crossbar reached through another is connected by the accesses made through
// it
// CHECK-LABEL: hw.module @ChainedConnectivity(
hw.module @ChainedConnectivity(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %dma, %dma_access = axi4.dummies.ext_manager "dma" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[TOP:.+]]:2 = axi4.xbar %clk, %rst_ni mgrs %core, %debug {PULP_CONFIG_Connectivity = {{\[}}[true, true], [true, false]]} : ({{.*}}) -> (!axi4.port<{{[^>]*}} windows = <<base = 0x2000, {{.*}}>, !axi4.port<{{[^>]*}} windows = <<base = 0x0, last = 0xfff, {{[^>]*}}>>>>,
  %top = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  // CHECK: axi4.xbar %clk, %rst_ni mgrs %[[TOP]]#1, %{{.+}} {PULP_CONFIG_Connectivity = {{\[}}[false, true], [true, false]]} : ({{.*}}) -> (!axi4.port<{{[^>]*}} windows = <<base = 0x1000, {{.*}}>, !axi4.port<{{[^>]*}} windows = <<base = 0x0, last = 0xfff,
  %bottom = axi4.dummies.xbar %clk, %rst_ni mgrs %top, %dma addr_width = 32, data_width = 64
  %periph_access = axi4.dummies.ext_subordinate "periph" %clk, %rst_ni, %top windows <<base = 0x2000, last = 0x2fff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %bottom windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  %rom_access = axi4.dummies.ext_subordinate "rom" %clk, %rst_ni, %bottom windows <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %core_access -> %periph_access with <base = 0x2000, last = 0x2fff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %debug_access -> %periph_access with <base = 0x2000, last = 0x2fff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %dma_access -> %rom_access with <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>
}

// -----

// A crossbar's own Connectivity config is kept rather than derived
// CHECK-LABEL: hw.module @ConnectivityConfig(
hw.module @ConnectivityConfig(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: axi4.xbar %clk, %rst_ni mgrs %core, %debug {PULP_CONFIG_Connectivity = "'1"}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64 {PULP_CONFIG_Connectivity = "'1"}
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  %periph_access = axi4.dummies.ext_subordinate "periph" %clk, %rst_ni, %xbar windows <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %core_access -> %periph_access with <base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>
  axi4.dummies.accesses %debug_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

// An endpoint's ID width is log2 of the requests it can hold, so a converter
// bridges a manager and subordinate that disagree
// CHECK-LABEL: hw.module @NarrowerSubordinateIds(
// CHECK-SAME:    in %manager : !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
// CHECK-SAME:    out subordinate : !axi4.port<{{.*}} write_id_width = 1, read_id_width = 1, {{.*}} concurrent_writes_per_id = 2, concurrent_reads_per_id = 2>
hw.module @NarrowerSubordinateIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[CONV:.+]] = axi4.id_width_converter %clk, %rst_ni, %manager : (!axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>) -> !axi4.port<{{.*}} write_id_width = 1, read_id_width = 1, {{.*}} concurrent_writes_per_id = 2, concurrent_reads_per_id = 2>
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 2, outstanding_reads = 2
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  // CHECK: hw.output %[[CONV]]
}

// -----

// A crossbar's upstream ports must agree on their ID widths, so the narrower
// manager is widened onto the wider one
// CHECK-LABEL: hw.module @UnequalManagerIds(
hw.module @UnequalManagerIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 2, outstanding_reads = 2
  // CHECK: %[[WIDENED:.+]] = axi4.id_width_converter %clk, %rst_ni, %debug : (!axi4.port<{{.*}} write_id_width = 1, read_id_width = 1, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>) -> !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
  // CHECK: axi4.xbar %clk, %rst_ni mgrs %core, %[[WIDENED]]
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  %sub_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %core_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %debug_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

// A crossbar's PULP config is kept on the crossbar it lowers to, but not put on
// the converters inserted around it
// CHECK-LABEL: hw.module @XbarPulpConfig(
hw.module @XbarPulpConfig(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 2, outstanding_reads = 2
  // CHECK: axi4.id_width_converter %clk, %rst_ni, %debug :
  // CHECK: axi4.xbar %clk, %rst_ni mgrs %core, %{{.+}} {PULP_CONFIG_LatencyMode = "axi_pkg::NO_LATENCY"} :
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64 {PULP_CONFIG_LatencyMode = "axi_pkg::NO_LATENCY", other = 1 : i32}
  %sub_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %core_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %debug_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

// A crossbar widens IDs to tag which manager a request came from, so reaching a
// subordinate that tags with fewer bits narrows them again
// CHECK-LABEL: hw.module @NarrowSubordinateBelowXbar(
// CHECK-SAME:    out mem : !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} concurrent_writes_per_id = 2, concurrent_reads_per_id = 2>
hw.module @NarrowSubordinateBelowXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[XBAR:.+]] = axi4.xbar {{.*}} -> !axi4.port<{{.*}} write_id_width = 3, read_id_width = 3, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  // CHECK: %[[CONV:.+]] = axi4.id_width_converter %clk, %rst_ni, %[[XBAR]] : (!axi4.port<{{.*}} write_id_width = 3, read_id_width = 3, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>) -> !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} concurrent_writes_per_id = 2, concurrent_reads_per_id = 2>
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %debug_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  // CHECK: hw.output %[[CONV]]
}

// -----

// A converter bridges a manager and subordinate that disagree on their data
// width, and the same bursts count more of the narrower beats. A port carries
// the requests reaching it, so the subordinate's port holds the manager's 3
// rather than the 4 it declares it can serve.
// CHECK-LABEL: hw.module @NarrowerSubordinateData(
// CHECK-SAME:    in %manager : !axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 8>>>>, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
// CHECK-SAME:    out subordinate : !axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 16>>{{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
hw.module @NarrowerSubordinateData(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 3, outstanding_reads = 3
  // CHECK: %[[CONV:.+]] = axi4.data_width_converter %clk, %rst_ni, %manager : (!axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 8>>{{.*}}) -> !axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 16>>{{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 8>>>
  // CHECK: hw.output %[[CONV]]
}

// -----

// A crossbar carries one data width, so a narrower manager is widened onto it
// before it routes, and the subordinate's IDs are narrowed after
// CHECK-LABEL: hw.module @NarrowerManagerData(
// USER-LABEL: hw.module @NarrowerManagerData(
// USER: axi4.data_width_converter {{.*}} -> !axi4.port<{{[^>]*}} user_width = 4,
// USER: axi4.xbar {{.*}} -> !axi4.port<{{[^>]*}} user_width = 4,
// USER: axi4.id_width_converter {{.*}} -> !axi4.port<{{[^>]*}} user_width = 4,
hw.module @NarrowerManagerData(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[WIDENED:.+]] = axi4.data_width_converter %clk, %rst_ni, %manager : (!axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 16>>{{.*}}) -> !axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 8>>
  // CHECK: %[[XBAR:.+]] = axi4.xbar %clk, %rst_ni mgrs %[[WIDENED]]
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64
  // CHECK: %[[CONV:.+]] = axi4.id_width_converter %clk, %rst_ni, %[[XBAR]] : (!axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}}) -> !axi4.port<{{.*}} write_id_width = 3, read_id_width = 3,
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  // CHECK: hw.output %[[CONV]]
}

// -----

// A crossbar can only ask a narrower subordinate for whole beats of its own
// width, so half of the 16 beats of 32 bits it supports are out of reach
// CHECK-LABEL: hw.module @NarrowerSubordinateDataBelowXbar(
// CHECK-SAME:    out mem : !axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 16>>
hw.module @NarrowerSubordinateDataBelowXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[XBAR:.+]] = axi4.xbar {{.*}} -> !axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 8>>
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64
  // CHECK: %[[CONV:.+]] = axi4.data_width_converter %clk, %rst_ni, %[[XBAR]]
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 8>>>
  // CHECK: hw.output %[[CONV]]
}

// -----

// The 256 beats of 64 bits the subordinate supports are 512 of the crossbar's
// 32, more than AXI4 permits, so the crossbar asks for the longest burst it can
// CHECK-LABEL: hw.module @ClampedBursts(
// CHECK-SAME:    out mem : !axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 128>>
hw.module @ClampedBursts(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[XBAR:.+]] = axi4.xbar {{.*}} -> !axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 256>>
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 32
  // CHECK: axi4.data_width_converter %clk, %rst_ni, %[[XBAR]]
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 256>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 256>>>
}

// -----

// Every data width on the way down is converted onto the next, so a burst is
// counted in beats of each of them in turn
// CHECK-LABEL: hw.module @MixedWidthCrossbars(
// CHECK-SAME:    in %core : !axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 4>>
// CHECK-SAME:    out mem : !axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 16>>
hw.module @MixedWidthCrossbars(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[TOP:.+]] = axi4.xbar %clk, %rst_ni mgrs %core {{.*}} -> !axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 8>>
  %top = axi4.dummies.xbar %clk, %rst_ni mgrs %core addr_width = 32, data_width = 64
  // CHECK: %[[NARROWED:.+]] = axi4.data_width_converter %clk, %rst_ni, %[[TOP]] : (!axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 8>>{{.*}}) -> !axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 16>>
  // CHECK: %[[BOTTOM:.+]] = axi4.xbar %clk, %rst_ni mgrs %[[NARROWED]] {{.*}} -> !axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 16>>
  %bottom = axi4.dummies.xbar %clk, %rst_ni mgrs %top addr_width = 32, data_width = 32
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %bottom windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>
  // CHECK: hw.output %[[BOTTOM]]
}

// -----

// A cut lowers to one on the same connection, keeping its PULP config
// CHECK-LABEL: hw.module @Cuts(
hw.module @Cuts(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[ABOVE:.+]] = axi4.cut %clk, %rst_ni, %core {PULP_CONFIG_Bypass = "1'b1"}
  %above = axi4.dummies.cut %clk, %rst_ni, %core {PULP_CONFIG_Bypass = "1'b1", other = 1 : i32}
  // CHECK: %[[XBAR:.+]] = axi4.xbar %clk, %rst_ni mgrs %[[ABOVE]]
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %above addr_width = 32, data_width = 64
  // CHECK: %[[FIRST:.+]] = axi4.cut %clk, %rst_ni, %[[XBAR]]
  %first = axi4.dummies.cut %clk, %rst_ni, %xbar
  // CHECK: %[[SECOND:.+]] = axi4.cut %clk, %rst_ni, %[[FIRST]]
  %second = axi4.dummies.cut %clk, %rst_ni, %first
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %second windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  // CHECK: hw.output %[[SECOND]]
}

// -----

// A converter goes in front of the endpoint it adapts the connection to, so
// after the cuts on the way
// CHECK-LABEL: hw.module @CutBeforeConverter(
hw.module @CutBeforeConverter(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[CUT:.+]] = axi4.cut %clk, %rst_ni, %manager : !axi4.port<{{.*}} data_width = 64,
  %cut = axi4.dummies.cut %clk, %rst_ni, %mgr
  // CHECK: %[[CONV:.+]] = axi4.data_width_converter %clk, %rst_ni, %[[CUT]] : {{.*}} -> !axi4.port<{{.*}} data_width = 32,
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %cut windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 8>>>
  // CHECK: hw.output %[[CONV]]
}

// -----

// A remapper below a crossbar compacts the IDs the crossbar widens, and holds
// as many requests as it tracks IDs. It keeps its PULP config.
// CHECK-LABEL: hw.module @RemapBelowXbar(
// CHECK-SAME:    out mem : !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>)
hw.module @RemapBelowXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[XBAR:.+]] = axi4.xbar %clk, %rst_ni mgrs %core, %debug
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  // CHECK: %[[REMAP:.+]] = axi4.id_remap %clk, %rst_ni, %[[XBAR]] max_unique_ids = 4 {PULP_CONFIG_AxiMaxTxnsPerId = 2 : i32} : (!axi4.port<{{.*}} write_id_width = 3, read_id_width = 3, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>) -> !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
  %remap = axi4.dummies.id_remap %clk, %rst_ni, %xbar max_unique_ids = 4 {PULP_CONFIG_AxiMaxTxnsPerId = 2 : i32, other = 1 : i32}
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %remap windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %debug_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  // CHECK: hw.output %[[REMAP]]
}

// -----

// A crossbar's upstream ports share ID widths, so it widens a remapper's
// narrower IDs onto the other manager's
// CHECK-LABEL: hw.module @RemapIntoXbar(
hw.module @RemapIntoXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %dma, %dma_access = axi4.dummies.ext_manager "dma" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 16, outstanding_reads = 16
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[REMAP:.+]] = axi4.id_remap %clk, %rst_ni, %dma max_unique_ids = 2 : {{.*}} -> !axi4.port<{{.*}} write_id_width = 1, read_id_width = 1, {{.*}} concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
  %remap = axi4.dummies.id_remap %clk, %rst_ni, %dma max_unique_ids = 2
  // CHECK: %[[WIDENED:.+]] = axi4.id_width_converter %clk, %rst_ni, %[[REMAP]] : {{.*}} -> !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2,
  // CHECK: axi4.xbar %clk, %rst_ni mgrs %[[WIDENED]], %core
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %remap, %core addr_width = 32, data_width = 64
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %dma_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

// CHECK-LABEL: hw.module @CutsAroundRemap(
hw.module @CutsAroundRemap(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  // CHECK: %[[ABOVE:.+]] = axi4.cut %clk, %rst_ni, %manager
  %above = axi4.dummies.cut %clk, %rst_ni, %mgr
  // CHECK: %[[REMAP:.+]] = axi4.id_remap %clk, %rst_ni, %[[ABOVE]] max_unique_ids = 4
  %remap = axi4.dummies.id_remap %clk, %rst_ni, %above max_unique_ids = 4
  // CHECK: %[[BELOW:.+]] = axi4.cut %clk, %rst_ni, %[[REMAP]]
  %below = axi4.dummies.cut %clk, %rst_ni, %remap
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %below windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  // CHECK: hw.output %[[BELOW]]
}

// -----

// The ID widths are inferred, so a remapper is never asked to track more IDs
// than its upstream port can carry
// CHECK-LABEL: hw.module @RemapPastUpstreamIds(
hw.module @RemapPastUpstreamIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 2, outstanding_reads = 2
  // CHECK: axi4.id_remap %clk, %rst_ni, %manager max_unique_ids = 2 : {{.*}} -> !axi4.port<{{.*}} write_id_width = 1, read_id_width = 1,
  %remap = axi4.dummies.id_remap %clk, %rst_ni, %mgr max_unique_ids = 8
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %remap windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_writes = 2, outstanding_reads = 2
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

// Two subordinates serve HBM, one directly below P and one below W. A port
// carries the windows of the subordinates the accesses through it target, and
// no access from P targets W's HBM, so P sends only the scratchpad towards W.
// CHECK-LABEL: hw.module @TwoWaysToHbm(
// CHECK:         %[[P:.+]]:2 = axi4.xbar %clk, %rst_ni mgrs %quad :
// CHECK-SAME:      -> (!axi4.port<{{[^>]*}} windows = <<base = 0x80000000, last = 0xffffffff, {{[^>]*}}>>>>, {{[^>]*}}>, !axi4.port<{{[^>]*}} windows = <<base = 0x71000000, last = 0x710fffff, {{[^>]*}}>>>>,
// CHECK:         %[[TO_W:.+]] = axi4.cut %clk, %rst_ni, %[[P]]#1
// CHECK:         axi4.xbar %clk, %rst_ni mgrs %m_w, %[[TO_W]] {PULP_CONFIG_Connectivity = {{\[}}[true, true], [true, false]]}
hw.module @TwoWaysToHbm(in %clk : !seq.clock, in %rst_ni : i1) {
  %quad, %quad_access = axi4.dummies.ext_manager "quad" %clk, %rst_ni addr_width = 48, data_width = 64, outstanding_writes = 16, outstanding_reads = 16
  %m_w, %m_w_access = axi4.dummies.ext_manager "m_w" %clk, %rst_ni addr_width = 48, data_width = 64, outstanding_writes = 16, outstanding_reads = 16

  %p = axi4.dummies.xbar %clk, %rst_ni mgrs %quad addr_width = 48, data_width = 64
  %p_to_w = axi4.dummies.cut %clk, %rst_ni, %p
  %w = axi4.dummies.xbar %clk, %rst_ni mgrs %m_w, %p_to_w addr_width = 48, data_width = 64

  %hbm_p_access = axi4.dummies.ext_subordinate "hbm_p" %clk, %rst_ni, %p windows <<base = 0x80000000, last = 0xffffffff, burst_specs = <<incr, len = 256>>>> addr_width = 48, data_width = 64, outstanding_writes = 16, outstanding_reads = 16
  %hbm_w_access = axi4.dummies.ext_subordinate "hbm_w" %clk, %rst_ni, %w windows <<base = 0x80000000, last = 0xffffffff, burst_specs = <<incr, len = 256>>>> addr_width = 48, data_width = 64, outstanding_writes = 32, outstanding_reads = 32
  %spm_access = axi4.dummies.ext_subordinate "spm" %clk, %rst_ni, %w windows <<base = 0x71000000, last = 0x710fffff, burst_specs = <<incr, len = 256>>>> addr_width = 48, data_width = 64, outstanding_writes = 32, outstanding_reads = 32

  axi4.dummies.accesses %quad_access -> %hbm_p_access with <base = 0x80000000, last = 0xffffffff, burst_specs = <<incr, len = 256>>>
  axi4.dummies.accesses %quad_access -> %spm_access with <base = 0x71000000, last = 0x710fffff, burst_specs = <<incr, len = 256>>>
  axi4.dummies.accesses %m_w_access -> %hbm_w_access with <base = 0x80000000, last = 0xffffffff, burst_specs = <<incr, len = 256>>>
  axi4.dummies.accesses %m_w_access -> %spm_access with <base = 0x71000000, last = 0x710fffff, burst_specs = <<incr, len = 256>>>
}
