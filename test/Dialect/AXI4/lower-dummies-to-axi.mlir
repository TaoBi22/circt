// RUN: circt-opt %s --lower-axi4-dummies-to-axi --split-input-file | FileCheck %s --implicit-check-not=axi4.dummies

// A module with no dummies ops is left alone
// CHECK-LABEL: hw.module @NoDummies(in %clk : !seq.clock, in %rst_ni : i1)
hw.module @NoDummies(in %clk : !seq.clock, in %rst_ni : i1) {
}

// -----

// The endpoints become ports of the module the network is described in, and the
// manager's windows come from the accesses it declares
// CHECK-LABEL: hw.module @PointToPoint(
// CHECK-SAME:    in %clk : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %[[MGR:.+]] : !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>,
// CHECK-SAME:    out subordinate : !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>)
hw.module @PointToPoint(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
  // CHECK: hw.output %[[MGR]]
}

// -----

// The endpoint names name the ports
// CHECK-LABEL: hw.module @Named(
// CHECK-SAME:    in %core : !axi4.port<{{.*}}>, out mem : !axi4.port<{{.*}}>)
hw.module @Named(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
}

// -----

// A manager declares the bursts it reaches a subordinate with, which may be
// narrower than the subordinate supports
// CHECK-LABEL: hw.module @NarrowerBursts(
// CHECK-SAME:    windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>
hw.module @NarrowerBursts(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 1, outstanding_reads = 1
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 16>, <incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 1, outstanding_reads = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 4>>
}

// -----

// Two managers reaching two subordinates through a crossbar. Each manager's
// windows are those of the subordinates it declares accesses to, and the
// crossbar widens the IDs to tag which manager a request came from.
// CHECK-LABEL: hw.module @Crossbar(
// CHECK-SAME:    in %core : !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>, <base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
// CHECK-SAME:    in %debug : !axi4.port<{{.*}} windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
// CHECK-SAME:    out mem : !axi4.port<{{.*}} write_id_width = 3, read_id_width = 3, {{.*}} windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 8, outstanding_reads = 8>
// CHECK-SAME:    out periph : !axi4.port<{{.*}} write_id_width = 3, read_id_width = 3, {{.*}} windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>
hw.module @Crossbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[XBAR:.+]]:2 = axi4.xbar %clk, %rst_ni mgrs %core, %debug
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  %periph_access = axi4.dummies.ext_subordinate "periph" %clk, %rst_ni, %xbar window <base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %core_access -> %mem_access with <<incr, len = 16>>
  axi4.dummies.accesses %core_access -> %periph_access with <<fixed, len = 4>>
  axi4.dummies.accesses %debug_access -> %mem_access with <<incr, len = 16>>
  // The crossbar's results follow its use list, so they run backwards here
  // CHECK: hw.output %[[XBAR]]#1, %[[XBAR]]#0
}

// -----

// A crossbar can reach a subordinate through another crossbar, which routes
// what the managers above can issue to it
// CHECK-LABEL: hw.module @ChainedCrossbars(
// CHECK-SAME:    out mem : !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} outstanding_writes = 4, outstanding_reads = 4>
hw.module @ChainedCrossbars(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[TOP:.+]] = axi4.xbar %clk, %rst_ni mgrs %core
  %top = axi4.dummies.xbar %clk, %rst_ni mgrs %core addr_width = 32, data_width = 64
  // CHECK: %[[BOTTOM:.+]] = axi4.xbar %clk, %rst_ni mgrs %[[TOP]]
  %bottom = axi4.dummies.xbar %clk, %rst_ni mgrs %top addr_width = 32, data_width = 64
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %bottom window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %core_access -> %mem_access with <<incr, len = 16>>
  // CHECK: hw.output %[[BOTTOM]]
}

// -----

// An endpoint's ID width is log2 of the requests it can hold, so a converter
// bridges a manager and subordinate that disagree
// CHECK-LABEL: hw.module @NarrowerSubordinateIds(
// CHECK-SAME:    in %manager : !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} outstanding_writes = 4, outstanding_reads = 4>
// CHECK-SAME:    out subordinate : !axi4.port<{{.*}} write_id_width = 1, read_id_width = 1, {{.*}} outstanding_writes = 2, outstanding_reads = 2>
hw.module @NarrowerSubordinateIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[CONV:.+]] = axi4.id_width_converter %clk, %rst_ni, %manager : (!axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} outstanding_writes = 4, outstanding_reads = 4>) -> !axi4.port<{{.*}} write_id_width = 1, read_id_width = 1, {{.*}} outstanding_writes = 2, outstanding_reads = 2>
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 2, outstanding_reads = 2
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
  // CHECK: hw.output %[[CONV]]
}

// -----

// A crossbar's upstream ports must agree on their ID widths, so the narrower
// manager is widened onto the wider one
// CHECK-LABEL: hw.module @UnequalManagerIds(
hw.module @UnequalManagerIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 2, outstanding_reads = 2
  // CHECK: %[[WIDENED:.+]] = axi4.id_width_converter %clk, %rst_ni, %debug : (!axi4.port<{{.*}} write_id_width = 1, read_id_width = 1, {{.*}} outstanding_writes = 2, outstanding_reads = 2>) -> !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} outstanding_writes = 2, outstanding_reads = 2>
  // CHECK: axi4.xbar %clk, %rst_ni mgrs %core, %[[WIDENED]]
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  %sub_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %core_access -> %sub_access with <<incr, len = 16>>
  axi4.dummies.accesses %debug_access -> %sub_access with <<incr, len = 16>>
}

// -----

// A crossbar widens IDs to tag which manager a request came from, so reaching a
// subordinate that tags with fewer bits narrows them again
// CHECK-LABEL: hw.module @NarrowSubordinateBelowXbar(
// CHECK-SAME:    out mem : !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} outstanding_writes = 4, outstanding_reads = 4>
hw.module @NarrowSubordinateBelowXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager "debug" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[XBAR:.+]] = axi4.xbar {{.*}} -> !axi4.port<{{.*}} write_id_width = 3, read_id_width = 3, {{.*}} outstanding_writes = 8, outstanding_reads = 8>
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  // CHECK: %[[CONV:.+]] = axi4.id_width_converter %clk, %rst_ni, %[[XBAR]] : (!axi4.port<{{.*}} write_id_width = 3, read_id_width = 3, {{.*}} outstanding_writes = 8, outstanding_reads = 8>) -> !axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}} outstanding_writes = 4, outstanding_reads = 4>
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %core_access -> %mem_access with <<incr, len = 16>>
  axi4.dummies.accesses %debug_access -> %mem_access with <<incr, len = 16>>
  // CHECK: hw.output %[[CONV]]
}

// -----

// A converter bridges a manager and subordinate that disagree on their data
// width, and the same bursts count more of the narrower beats. The subordinate
// keeps the requests it declares it can hold, which the manager's ID width has
// the tags for.
// CHECK-LABEL: hw.module @NarrowerSubordinateData(
// CHECK-SAME:    in %manager : !axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 8>>{{.*}} outstanding_writes = 3, outstanding_reads = 3>
// CHECK-SAME:    out subordinate : !axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 16>>{{.*}} outstanding_writes = 4, outstanding_reads = 4>
hw.module @NarrowerSubordinateData(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 3, outstanding_reads = 3
  // CHECK: %[[CONV:.+]] = axi4.data_width_converter %clk, %rst_ni, %manager : (!axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 8>>{{.*}}) -> !axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 16>>{{.*}} outstanding_writes = 4, outstanding_reads = 4>
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 8>>
  // CHECK: hw.output %[[CONV]]
}

// -----

// A crossbar carries one data width, so a narrower manager is widened onto it
// before it routes, and the subordinate's IDs are narrowed after
// CHECK-LABEL: hw.module @NarrowerManagerData(
hw.module @NarrowerManagerData(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  // CHECK: %[[WIDENED:.+]] = axi4.data_width_converter %clk, %rst_ni, %manager : (!axi4.port<{{.*}} data_width = 32, {{.*}} burst_specs = <<incr, len = 16>>{{.*}}) -> !axi4.port<{{.*}} data_width = 64, {{.*}} burst_specs = <<incr, len = 8>>
  // CHECK: %[[XBAR:.+]] = axi4.xbar %clk, %rst_ni mgrs %[[WIDENED]]
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64
  // CHECK: %[[CONV:.+]] = axi4.id_width_converter %clk, %rst_ni, %[[XBAR]] : (!axi4.port<{{.*}} write_id_width = 2, read_id_width = 2, {{.*}}) -> !axi4.port<{{.*}} write_id_width = 3, read_id_width = 3,
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
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
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %mem_access with <<incr, len = 8>>
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
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 256>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %mem_access with <<incr, len = 256>>
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
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %bottom window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %core_access -> %mem_access with <<incr, len = 4>>
  // CHECK: hw.output %[[BOTTOM]]
}
