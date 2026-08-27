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
