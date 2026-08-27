// RUN: circt-opt %s --lower-axi4-dummies-to-axi --split-input-file --verify-diagnostics

// expected-error @below {{'hw.module' op cannot lower a dummies network in an instantiated module; its external endpoints must become ports of a top-level module}}
hw.module @Instantiated(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
}

hw.module @Top(in %clk : !seq.clock, in %rst_ni : i1) {
  hw.instance "inner" @Instantiated(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> ()
}

// -----

// A manager reaching nothing has no windows to annotate
hw.module @Disconnected(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.ext_manager' op must declare an access to reach a subordinate}}
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
}

// -----

// A manager can only be granted access to a subordinate it reaches
hw.module @Unreachable(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %other, %other_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %other_sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %other window <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
  // expected-error @below {{'axi4.dummies.accesses' op declares an access to a subordinate the manager cannot reach}}
  axi4.dummies.accesses %mgr_access -> %other_sub_access with <<incr, len = 16>>
  axi4.dummies.accesses %other_access -> %other_sub_access with <<incr, len = 16>>
}

// -----

hw.module @UnsupportedBursts(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // expected-error @below {{'axi4.dummies.accesses' op declares bursts #axi4.burst_set<<incr, len = 16>> the subordinate does not support in #axi4.window<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>}}
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
}

// -----

hw.module @NoAccesses(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.ext_manager' op must declare an access to reach a subordinate}}
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
}

// -----

hw.module @MismatchedAddresses(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // expected-error @below {{'axi4.dummies.ext_subordinate' op 'addr_width' (64) must match the manager's (32)}}
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 64, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
}

// -----

// The declared bursts convert cleanly onto the subordinate here, so what it
// cannot serve is the data width itself
hw.module @MismatchedData(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // expected-error @below {{'axi4.dummies.ext_subordinate' op 'data_width' (32) must match the manager's (64); inserting data width converters is not yet implemented}}
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 8>>
}

// -----

// A manager's bursts count beats of its own data width, so 16 beats of 64 bits
// is 32 beats of the subordinate's 32
hw.module @BurstsBeyondSubordinate(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  // expected-error @below {{'axi4.dummies.accesses' op declares bursts #axi4.burst_set<<incr, len = 16>> the subordinate does not support in #axi4.window<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>}}
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
}

// -----

// A single beat of 32 bits is half a beat of the subordinate's 64
hw.module @IndivisibleBursts(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // expected-error @below {{'axi4.dummies.accesses' op burst #axi4.burst_spec<incr, len = 1> does not divide into whole 64-bit beats}}
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 1>>
}

// -----

hw.module @MismatchedIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // expected-error @below {{'axi4.dummies.ext_subordinate' op needs different ID widths to the manager reaching it; inserting ID width converters is not yet implemented}}
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 2, outstanding_reads = 2
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
}

// -----

// A direct connection carries one port type, so an undersized subordinate is
// only visible here. It costs throughput rather than correctness.
hw.module @Bottleneck(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // expected-warning @below {{can hold fewer outstanding writes than the manager reaching it can issue (3 < 4)}}
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 3, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
}

// -----

hw.module @ConvertingXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  // expected-error @below {{'axi4.dummies.xbar' op 'data_width' (64) must match that of the port reaching it (32); inserting data width converters is not yet implemented}}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
}

// -----

hw.module @UnequalManagerIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 2, outstanding_reads = 2
  // expected-error @below {{'axi4.dummies.xbar' op is reached by ports needing different ID widths; inserting ID width converters is not yet implemented}}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %core_access -> %sub_access with <<incr, len = 16>>
  axi4.dummies.accesses %debug_access -> %sub_access with <<incr, len = 16>>
}

// -----

// A crossbar widens IDs to tag the manager a request came from, so a
// subordinate below one needs wider IDs than the managers above it
hw.module @NarrowSubordinateIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %debug, %debug_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64
  // expected-error @below {{'axi4.dummies.ext_subordinate' op needs different ID widths to the crossbar reaching it; inserting ID width converters is not yet implemented}}
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %core_access -> %sub_access with <<incr, len = 16>>
  axi4.dummies.accesses %debug_access -> %sub_access with <<incr, len = 16>>
}

// -----

hw.module @DanglingXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // expected-error @below {{'axi4.dummies.xbar' op must reach at least one subordinate}}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64
}

// -----

hw.module @Cycle(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.xbar' op is part of a cycle in the dummies network}}
  %ab = axi4.dummies.xbar %clk, %rst_ni mgrs %ba addr_width = 32, data_width = 64
  %ba = axi4.dummies.xbar %clk, %rst_ni mgrs %ab addr_width = 32, data_width = 64
}
