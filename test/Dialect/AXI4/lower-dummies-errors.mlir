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

// A direct connection carries one port type, so an undersized subordinate is
// only visible here. It costs throughput rather than correctness.
hw.module @Bottleneck(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // expected-warning @below {{can hold fewer outstanding writes than the manager reaching it can issue (3 < 4)}}
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 3, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 16>>
}

// -----

// A manager's bursts have to be expressible at every data width on the way to
// the subordinate, the crossbar's included
hw.module @IndivisibleThroughXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  // expected-error @below {{'axi4.dummies.xbar' op burst #axi4.burst_spec<incr, len = 1> does not divide into whole 64-bit beats}}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 32, outstanding_writes = 8, outstanding_reads = 8
  axi4.dummies.accesses %mgr_access -> %sub_access with <<incr, len = 1>>
}

// -----

// A subordinate below a crossbar that serves nothing as long as a beat of the
// crossbar's is unreachable through it
hw.module @UnreachableThroughXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>> addr_width = 32, data_width = 64, outstanding_writes = 4, outstanding_reads = 4
  // expected-error @below {{'axi4.dummies.ext_subordinate' op supports no burst a port of 64 bits can ask for}}
  %periph_access = axi4.dummies.ext_subordinate "periph" %clk, %rst_ni, %xbar window <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 1>>> addr_width = 32, data_width = 32, outstanding_writes = 4, outstanding_reads = 4
  axi4.dummies.accesses %mgr_access -> %mem_access with <<incr, len = 16>>
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
