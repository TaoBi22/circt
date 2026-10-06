// RUN: circt-opt %s --lower-axi4-dummies-to-axi --split-input-file --verify-diagnostics

// expected-error @below {{'hw.module' op cannot lower a dummies network in an instantiated module; its external endpoints must become ports of a top-level module}}
hw.module @Instantiated(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

hw.module @Top(in %clk : !seq.clock, in %rst_ni : i1) {
  hw.instance "inner" @Instantiated(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> ()
}

// -----

// A manager reaching nothing has no windows to annotate
hw.module @Disconnected(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.ext_manager' op must declare an access to reach a subordinate}}
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
}

// -----

// A manager can only be granted access to a subordinate it reaches
hw.module @Unreachable(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %other, %other_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %other_sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %other windows <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  // expected-error @below {{'axi4.dummies.accesses' op declares an access to a subordinate the manager cannot reach}}
  axi4.dummies.accesses %mgr_access -> %other_sub_access with <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %other_access -> %other_sub_access with <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @UnsupportedBursts(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.accesses' op declares bursts #axi4.burst_set<<incr, len = 16>> the subordinate does not support in #axi4.window<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>}}
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @NoAccesses(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.ext_manager' op must declare an access to reach a subordinate}}
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
}

// -----

hw.module @MismatchedAddresses(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.ext_subordinate' op 'addr_width' (64) must match the manager's (32)}}
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 64, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

// A manager's bursts count beats of its own data width, so 16 beats of 64 bits
// is 32 beats of the subordinate's 32
hw.module @BurstsBeyondSubordinate(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 32, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.accesses' op declares bursts #axi4.burst_set<<incr, len = 16>> the subordinate does not support in #axi4.window<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>}}
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

// A single beat of 32 bits is half a beat of the subordinate's 64
hw.module @IndivisibleBursts(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 32, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.accesses' op burst #axi4.burst_spec<incr, len = 1> does not divide into whole 64-bit beats}}
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>
}

// -----

// A subordinate's port carries what reaches it rather than what it can hold, so
// an undersized subordinate is only visible here. It costs throughput rather
// than correctness.
hw.module @Bottleneck(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-warning @below {{can track fewer write IDs than reach it (3 < 4)}}
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 3, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @AccessPastWindow(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.accesses' op declares an access to #axi4.window<base = 0x0, last = 0x1fff, burst_specs = <<incr, len = 16>>> the subordinate does not serve in full}}
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0x1fff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @AccessAcrossGap(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>, <base = 0x2000, last = 0x2fff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.accesses' op declares an access to #axi4.window<base = 0x0, last = 0x2fff, burst_specs = <<incr, len = 16>>> the subordinate does not serve in full}}
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0x2fff, burst_specs = <<incr, len = 16>>>
}

// -----

// A manager's bursts have to be expressible at every data width on the way to
// the subordinate, the crossbar's included
hw.module @IndivisibleThroughXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 32, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.xbar' op burst #axi4.burst_spec<incr, len = 1> does not divide into whole 64-bit beats}}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 32, outstanding_write_ids = 8, outstanding_read_ids = 8, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>
}

// -----

// A subordinate below a crossbar that serves nothing as long as a beat of the
// crossbar's is unreachable through it
hw.module @UnreachableThroughXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.ext_subordinate' op supports no burst a port of 64 bits can ask for}}
  %periph_access = axi4.dummies.ext_subordinate "periph" %clk, %rst_ni, %xbar windows <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 1>>>> addr_width = 32, data_width = 32, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @DanglingXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.xbar' op must reach at least one subordinate}}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
}

// -----

hw.module @Cycle(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.xbar' op is part of a loop with no ID remapper, around which its IDs would grow without bound}}
  %ab = axi4.dummies.xbar %clk, %rst_ni mgrs %ba addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %ba = axi4.dummies.xbar %clk, %rst_ni mgrs %ab addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
}

// -----

hw.module @WideWindowBelowNarrowXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.xbar' op window #axi4.window<base = 0x100000000, last = 0x100000fff, burst_specs = <<incr, len = 16>>> does not fit in an 'addr_width' of 32}}
  %outer = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %inner = axi4.dummies.xbar %clk, %rst_ni mgrs %outer addr_width = 64, data_width = 64, upstream_concurrent_per_id = 4
  %high_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %inner windows <<base = 0x100000000, last = 0x100000fff, burst_specs = <<incr, len = 16>>>> addr_width = 64, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %low_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %outer windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %low_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @WideSubordinateBelowNestedXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.xbar' op window #axi4.window<base = 0x100000000, last = 0x100000fff, burst_specs = <<incr, len = 16>>> does not fit in an 'addr_width' of 32}}
  %outer = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %inner = axi4.dummies.xbar %clk, %rst_ni mgrs %outer addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %high_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %inner windows <<base = 0x100000000, last = 0x100000fff, burst_specs = <<incr, len = 16>>>> addr_width = 64, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %low_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %outer windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %low_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

// A crossbar carries every manager that can address a subordinate, and
// merging their 8 IDs onto the subordinate's 4 puts 2 requests on each
hw.module @BottleneckBelowXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %debug, %debug_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  // expected-warning @below {{can hold fewer writes per ID than reach it (1 < 2)}}
  // expected-warning @below {{can hold fewer reads per ID than reach it (1 < 2)}}
  %mem_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %debug_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @FilteredManager(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.ext_manager' op is marked 'pulp.atop_filter', but only a subordinate can have atomics filtered out in front of it}}
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1 {pulp.atop_filter}
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @ManagerClockDomain(in %clk : !seq.clock, in %other_clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.ext_manager' op is in a different clock domain to the 'axi4.dummies.xbar' connected to it}}
  %mgr, %mgr_access = axi4.dummies.ext_manager %other_clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-note @below {{connected operation here}}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @SubordinateResetDomain(in %clk : !seq.clock, in %rst_ni : i1, in %other_rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-note @below {{connected operation here}}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %mgr addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  // expected-error @below {{'axi4.dummies.ext_subordinate' op is in a different reset domain to the 'axi4.dummies.xbar' connected to it}}
  %sub_access = axi4.dummies.ext_subordinate %clk, %other_rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @DirectClockDomain(in %clk : !seq.clock, in %other_clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.ext_manager' op is in a different clock domain to the 'axi4.dummies.ext_subordinate' connected to it}}
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-note @below {{connected operation here}}
  %sub_access = axi4.dummies.ext_subordinate %other_clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @DanglingCut(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.cut' op must reach a subordinate}}
  %cut = axi4.dummies.cut %clk, %rst_ni, %mgr
}

// -----

hw.module @CutCycle(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.cut' op is part of a cycle in the dummies network}}
  %a = axi4.dummies.cut %clk, %rst_ni, %b
  %b = axi4.dummies.cut %clk, %rst_ni, %a
}

// -----

hw.module @CutThroughXbarCycle(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.xbar' op is part of a loop with no ID remapper, around which its IDs would grow without bound}}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %cut addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %cut = axi4.dummies.cut %clk, %rst_ni, %xbar
}

// -----

hw.module @ManagerCutClockDomain(in %clk : !seq.clock, in %other_clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.ext_manager' op is in a different clock domain to the 'axi4.dummies.cut' connected to it}}
  %mgr, %mgr_access = axi4.dummies.ext_manager %other_clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-note @below {{connected operation here}}
  %cut = axi4.dummies.cut %clk, %rst_ni, %mgr
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %cut windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @DanglingRemap(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.id_remap' op must reach a subordinate}}
  %remap = axi4.dummies.id_remap %clk, %rst_ni, %mgr max_unique_ids = 4, concurrent_per_id = 4
}

// -----

hw.module @RemapCycle(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.dummies.id_remap' op is part of a cycle in the dummies network}}
  %remap = axi4.dummies.id_remap %clk, %rst_ni, %cut max_unique_ids = 4, concurrent_per_id = 4
  %cut = axi4.dummies.cut %clk, %rst_ni, %remap
}

// -----

hw.module @LoopWithoutRemap(in %clk : !seq.clock, in %rst_ni : i1) {
  %soc, %soc_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %core, %core_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.xbar' op is part of a loop with no ID remapper, around which its IDs would grow without bound}}
  %q = axi4.dummies.xbar %clk, %rst_ni mgrs %soc, %c addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %c = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %q addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %tcdm_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %c windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 8, outstanding_read_ids = 8, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %out_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %q windows <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 8, outstanding_read_ids = 8, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %soc_access -> %tcdm_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %core_access -> %out_access with <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @NothingBelowLoop(in %clk : !seq.clock, in %rst_ni : i1) {
  %soc, %soc_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.xbar' op downstream port #1 reaches no subordinate without looping back through a crossbar}}
  %q = axi4.dummies.xbar %clk, %rst_ni mgrs %soc, %c addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %down = axi4.dummies.id_remap %clk, %rst_ni, %q max_unique_ids = 4, concurrent_per_id = 4
  %c = axi4.dummies.xbar %clk, %rst_ni mgrs %down addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %out_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %q windows <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 8, outstanding_read_ids = 8, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %soc_access -> %out_access with <base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @ConflictingAccesses(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.xbar' op downstream ports #0 and #1 have overlapping windows}}
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %a_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 8, outstanding_read_ids = 8, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %b_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 8, outstanding_read_ids = 8, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  axi4.dummies.accesses %core_access -> %a_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %core_access -> %b_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @AmbiguousAccess(in %clk : !seq.clock, in %rst_ni : i1) {
  %core, %core_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %a = axi4.dummies.xbar %clk, %rst_ni mgrs %core addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %left = axi4.dummies.cut %clk, %rst_ni, %a
  %right = axi4.dummies.cut %clk, %rst_ni, %a
  %b = axi4.dummies.xbar %clk, %rst_ni mgrs %left, %right addr_width = 32, data_width = 64, upstream_concurrent_per_id = 4
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %b windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 8, outstanding_read_ids = 8, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  // expected-error @below {{'axi4.dummies.accesses' op is ambiguous: the manager reaches the subordinate by more than one route}}
  axi4.dummies.accesses %core_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

hw.module @PerIdBottleneck(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr, %mgr_access = axi4.dummies.ext_manager %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 4, concurrent_reads_per_id = 2
  // expected-warning @below {{can hold fewer writes per ID than reach it (2 < 4)}}
  %sub_access = axi4.dummies.ext_subordinate %clk, %rst_ni, %mgr windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 2, concurrent_reads_per_id = 2
  axi4.dummies.accesses %mgr_access -> %sub_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}

// -----

// A subordinate is measured against what the budgets on the way to it let
// through, not what the managers issue
hw.module @NoBottleneckPastBudgets(in %clk : !seq.clock, in %rst_ni : i1) {
  %dma, %dma_access = axi4.dummies.ext_manager "dma" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 7, concurrent_reads_per_id = 7
  %core, %core_access = axi4.dummies.ext_manager "core" %clk, %rst_ni addr_width = 32, data_width = 64, outstanding_write_ids = 4, outstanding_read_ids = 4, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1
  %remap = axi4.dummies.id_remap %clk, %rst_ni, %dma max_unique_ids = 4, concurrent_per_id = 4
  %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %remap, %core addr_width = 32, data_width = 64, upstream_concurrent_per_id = 3
  %mem_access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %xbar windows <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>> addr_width = 32, data_width = 64, outstanding_write_ids = 8, outstanding_read_ids = 8, concurrent_writes_per_id = 3, concurrent_reads_per_id = 3
  axi4.dummies.accesses %dma_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
  axi4.dummies.accesses %core_access -> %mem_access with <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
}
