// RUN: circt-opt %s --verify-axi4-networks --split-input-file --verify-diagnostics

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// A well-formed network produces no diagnostics
hw.module @Clean(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  axi4.abstract_subordinate %clk, %rst_ni, %mgr concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// expected-error @below {{AXI4 port must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
hw.module @BlockArgFanout(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port) {
  axi4.abstract_subordinate %clk, %rst_ni, %port concurrent_writes 4 concurrent_reads 4 : !port
  axi4.abstract_subordinate %clk, %rst_ni, %port concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !port)

hw.module @InstanceFanout(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{AXI4 port must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
  %axi = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !port)
  axi4.abstract_subordinate %clk, %rst_ni, %axi concurrent_writes 4 concurrent_reads 4 : !port
  axi4.abstract_subordinate %clk, %rst_ni, %axi concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @Dangling(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-warning @below {{AXI4 port has no uses, so takes no part in a network}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ClockCrossing(in %clk : !seq.clock, in %other_clk : !seq.clock, in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-error @below {{'axi4.abstract_subordinate' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  axi4.abstract_subordinate %other_clk, %rst_ni, %mgr concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ResetCrossing(in %clk : !seq.clock, in %rst_ni : i1, in %other_rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-error @below {{'axi4.abstract_subordinate' op is in a different reset domain to the 'axi4.abstract_manager' connected to it}}
  axi4.abstract_subordinate %clk, %other_rst_ni, %mgr concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UndersizedSubordinate(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-warning @below {{endpoint can handle fewer writes than the port reaching it can have concurrently outstanding (2 < 4)}}
  axi4.abstract_subordinate %clk, %rst_ni, %mgr concurrent_writes 2 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 3, user_width = 4, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!b = !hw.struct<id: i5, resp: i2, user: i4>
!r = !hw.struct<id: i3, data: i64, resp: i2, last: i1, user: i4>

hw.module @UndersizedBridge(in %clk : !seq.clock, in %rst_ni : i1,
                            in %port : !port, in %aw_ready : i1,
                            in %w_ready : i1, in %b : !b, in %b_valid : i1,
                            in %ar_ready : i1, in %r : !r, in %r_valid : i1) {
  // expected-warning @+2 {{endpoint can handle fewer reads than the port reaching it can have concurrently outstanding (1 < 4)}}
  %aw, %aw_valid, %w, %w_valid, %b_ready, %ar, %ar_valid, %r_ready =
    axi4.port_to_channel_structs %clk, %rst_ni, %port
      aw %aw_ready w %w_ready b %b, %b_valid
      ar %ar_ready r %r, %r_valid
      concurrent_writes 4 concurrent_reads 1 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @CutCrossing(in %clk : !seq.clock, in %other_clk : !seq.clock,
                       in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-error @below {{'axi4.cut' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  %cut = axi4.cut %other_clk, %rst_ni, %mgr : !port
  axi4.abstract_subordinate %other_clk, %rst_ni, %cut concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ConverterCrossing(in %clk : !seq.clock, in %rst_ni : i1,
                             in %other_rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !wide
  // expected-error @below {{'axi4.data_width_converter' op is in a different reset domain to the 'axi4.abstract_manager' connected to it}}
  %dwc = axi4.data_width_converter %clk, %other_rst_ni, %mgr : (!wide) -> !thin
  axi4.abstract_subordinate %clk, %other_rst_ni, %dwc concurrent_writes 4 concurrent_reads 4 : !thin
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// A cdc is the one op allowed to change clock domain, so neither side is a
// crossing
hw.module @CdcCrosses(in %clk : !seq.clock, in %other_clk : !seq.clock,
                      in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  %cdc = axi4.cdc from %clk to %other_clk, %rst_ni, %mgr : !port
  axi4.abstract_subordinate %other_clk, %rst_ni, %cdc concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Its upstream clock must still be its producer's
hw.module @CdcFromWrongClock(in %clk : !seq.clock, in %other_clk : !seq.clock,
                             in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-error @below {{'axi4.cdc' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  %cdc = axi4.cdc from %other_clk to %clk, %rst_ni, %mgr : !port
  axi4.abstract_subordinate %clk, %rst_ni, %cdc concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// And a cdc is not a reset crossing
hw.module @CdcCrossesReset(in %clk : !seq.clock, in %other_clk : !seq.clock,
                           in %rst_ni : i1, in %other_rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-error @below {{'axi4.cdc' op is in a different reset domain to the 'axi4.abstract_manager' connected to it}}
  %cdc = axi4.cdc from %clk to %other_clk, %other_rst_ni, %mgr : !port
  axi4.abstract_subordinate %other_clk, %other_rst_ni, %cdc concurrent_writes 4 concurrent_reads 4 : !port
}

// -----

!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @IdConverterCrossing(in %clk : !seq.clock,
                               in %other_clk : !seq.clock, in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !wide_ids
  // expected-error @below {{'axi4.id_width_converter' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  %iwc = axi4.id_width_converter %other_clk, %rst_ni, %mgr : (!wide_ids) -> !narrow_ids
  axi4.abstract_subordinate %other_clk, %rst_ni, %iwc concurrent_writes 4 concurrent_reads 4 : !narrow_ids
}

// -----

!burstty = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!beats = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @SplitterCrossing(in %clk : !seq.clock, in %other_clk : !seq.clock,
                            in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !burstty
  // expected-error @below {{'axi4.burst_splitter' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  %split = axi4.burst_splitter %other_clk, %rst_ni, %mgr : (!burstty) -> !beats
  axi4.abstract_subordinate %other_clk, %rst_ni, %split concurrent_writes 4 concurrent_reads 4 : !beats
}

// -----

!wrapping = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!unwrapped = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UnwrapperCrossing(in %clk : !seq.clock, in %other_clk : !seq.clock,
                             in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !wrapping
  // expected-error @below {{'axi4.burst_unwrapper' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  %unwrapped = axi4.burst_unwrapper %other_clk, %rst_ni, %mgr : (!wrapping) -> !unwrapped
  axi4.abstract_subordinate %other_clk, %rst_ni, %unwrapped concurrent_writes 4 concurrent_reads 4 : !unwrapped
}

// -----

// Without a wrapping burst to split there is nothing to double, so an
// unwrapper's slots pass straight through
!incrementing = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UnwrapperWithoutWraps(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !incrementing
  %unwrapped = axi4.burst_unwrapper %clk, %rst_ni, %mgr : (!incrementing) -> !incrementing
  axi4.abstract_subordinate %clk, %rst_ni, %unwrapped concurrent_writes 4 concurrent_reads 4 : !incrementing
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @DemuxCrossing(in %clk : !seq.clock, in %other_clk : !seq.clock,
                         in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !mgr
  // expected-error @below {{'axi4.demux' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  %a, %b = axi4.demux %other_clk, %rst_ni, %mgr : (!mgr) -> (!lo, !hi)
  axi4.abstract_subordinate %other_clk, %rst_ni, %a concurrent_writes 4 concurrent_reads 4 : !lo
  axi4.abstract_subordinate %other_clk, %rst_ni, %b concurrent_writes 4 concurrent_reads 4 : !hi
}

// -----

!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>

hw.module @MuxCrossing(in %clk : !seq.clock, in %rst_ni : i1,
                       in %other_rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %a = axi4.abstract_manager %clk, %rst_ni : !lo
  %b = axi4.abstract_manager %clk, %other_rst_ni : !hi
  // expected-error @below {{'axi4.mux' op is in a different reset domain to the 'axi4.abstract_manager' connected to it}}
  %sub = axi4.mux %clk, %other_rst_ni, %a, %b : (!lo, !hi) -> !sub
  axi4.abstract_subordinate %clk, %other_rst_ni, %sub concurrent_writes 8 concurrent_reads 8 : !sub
}

// -----

!mem_port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// A to_mem ends the network in the memory it fronts, so its port is the only
// connection to check
hw.module @ToMem(in %clk : !seq.clock, in %rst_ni : i1,
                 in %rvalid : i1, in %rdata : i64) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !mem_port
  %valid, %addr, %wdata, %strb, %we = axi4.to_mem %clk, %rst_ni, %mgr read %rvalid, %rdata : !mem_port
}

// -----

!mem_port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ToMemCrossing(in %clk : !seq.clock, in %other_clk : !seq.clock,
                         in %rst_ni : i1, in %rvalid : i1, in %rdata : i64) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !mem_port
  // expected-error @below {{'axi4.to_mem' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  %valid, %addr, %wdata, %strb, %we = axi4.to_mem %other_clk, %rst_ni, %mgr read %rvalid, %rdata : !mem_port
}
