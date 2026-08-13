// RUN: circt-opt %s --verify-axi4-networks --split-input-file --verify-diagnostics

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// A well-formed network produces no diagnostics
hw.module @Clean(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  axi4.abstract_subordinate %clk, %rst_ni, %mgr : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// expected-error @below {{AXI4 port must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
hw.module @BlockArgFanout(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port) {
  axi4.abstract_subordinate %clk, %rst_ni, %port : !port
  axi4.abstract_subordinate %clk, %rst_ni, %port : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !port)

hw.module @InstanceFanout(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{AXI4 port must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
  %axi = hw.instance "mgr" @Manager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !port)
  axi4.abstract_subordinate %clk, %rst_ni, %axi : !port
  axi4.abstract_subordinate %clk, %rst_ni, %axi : !port
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
  axi4.abstract_subordinate %other_clk, %rst_ni, %mgr : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ResetCrossing(in %clk : !seq.clock, in %rst_ni : i1, in %other_rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-error @below {{'axi4.abstract_subordinate' op is in a different reset domain to the 'axi4.abstract_manager' connected to it}}
  axi4.abstract_subordinate %clk, %other_rst_ni, %mgr : !port
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 5, outstanding_reads = 8>

hw.module @Undersized(in %clk : !seq.clock, in %rst_ni : i1) {
  %a = axi4.abstract_manager %clk, %rst_ni : !mgr
  %b = axi4.abstract_manager %clk, %rst_ni : !mgr
  // expected-warning @below {{downstream port #0 can hold fewer outstanding writes than the managers reaching it can issue (5 < 8)}}
  %sub = axi4.xbar %clk, %rst_ni mgrs %a, %b : (!mgr, !mgr) -> !sub
  axi4.abstract_subordinate %clk, %rst_ni, %sub : !sub
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!other_mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!other_sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Ensure only the managers whose windows reach a port count towards its total
hw.module @DisjointManagers(in %clk : !seq.clock, in %rst_ni : i1) {
  %a = axi4.abstract_manager %clk, %rst_ni : !mgr
  %b = axi4.abstract_manager %clk, %rst_ni : !other_mgr
  %sub, %other = axi4.xbar %clk, %rst_ni mgrs %a, %b : (!mgr, !other_mgr) -> (!sub, !other_sub)
  axi4.abstract_subordinate %clk, %rst_ni, %sub : !sub
  axi4.abstract_subordinate %clk, %rst_ni, %other : !other_sub
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @CutCrossing(in %clk : !seq.clock, in %other_clk : !seq.clock,
                       in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  // expected-error @below {{'axi4.cut' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  %cut = axi4.cut %other_clk, %rst_ni, %mgr : !port
  axi4.abstract_subordinate %other_clk, %rst_ni, %cut : !port
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
  axi4.abstract_subordinate %clk, %other_rst_ni, %dwc : !thin
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// A cdc is the one op allowed to change clock domain, so neither side is a
// crossing
hw.module @CdcCrosses(in %clk : !seq.clock, in %other_clk : !seq.clock,
                      in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  %cdc = axi4.cdc from %clk to %other_clk, %rst_ni, %mgr : !port
  axi4.abstract_subordinate %other_clk, %rst_ni, %cdc : !port
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
  axi4.abstract_subordinate %clk, %rst_ni, %cdc : !port
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
  axi4.abstract_subordinate %other_clk, %other_rst_ni, %cdc : !port
}

// -----

!up = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!down = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 8>>>>, outstanding_writes = 2, outstanding_reads = 4>

// A converter's downstream port is a bottleneck the same way a crossbar's is
hw.module @UndersizedConverter(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !up
  // expected-warning @below {{downstream port can hold fewer outstanding writes than the upstream port can issue (2 < 4)}}
  %dwc = axi4.data_width_converter %clk, %rst_ni, %mgr : (!up) -> !down
  axi4.abstract_subordinate %clk, %rst_ni, %dwc : !down
}

// -----

!burstty = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!beats = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>>, outstanding_writes = 16, outstanding_reads = 16>

hw.module @SplitterCrossing(in %clk : !seq.clock, in %other_clk : !seq.clock,
                            in %rst_ni : i1) {
  // expected-note @below {{connected operation here}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !burstty
  // expected-error @below {{'axi4.burst_splitter' op is in a different clock domain to the 'axi4.abstract_manager' connected to it}}
  %split = axi4.burst_splitter %other_clk, %rst_ni, %mgr : (!burstty) -> !beats
  axi4.abstract_subordinate %other_clk, %rst_ni, %split : !beats
}

// -----

!burstty = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 8>>>>, outstanding_writes = 2, outstanding_reads = 2>
!beats = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>>, outstanding_writes = 16, outstanding_reads = 4>

// A splitter needs a downstream slot per beat of every burst in flight, so the
// writes have exactly enough here and the reads do not
hw.module @UndersizedSplitter(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !burstty
  // expected-warning @below {{downstream port can hold fewer outstanding reads than splitting the upstream port's bursts of up to 8 beats can issue (4 < 16)}}
  %split = axi4.burst_splitter %clk, %rst_ni, %mgr : (!burstty) -> !beats
  axi4.abstract_subordinate %clk, %rst_ni, %split : !beats
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
  axi4.abstract_subordinate %other_clk, %rst_ni, %a : !lo
  axi4.abstract_subordinate %other_clk, %rst_ni, %b : !hi
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 2, outstanding_reads = 4>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Each leg of a demux carries all of the manager's transactions, since it does
// not know which of them the manager will send its way
hw.module @UndersizedDemux(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !mgr
  // expected-warning @below {{downstream port #0 can hold fewer outstanding writes than the managers reaching it can issue (2 < 4)}}
  %a, %b = axi4.demux %clk, %rst_ni, %mgr : (!mgr) -> (!lo, !hi)
  axi4.abstract_subordinate %clk, %rst_ni, %a : !lo
  axi4.abstract_subordinate %clk, %rst_ni, %b : !hi
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
  axi4.abstract_subordinate %clk, %other_rst_ni, %sub : !sub
}

// -----

!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 4>

// A mux's downstream port carries every manager's transactions at once
hw.module @UndersizedMux(in %clk : !seq.clock, in %rst_ni : i1) {
  %a = axi4.abstract_manager %clk, %rst_ni : !lo
  %b = axi4.abstract_manager %clk, %rst_ni : !hi
  // expected-warning @below {{downstream port #0 can hold fewer outstanding reads than the managers reaching it can issue (4 < 8)}}
  %sub = axi4.mux %clk, %rst_ni, %a, %b : (!lo, !hi) -> !sub
  axi4.abstract_subordinate %clk, %rst_ni, %sub : !sub
}
