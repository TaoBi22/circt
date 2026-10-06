// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true --split-input-file --verify-diagnostics

// Every wrapper types its ID fields through a typedef, so a port with no ID
// bits has nothing to declare them from
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 0, read_id_width = 0, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 0, read_id_width = 0, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 1, concurrent_reads_per_id = 1>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !mgr)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !sub)

hw.module @NoIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op cannot be lowered to PULP because its upstream port #0 has a zero-width write ID, which PULP cannot express}}
  %s = axi4.xbar %clk, %rst_ni mgrs %m upstream_concurrent_per_id 4 : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

// PULP's axi_xbar has one ID width per side, shared by writes and reads
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !mgr)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !sub)

hw.module @SplitUpstreamIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op cannot be lowered to a PULP axi_xbar, which uses a single ID width per side, because its upstream write ID width (4) and read ID width (3) differ}}
  %s = axi4.xbar %clk, %rst_ni mgrs %m upstream_concurrent_per_id 4 : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

// One ID width is shared by every downstream port too, though the xbar
// verifier lets them widen independently
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!sub_lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0x7ff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!sub_hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 6, read_id_width = 6, user_width = 0, windows = <<base = 0x800, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !mgr)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Low(in %axi : !sub_lo)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @High(in %axi : !sub_hi)

hw.module @MixedDownstreamIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op cannot be lowered to a PULP axi_xbar, which uses one ID width for every downstream port, because downstream port #1's ID width (6) differs from downstream port #0's (5)}}
  %lo, %hi = axi4.xbar %clk, %rst_ni mgrs %m upstream_concurrent_per_id 4 : (!mgr) -> (!sub_lo, !sub_hi)
  hw.instance "lo" @Low(axi: %lo: !sub_lo) -> ()
  hw.instance "hi" @High(axi: %hi: !sub_hi) -> ()
}

// -----

// PULP widens the ID by exactly the bits it needs to tag its managers, so a
// downstream port wider than that is not expressible, though the xbar verifier
// allows it
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !mgr)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !sub)

hw.module @OverWideForOneManager(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op cannot be lowered to a PULP axi_xbar, which widens IDs by exactly the 0 bits needed to tag 1 manager, so its downstream ID width must be 4, not 5}}
  %s = axi4.xbar %clk, %rst_ni mgrs %m upstream_concurrent_per_id 4 : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

// Two managers need one tag bit, so the downstream width must be exactly one
// wider - no more
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 6, read_id_width = 6, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !mgr)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !sub)

hw.module @OverWideForTwoManagers(in %clk : !seq.clock, in %rst_ni : i1) {
  %a = hw.instance "mgr_a" @Manager() -> (axi: !mgr)
  %b = hw.instance "mgr_b" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op cannot be lowered to a PULP axi_xbar, which widens IDs by exactly the 1 bits needed to tag 2 managers, so its downstream ID width must be 5, not 6}}
  %s = axi4.xbar %clk, %rst_ni mgrs %a, %b upstream_concurrent_per_id 4 : (!mgr, !mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @SplitMuxIds(in %clk : !seq.clock, in %rst_ni : i1,
                       in %a : !mgr, in %b : !mgr, out downstream : !sub) {
  // expected-error @below {{'axi4.mux' op cannot be lowered to a PULP axi_mux, which uses a single ID width per side, because its upstream write ID width (4) and read ID width (3) differ}}
  %downstream = axi4.mux %clk, %rst_ni, %a, %b : (!mgr, !mgr) -> !sub
  hw.output %downstream : !sub
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @SplitMuxDownstreamIds(in %clk : !seq.clock, in %rst_ni : i1,
                                 in %upstream : !mgr, out downstream : !sub) {
  // expected-error @below {{'axi4.mux' op cannot be lowered to a PULP axi_mux, which uses a single ID width per side, because its downstream write ID width (4) and read ID width (5) differ}}
  %downstream = axi4.mux %clk, %rst_ni, %upstream : (!mgr) -> !sub
  hw.output %downstream : !sub
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 6, read_id_width = 6, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @OverWideMuxIds(in %clk : !seq.clock, in %rst_ni : i1,
                          in %a : !mgr, in %b : !mgr, out downstream : !sub) {
  // expected-error @below {{'axi4.mux' op cannot be lowered to a PULP axi_mux, which widens IDs by exactly the 1 bits needed to tag 2 managers, so its downstream ID width must be 5, not 6}}
  %downstream = axi4.mux %clk, %rst_ni, %a, %b : (!mgr, !mgr) -> !sub
  hw.output %downstream : !sub
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 3, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @SplitDemuxIds(in %clk : !seq.clock, in %rst_ni : i1,
                         in %upstream : !mgr, out lo : !lo, out hi : !hi) {
  // expected-error @below {{'axi4.demux' op cannot be lowered to a PULP axi_demux, which uses a single ID width, because its write ID width (4) and read ID width (3) differ}}
  %a, %b = axi4.demux %clk, %rst_ni, %upstream upstream_concurrent_per_id 4 : (!mgr) -> (!lo, !hi)
  hw.output %a, %b : !lo, !hi
}

// -----

// PULP's axi_dw_converter converts over a single ID width, shared by writes and
// reads, so the two must agree - unlike a cut, which never inspects them
!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 8>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 1>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !wide)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !thin)

hw.module @SplitIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !wide)
  // expected-error @below {{'axi4.data_width_converter' op cannot be lowered to a PULP axi_dw_converter, which uses a single ID width, because its write ID width (4) and read ID width (2) differ}}
  %dwc = axi4.data_width_converter %clk, %rst_ni, %m max_unique_read_ids 4 : (!wide) -> !thin
  hw.instance "sub" @Subordinate(axi: %dwc: !thin) -> ()
}

// -----

// PULP's axi_iw_converter converts over a single ID width per side, since it
// re-tags both channels together
!split_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @SplitIdConverterIds(in %clk : !seq.clock, in %rst_ni : i1,
                               in %upstream : !split_ids,
                               out downstream : !narrow_ids) {
  // expected-error @below {{'axi4.id_width_converter' op cannot be lowered to a PULP axi_iw_converter, which uses a single ID width per side, because its upstream write ID width (4) and read ID width (3) differ}}
  %iwc = axi4.id_width_converter %clk, %rst_ni, %upstream max_unique_ids = 4, concurrent_per_id = 4 : (!split_ids) -> !narrow_ids
  hw.output %iwc : !narrow_ids
}

// -----

// And the downstream side is checked the same way
!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!split_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @SplitIdConverterDownstreamIds(in %clk : !seq.clock, in %rst_ni : i1,
                                         in %upstream : !wide_ids,
                                         out downstream : !split_ids) {
  // expected-error @below {{'axi4.id_width_converter' op cannot be lowered to a PULP axi_iw_converter, which uses a single ID width per side, because its downstream write ID width (2) and read ID width (3) differ}}
  %iwc = axi4.id_width_converter %clk, %rst_ni, %upstream max_unique_ids = 4, concurrent_per_id = 4 : (!wide_ids) -> !split_ids
  hw.output %iwc : !split_ids
}

// -----

// PULP's axi_burst_splitter splits over a single ID width too
!burstty = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!beats = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @SplitterIds(in %clk : !seq.clock, in %rst_ni : i1,
                       in %upstream : !burstty, out downstream : !beats) {
  // expected-error @below {{'axi4.burst_splitter' op cannot be lowered to a PULP axi_burst_splitter, which uses a single ID width, because its write ID width (4) and read ID width (2) differ}}
  %split = axi4.burst_splitter %clk, %rst_ni, %upstream concurrent_writes 4 concurrent_reads 4 : (!burstty) -> !beats
  hw.output %split : !beats
}

// -----

// PULP's burst splitter does not support wrap bursts
!wrapping = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!beats = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @SplitterWrappingBurst(in %clk : !seq.clock, in %rst_ni : i1,
                                 in %upstream : !wrapping, out downstream : !beats) {
  // expected-error @below {{'axi4.burst_splitter' op cannot be lowered to a PULP axi_burst_splitter, which does not support wrapping bursts, because its upstream port issues #axi4.burst_spec<wrap, len = 4>}}
  %split = axi4.burst_splitter %clk, %rst_ni, %upstream concurrent_writes 4 concurrent_reads 4 : (!wrapping) -> !beats
  hw.output %split : !beats
}

// -----

// PULP's axi_burst_unwrap unwraps over a single ID width
!split_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!unwrapped = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @UnwrapperIds(in %clk : !seq.clock, in %rst_ni : i1,
                        in %upstream : !split_ids, out downstream : !unwrapped) {
  // expected-error @below {{'axi4.burst_unwrapper' op cannot be lowered to a PULP axi_burst_unwrap, which uses a single ID width, because its write ID width (4) and read ID width (2) differ}}
  %unwrapped = axi4.burst_unwrapper %clk, %rst_ni, %upstream concurrent_writes 4 concurrent_reads 4 : (!split_ids) -> !unwrapped
  hw.output %unwrapped : !unwrapped
}

// -----

// A 16 beat wrap of 128 byte beats totals 2048 bytes, which overflows the 11
// bits PULP totals it in
!wrapping = !axi4.port<addr_width = 32, data_width = 1024, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xffff, burst_specs = <<wrap, len = 16>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!unwrapped = !axi4.port<addr_width = 32, data_width = 1024, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xffff, burst_specs = <<incr, len = 16>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @UnwrapperOversizedContainer(in %clk : !seq.clock, in %rst_ni : i1,
                                       in %upstream : !wrapping,
                                       out downstream : !unwrapped) {
  // expected-error @below {{'axi4.burst_unwrapper' op cannot be lowered to a PULP axi_burst_unwrap, which computes a wrapping burst's total size in 11 bits and so supports at most 2047 bytes, because its upstream port issues #axi4.burst_spec<wrap, len = 16> over 1024-bit beats, totalling 2048 bytes}}
  %unwrapped = axi4.burst_unwrapper %clk, %rst_ni, %upstream concurrent_writes 4 concurrent_reads 4 : (!wrapping) -> !unwrapped
  hw.output %unwrapped : !unwrapped
}

// -----

// PULP's axi_to_mem converts over a single ID width
!split_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @ToMemIds(in %clk : !seq.clock, in %rst_ni : i1, in %port : !split_ids,
                    in %rvalid : i1, in %rdata : i64,
                    out valid : i1, out addr : i32, out wdata : i64,
                    out strb : i8, out we : i1) {
  // expected-error @below {{'axi4.to_mem' op cannot be lowered to a PULP axi_to_mem, which uses a single ID width, because its write ID width (4) and read ID width (2) differ}}
  %valid, %addr, %wdata, %strb, %we = axi4.to_mem %clk, %rst_ni, %port read %rvalid, %rdata : !split_ids
  hw.output %valid, %addr, %wdata, %strb, %we : i1, i32, i64, i8, i1
}

// -----

// A memory is walked up its addresses, so a burst of several beats that does
// not increment has nowhere to put them
!wrapping = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @ToMemWrappingBurst(in %clk : !seq.clock, in %rst_ni : i1,
                              in %port : !wrapping, in %rvalid : i1, in %rdata : i64,
                              out valid : i1, out addr : i32, out wdata : i64,
                              out strb : i8, out we : i1) {
  // expected-error @below {{'axi4.to_mem' op cannot be lowered to a PULP axi_to_mem, which supports bursts of more than one beat only where they increment, because its port issues #axi4.burst_spec<wrap, len = 4>}}
  %valid, %addr, %wdata, %strb, %we = axi4.to_mem %clk, %rst_ni, %port read %rvalid, %rdata : !wrapping
  hw.output %valid, %addr, %wdata, %strb, %we : i1, i32, i64, i8, i1
}

// -----

// The wrapper's typedefs are built from the ports, so config cannot change a
// parameter derived from them
!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @ConfigDerived(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port, out out : !port) {
  // expected-error @below {{'axi4.cut' op cannot set PULP parameter 'axi_req_t' through 'PULP_CONFIG_axi_req_t', because the wrapper derives it from the op}}
  %cut = axi4.cut %clk, %rst_ni, %port {PULP_CONFIG_axi_req_t = "logic"} : !port
  hw.output %cut : !port
}

// -----

// Nor can it change a field of the crossbar's Cfg derived from them
!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @ConfigDerivedCfg(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port, out out : !port) {
  // expected-error @below {{'axi4.xbar' op cannot set PULP parameter 'AxiAddrWidth' through 'PULP_CONFIG_AxiAddrWidth', because the wrapper derives it from the op}}
  %s = axi4.xbar %clk, %rst_ni mgrs %port upstream_concurrent_per_id 4 {PULP_CONFIG_AxiAddrWidth = 64 : i32} : (!port) -> (!port)
  hw.output %s : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @ConfigMaxMstTrans(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port, out out : !port) {
  // expected-error @below {{'axi4.xbar' op cannot set PULP parameter 'MaxMstTrans' through 'PULP_CONFIG_MaxMstTrans', because the wrapper derives it from the op}}
  %s = axi4.xbar %clk, %rst_ni mgrs %port upstream_concurrent_per_id 4 {PULP_CONFIG_MaxMstTrans = 8 : i32} : (!port) -> (!port)
  hw.output %s : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @ConfigMaxSlvTrans(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port, out out : !port) {
  // expected-error @below {{'axi4.xbar' op cannot set PULP parameter 'MaxSlvTrans' through 'PULP_CONFIG_MaxSlvTrans', because the wrapper derives it from the op}}
  %s = axi4.xbar %clk, %rst_ni mgrs %port upstream_concurrent_per_id 4 {PULP_CONFIG_MaxSlvTrans = 16 : i32} : (!port) -> (!port)
  hw.output %s : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @ConfigValue(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port, out out : !port) {
  // expected-error @below {{'axi4.cut' op has 'PULP_CONFIG_Bypass', which must be an integer, a string or a matrix of booleans to set a PULP parameter}}
  %cut = axi4.cut %clk, %rst_ni, %port {PULP_CONFIG_Bypass = [true]} : !port
  hw.output %cut : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @ConfigNoName(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port, out out : !port) {
  // expected-error @below {{'axi4.cut' op has a 'PULP_CONFIG_' attribute with no parameter name after the prefix}}
  %cut = axi4.cut %clk, %rst_ni, %port {PULP_CONFIG_ = 1 : i32} : !port
  hw.output %cut : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @ConnectivityShape(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port, out out : !port) {
  // expected-error @below {{'axi4.xbar' op has 'PULP_CONFIG_Connectivity', which must have a row for each of its 1 upstream ports with a column for each of its 1 downstream ports}}
  %s = axi4.xbar %clk, %rst_ni mgrs %port upstream_concurrent_per_id 4 {PULP_CONFIG_Connectivity = [[true, false]]} : (!port) -> (!port)
  hw.output %s : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @AtopsDisabled(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port {pulp.atops}, out out : !port {pulp.atops}) {
  // expected-error @below {{'axi4.xbar' op has 'PULP_CONFIG_ATOPs', which must be true because it carries atomics from a port marked 'pulp.atops'}}
  %s = axi4.xbar %clk, %rst_ni mgrs %port upstream_concurrent_per_id 4 {PULP_CONFIG_ATOPs = false} : (!port) -> (!port)
  hw.output %s : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-note @below {{atomics issued from the port marked 'pulp.atops' here}}
hw.module @AtopsToUnmarkedOutput(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port {pulp.atops}, out out : !port) {
  %cut = axi4.cut %clk, %rst_ni, %port : !port
  // expected-error @below {{atomics reach output port 'out', which is marked neither 'pulp.atops' to accept them nor 'pulp.atop_filter' to filter them out}}
  hw.output %cut : !port
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!mem = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!periph = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Core(out axi : !mgr {pulp.atops})
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Mem(in %axi : !mem {pulp.atops})
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Periph(in %axi : !periph)

// The core addresses the peripheral as well as the memory, so its atomics can
// reach both
hw.module @AtopsToUnmarkedInstance(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-note @below {{atomics issued from the port marked 'pulp.atops' here}}
  %c = hw.instance "core" @Core() -> (axi: !mgr)
  %mem, %periph = axi4.demux %clk, %rst_ni, %c upstream_concurrent_per_id 4 : (!mgr) -> (!mem, !periph)
  hw.instance "mem" @Mem(axi: %mem: !mem) -> ()
  // expected-error @below {{atomics reach port 'axi' of instance 'periph', which is marked neither 'pulp.atops' to accept them nor 'pulp.atop_filter' to filter them out}}
  hw.instance "periph" @Periph(axi: %periph: !periph) -> ()
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!split = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-note @below {{atomics issued from the port marked 'pulp.atops' here}}
hw.module @AtopsThroughSplitter(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port {pulp.atops}, out out : !split {pulp.atops}) {
  // expected-error @below {{'axi4.burst_splitter' op cannot carry atomics, which PULP's axi_burst_splitter answers with an error}}
  %split = axi4.burst_splitter %clk, %rst_ni, %port concurrent_writes 4 concurrent_reads 4 : (!port) -> !split
  hw.output %split : !split
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-note @below {{atomics issued from the port marked 'pulp.atops' here}}
hw.module @AtopsToMem(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port {pulp.atops}, in %rvalid : i1, in %rdata : i64) {
  // expected-error @below {{'axi4.to_mem' op cannot accept atomics, because it has no result to carry the atop PULP's axi_to_mem passes on to the memory}}
  %valid, %addr, %wdata, %strb, %we = axi4.to_mem %clk, %rst_ni, %port read %rvalid, %rdata : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-note @below {{atomics issued from the port marked 'pulp.atops' here}}
hw.module @FilterWithoutDomain(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port {pulp.atops}, out out : !port {pulp.atop_filter}) {
  // expected-error @below {{atomics reach output port 'out', which is marked 'pulp.atop_filter', but no component drives it to take a clock and reset for the filter from}}
  hw.output %port : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @FilterSplitIds(in %clk : !seq.clock, in %rst_ni : i1, in %port : !port {pulp.atops}, out out : !port {pulp.atop_filter}) {
  %cut = axi4.cut %clk, %rst_ni, %port : !port
  // expected-error @below {{cannot filter the atomics reaching this port out with a PULP axi_atop_filter, which uses a single ID width, because its write ID width (4) and read ID width (3) differ}}
  hw.output %cut : !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
// expected-error @below {{port 'axi' is marked both 'pulp.atops' and 'pulp.atop_filter'}}
hw.module.extern @BothMarkers(in %axi : !port {pulp.atops, pulp.atop_filter})

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
// expected-error @below {{port 'axi' has a 'pulp.atop_filter' that is neither a unit nor a positive i32}}
hw.module.extern @ZeroFilterBudget(in %axi : !port {pulp.atop_filter = 0 : i32})

// -----

!split_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @SplitIdRemapIds(in %clk : !seq.clock, in %rst_ni : i1,
                           in %upstream : !split_ids,
                           out downstream : !narrow_ids) {
  // expected-error @below {{'axi4.id_remap' op cannot be lowered to a PULP axi_id_remap, which uses a single ID width per side, because its upstream write ID width (4) and read ID width (3) differ}}
  %remap = axi4.id_remap %clk, %rst_ni, %upstream max_unique_ids = 4, concurrent_per_id = 4 : (!split_ids) -> !narrow_ids
  hw.output %remap : !narrow_ids
}

// -----

!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!split_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @SplitIdRemapDownstreamIds(in %clk : !seq.clock, in %rst_ni : i1,
                                     in %upstream : !wide_ids,
                                     out downstream : !split_ids) {
  // expected-error @below {{'axi4.id_remap' op cannot be lowered to a PULP axi_id_remap, which uses a single ID width per side, because its downstream write ID width (2) and read ID width (3) differ}}
  %remap = axi4.id_remap %clk, %rst_ni, %upstream max_unique_ids = 4, concurrent_per_id = 4 : (!wide_ids) -> !split_ids
  hw.output %remap : !split_ids
}

// -----

!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

hw.module @RemapTrackedIdsConfig(in %clk : !seq.clock, in %rst_ni : i1,
                                 in %upstream : !wide_ids,
                                 out downstream : !narrow_ids) {
  // expected-error @below {{cannot set PULP parameter 'AxiSlvPortMaxUniqIds'}}
  %remap = axi4.id_remap %clk, %rst_ni, %upstream max_unique_ids = 4, concurrent_per_id = 4 {PULP_CONFIG_AxiSlvPortMaxUniqIds = 2 : i32} : (!wide_ids) -> !narrow_ids
  hw.output %remap : !narrow_ids
}
