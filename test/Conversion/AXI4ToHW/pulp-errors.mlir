// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true --split-input-file --verify-diagnostics

// Every wrapper types its ID fields through a typedef, so a port with no ID
// bits has nothing to declare them from
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 0, read_id_width = 0, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 1, outstanding_reads = 1>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 0, read_id_width = 0, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 1, outstanding_reads = 1>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !mgr)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !sub)

hw.module @NoIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op cannot be lowered to PULP because its upstream port #0 has a zero-width write ID, which PULP cannot express}}
  %s = axi4.xbar %clk, %rst_ni mgrs %m : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

// PULP's axi_xbar has one ID width per side, shared by writes and reads
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !mgr)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !sub)

hw.module @SplitUpstreamIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op cannot be lowered to a PULP axi_xbar, which uses a single ID width per side, because its upstream write ID width (4) and read ID width (3) differ}}
  %s = axi4.xbar %clk, %rst_ni mgrs %m : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

// One ID width is shared by every downstream port too, though the xbar
// verifier lets them widen independently
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub_lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0x7ff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub_hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 6, read_id_width = 6, user_width = 0, windows = <<base = 0x800, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !mgr)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Low(in %axi : !sub_lo)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @High(in %axi : !sub_hi)

hw.module @MixedDownstreamIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op cannot be lowered to a PULP axi_xbar, which uses one ID width for every downstream port, because downstream port #1's ID width (6) differs from downstream port #0's (5)}}
  %lo, %hi = axi4.xbar %clk, %rst_ni mgrs %m : (!mgr) -> (!sub_lo, !sub_hi)
  hw.instance "lo" @Low(axi: %lo: !sub_lo) -> ()
  hw.instance "hi" @High(axi: %hi: !sub_hi) -> ()
}

// -----

// PULP widens the ID by exactly the bits it needs to tag its managers, so a
// downstream port wider than that is not expressible, though the xbar verifier
// allows it
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !mgr)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !sub)

hw.module @OverWideForOneManager(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op cannot be lowered to a PULP axi_xbar, which widens IDs by exactly the 0 bits needed to tag 1 manager, so its downstream ID width must be 4, not 5}}
  %s = axi4.xbar %clk, %rst_ni mgrs %m : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

// Two managers need one tag bit, so the downstream width must be exactly one
// wider - no more
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 6, read_id_width = 6, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !mgr)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !sub)

hw.module @OverWideForTwoManagers(in %clk : !seq.clock, in %rst_ni : i1) {
  %a = hw.instance "mgr_a" @Manager() -> (axi: !mgr)
  %b = hw.instance "mgr_b" @Manager() -> (axi: !mgr)
  // expected-error @below {{'axi4.xbar' op cannot be lowered to a PULP axi_xbar, which widens IDs by exactly the 1 bits needed to tag 2 managers, so its downstream ID width must be 5, not 6}}
  %s = axi4.xbar %clk, %rst_ni mgrs %a, %b : (!mgr, !mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

// PULP's axi_dw_converter converts over a single ID width, shared by writes and
// reads, so the two must agree - unlike a cut, which never inspects them
!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>

// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Manager(out axi : !wide)
// expected-warning @below {{lowering AXI4 port 'axi' changes the ports of this module; its implementation must match the new port list}}
hw.module.extern @Subordinate(in %axi : !thin)

hw.module @SplitIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !wide)
  // expected-error @below {{'axi4.data_width_converter' op cannot be lowered to a PULP axi_dw_converter, which uses a single ID width, because its write ID width (4) and read ID width (2) differ}}
  %dwc = axi4.data_width_converter %clk, %rst_ni, %m : (!wide) -> !thin
  hw.instance "sub" @Subordinate(axi: %dwc: !thin) -> ()
}
