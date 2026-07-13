// Mixed fan-out inside a top module: one crossbar feeds a subordinate AND a
// chained crossbar. Emits AXITop + two wrappers: axi_xbar_1u2d (the fan-out)
// and axi_xbar_1u1d (the leaf). mgr_module issues two sequential 4-beat AXI4
// INCR bursts: first to address 0 (sub_module_a, direct via xbar1), then to
// address 4096 (sub_module_b, via the chained xbar2), proving both fan-out
// paths actually work. See sim/tb_axitop_mixed_fanout.sv for the waveform
// testbench that exercises this end to end.
hw.module @mgr_module(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1, out done : i1, out a_beat0 : i64, out a_beat1 : i64, out a_beat2 : i64, out a_beat3 : i64, out b_beat0 : i64, out b_beat1 : i64, out b_beat2 : i64, out b_beat3 : i64) {
  %f = hw.constant 0 : i1
  %t = hw.constant 1 : i1
  %c2 = hw.constant 0 : i2
  %c3 = hw.constant 0 : i3
  %c4 = hw.constant 0 : i4
  %c8 = hw.constant 0 : i8
  %c32 = hw.constant 0 : i32
  %c64 = hw.constant 0 : i64
  %araddr_a = hw.constant 0 : i32     // sub_module_a's window (base 0, direct via xbar1)
  %araddr_b = hw.constant 4096 : i32  // sub_module_b's window (base 4096, chained via xbar2)
  %arsize = hw.constant 3 : i3
  %arlen4 = hw.constant 3 : i8        // arlen = beats - 1 -> 4-beat burst
  %arburst_incr = hw.constant 1 : i2  // INCR

  %clock = seq.to_clock %clk

  // Explicit 4-state sequencer:
  //   0 ISSUE_A: arvalid=1, araddr=0.    AR accept  -> 1
  //   1 RECV_A:  rready=1, capture a_beatN.  rlast accept -> 2
  //   2 ISSUE_B: arvalid=1, araddr=4096. AR accept  -> 3
  //   3 RECV_B:  rready=1, capture b_beatN.  rlast accept -> done=1, stays at 3
  %p0 = hw.constant 0 : i2
  %p1 = hw.constant 1 : i2
  %p2 = hw.constant 2 : i2
  %p3 = hw.constant 3 : i2
  %phase_q = seq.compreg %phase_next, %clock reset %f, %p0 : i2

  %in_issue_a = comb.icmp eq %phase_q, %p0 : i2
  %in_recv_a = comb.icmp eq %phase_q, %p1 : i2
  %in_issue_b = comb.icmp eq %phase_q, %p2 : i2
  %in_recv_b = comb.icmp eq %phase_q, %p3 : i2

  %issuing = comb.or %in_issue_a, %in_issue_b : i1
  %araddr_sel = comb.mux %in_issue_b, %araddr_b, %araddr_a : i32
  %ar_accept = comb.and %issuing, %m_axi_m0_arready : i1
  %ar_accept_a = comb.and %in_issue_a, %m_axi_m0_arready : i1
  %ar_accept_b = comb.and %in_issue_b, %m_axi_m0_arready : i1
  %r_last_beat = comb.and %m_axi_m0_rvalid, %m_axi_m0_rlast : i1
  %advance_from_a = comb.and %in_recv_a, %r_last_beat : i1
  %advance_from_b = comb.and %in_recv_b, %r_last_beat : i1

  // Each condition below is mutually exclusive with the others by
  // construction (only one is true per cycle, since each is gated on a
  // distinct phase), so the mux chain composes independent transitions
  // rather than encoding a priority order.
  %phase_after_ar_a = comb.mux %ar_accept_a, %p1, %phase_q : i2
  %phase_after_a = comb.mux %advance_from_a, %p2, %phase_after_ar_a : i2
  %phase_after_ar_b = comb.mux %ar_accept_b, %p3, %phase_after_a : i2
  %phase_next = comb.mux %advance_from_b, %p3, %phase_after_ar_b : i2

  %done_next = comb.mux %advance_from_b, %t, %done_q : i1
  %done_q = seq.compreg %done_next, %clock reset %f, %f : i1

  // Beat counter resets whenever a fresh AR is accepted, then increments per beat.
  %bc0 = hw.constant 0 : i2
  %bc1 = hw.constant 1 : i2
  %bc2 = hw.constant 2 : i2
  %bc3 = hw.constant 3 : i2
  %beat_count_after_recv = comb.mux %m_axi_m0_rvalid, %beat_count_inc, %beat_count_q : i2
  %beat_count_next = comb.mux %ar_accept, %bc0, %beat_count_after_recv : i2
  %beat_count_q = seq.compreg %beat_count_next, %clock reset %f, %bc0 : i2
  %beat_count_inc = comb.add %beat_count_q, %bc1 : i2

  %is_beat0 = comb.icmp eq %beat_count_q, %bc0 : i2
  %is_beat1 = comb.icmp eq %beat_count_q, %bc1 : i2
  %is_beat2 = comb.icmp eq %beat_count_q, %bc2 : i2
  %is_beat3 = comb.icmp eq %beat_count_q, %bc3 : i2
  %en_a = comb.and %m_axi_m0_rvalid, %in_recv_a : i1
  %en_b = comb.and %m_axi_m0_rvalid, %in_recv_b : i1
  %en_a_beat0 = comb.and %en_a, %is_beat0 : i1
  %en_a_beat1 = comb.and %en_a, %is_beat1 : i1
  %en_a_beat2 = comb.and %en_a, %is_beat2 : i1
  %en_a_beat3 = comb.and %en_a, %is_beat3 : i1
  %en_b_beat0 = comb.and %en_b, %is_beat0 : i1
  %en_b_beat1 = comb.and %en_b, %is_beat1 : i1
  %en_b_beat2 = comb.and %en_b, %is_beat2 : i1
  %en_b_beat3 = comb.and %en_b, %is_beat3 : i1

  %a_beat0_next = comb.mux %en_a_beat0, %m_axi_m0_rdata, %a_beat0_q : i64
  %a_beat0_q = seq.compreg %a_beat0_next, %clock reset %f, %c64 : i64
  %a_beat1_next = comb.mux %en_a_beat1, %m_axi_m0_rdata, %a_beat1_q : i64
  %a_beat1_q = seq.compreg %a_beat1_next, %clock reset %f, %c64 : i64
  %a_beat2_next = comb.mux %en_a_beat2, %m_axi_m0_rdata, %a_beat2_q : i64
  %a_beat2_q = seq.compreg %a_beat2_next, %clock reset %f, %c64 : i64
  %a_beat3_next = comb.mux %en_a_beat3, %m_axi_m0_rdata, %a_beat3_q : i64
  %a_beat3_q = seq.compreg %a_beat3_next, %clock reset %f, %c64 : i64
  %b_beat0_next = comb.mux %en_b_beat0, %m_axi_m0_rdata, %b_beat0_q : i64
  %b_beat0_q = seq.compreg %b_beat0_next, %clock reset %f, %c64 : i64
  %b_beat1_next = comb.mux %en_b_beat1, %m_axi_m0_rdata, %b_beat1_q : i64
  %b_beat1_q = seq.compreg %b_beat1_next, %clock reset %f, %c64 : i64
  %b_beat2_next = comb.mux %en_b_beat2, %m_axi_m0_rdata, %b_beat2_q : i64
  %b_beat2_q = seq.compreg %b_beat2_next, %clock reset %f, %c64 : i64
  %b_beat3_next = comb.mux %en_b_beat3, %m_axi_m0_rdata, %b_beat3_q : i64
  %b_beat3_q = seq.compreg %b_beat3_next, %clock reset %f, %c64 : i64

  hw.output %c4, %araddr_sel, %c8, %c3, %c2, %f, %c4, %c3, %c4, %c4, %f,
            %c64, %c8, %f, %f,
            %f,
            %c4, %araddr_sel, %arlen4, %arsize, %arburst_incr, %f, %c4, %c3, %c4, %c4, %issuing,
            %t,
            %done_q, %a_beat0_q, %a_beat1_q, %a_beat2_q, %a_beat3_q, %b_beat0_q, %b_beat1_q, %b_beat2_q, %b_beat3_q
    : i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i64, i8, i1, i1, i1,
      i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i1, i1, i64, i64, i64, i64, i64, i64, i64, i64
}
hw.module @sub_module_a(in %clk : i1, in %s_axi_s0_awid : i4, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i4, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i4, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i4, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1) {
  %false = hw.constant false
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c0_i4 = hw.constant 0 : i4
  %c0_i8 = hw.constant 0 : i8
  %c1_i2 = hw.constant 1 : i2
  %c1_i8 = hw.constant 1 : i8
  %c0_i64 = hw.constant 0 : i64

  %clock = seq.to_clock %clk

  // ROM contents (depth 4, walked in order across a burst); the manager
  // issues a 4-beat INCR burst starting at word 0, so all four are fetched.
  %rom0 = hw.constant 0xCAFEF00DCAFEF00D : i64
  %rom1 = hw.constant 0xDEADBEEFDEADBEEF : i64
  %rom2 = hw.constant 0xFACEFEEDFACEFEED : i64
  %rom3 = hw.constant 0x8BADF00D8BADF00D : i64

  // 2-state FSM: state_q = 0 IDLE (accept AR) / 1 BURST (stream R beats).
  %state_q = seq.compreg %state_next, %clock reset %false, %false : i1
  %idle = comb.xor %state_q, %true : i1
  %accept = comb.and %s_axi_s0_arvalid, %idle : i1
  %r_handshake = comb.and %state_q, %s_axi_s0_rready : i1
  %is_last_beat = comb.icmp eq %remaining_q, %c0_i8 : i8
  %burst_done = comb.and %r_handshake, %is_last_beat : i1
  %back_to_idle = comb.mux %burst_done, %false, %state_q : i1
  %state_next = comb.mux %accept, %true, %back_to_idle : i1

  %rid_next = comb.mux %accept, %s_axi_s0_arid, %rid_q : i4
  %rid_q = seq.compreg %rid_next, %clock reset %false, %c0_i4 : i4

  // Beats remaining, latched from arlen at AR accept, decremented per beat.
  %remaining_dec = comb.sub %remaining_q, %c1_i8 : i8
  %remaining_after_beat = comb.mux %r_handshake, %remaining_dec, %remaining_q : i8
  %remaining_next = comb.mux %accept, %s_axi_s0_arlen, %remaining_after_beat : i8
  %remaining_q = seq.compreg %remaining_next, %clock reset %false, %c0_i8 : i8

  // ROM word index, latched from araddr at AR accept, incremented per beat.
  %idx_start = comb.extract %s_axi_s0_araddr from 3 : (i32) -> i2
  %idx_inc = comb.add %idx_q, %c1_i2 : i2
  %idx_after_beat = comb.mux %r_handshake, %idx_inc, %idx_q : i2
  %idx_next = comb.mux %accept, %idx_start, %idx_after_beat : i2
  %idx_q = seq.compreg %idx_next, %clock reset %false, %c0_i2 : i2

  %idx0 = comb.extract %idx_q from 0 : (i2) -> i1
  %idx1 = comb.extract %idx_q from 1 : (i2) -> i1
  %sel_lo = comb.mux %idx0, %rom1, %rom0 : i64
  %sel_hi = comb.mux %idx0, %rom3, %rom2 : i64
  %selected = comb.mux %idx1, %sel_hi, %sel_lo : i64

  hw.output %false, %false, %c0_i4, %c0_i2, %false, %idle, %rid_q, %selected, %c0_i2, %is_last_beat, %state_q
    : i1, i1, i4, i2, i1, i1, i4, i64, i2, i1, i1
}
hw.module @sub_module_b(in %clk : i1, in %s_axi_s0_awid : i4, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i4, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i4, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i4, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1) {
  %false = hw.constant false
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c0_i4 = hw.constant 0 : i4
  %c0_i8 = hw.constant 0 : i8
  %c1_i2 = hw.constant 1 : i2
  %c1_i8 = hw.constant 1 : i8
  %c0_i64 = hw.constant 0 : i64

  %clock = seq.to_clock %clk

  // Distinct ROM contents from sub_module_a's, so the waveform/self-check
  // can prove which subordinate answered which burst.
  %rom0 = hw.constant 0xFEEDFACEFEEDFACE : i64
  %rom1 = hw.constant 0xB105F00DB105F00D : i64
  %rom2 = hw.constant 0x5CA1AB1E5CA1AB1E : i64
  %rom3 = hw.constant 0x0DEFACED0DEFACED : i64

  %state_q = seq.compreg %state_next, %clock reset %false, %false : i1
  %idle = comb.xor %state_q, %true : i1
  %accept = comb.and %s_axi_s0_arvalid, %idle : i1
  %r_handshake = comb.and %state_q, %s_axi_s0_rready : i1
  %is_last_beat = comb.icmp eq %remaining_q, %c0_i8 : i8
  %burst_done = comb.and %r_handshake, %is_last_beat : i1
  %back_to_idle = comb.mux %burst_done, %false, %state_q : i1
  %state_next = comb.mux %accept, %true, %back_to_idle : i1

  %rid_next = comb.mux %accept, %s_axi_s0_arid, %rid_q : i4
  %rid_q = seq.compreg %rid_next, %clock reset %false, %c0_i4 : i4

  %remaining_dec = comb.sub %remaining_q, %c1_i8 : i8
  %remaining_after_beat = comb.mux %r_handshake, %remaining_dec, %remaining_q : i8
  %remaining_next = comb.mux %accept, %s_axi_s0_arlen, %remaining_after_beat : i8
  %remaining_q = seq.compreg %remaining_next, %clock reset %false, %c0_i8 : i8

  %idx_start = comb.extract %s_axi_s0_araddr from 3 : (i32) -> i2
  %idx_inc = comb.add %idx_q, %c1_i2 : i2
  %idx_after_beat = comb.mux %r_handshake, %idx_inc, %idx_q : i2
  %idx_next = comb.mux %accept, %idx_start, %idx_after_beat : i2
  %idx_q = seq.compreg %idx_next, %clock reset %false, %c0_i2 : i2

  %idx0 = comb.extract %idx_q from 0 : (i2) -> i1
  %idx1 = comb.extract %idx_q from 1 : (i2) -> i1
  %sel_lo = comb.mux %idx0, %rom1, %rom0 : i64
  %sel_hi = comb.mux %idx0, %rom3, %rom2 : i64
  %selected = comb.mux %idx1, %sel_hi, %sel_lo : i64

  hw.output %false, %false, %c0_i4, %c0_i2, %false, %idle, %rid_q, %selected, %c0_i2, %is_last_beat, %state_q
    : i1, i1, i4, i2, i1, i1, i4, i64, i2, i1, i1
}

hw.module @AXITop(in %clk_i : i1) {
  %clk = builtin.unrealized_conversion_cast %clk_i : i1 to !axi4.clock
  %mnode = axi4.node @mgr_module : !axi4.node
  %snode_a = axi4.node @sub_module_a : !axi4.node
  %snode_b = axi4.node @sub_module_b : !axi4.node
  %mgr = axi4.manager_port %clk node %mnode {
    port_mapping = #axi4.port_wires<"clk", "m0">,
    access = [#axi4.window<base = 0, size = 8192, burst_specs = [<incr, len = 4>]>],
    outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
  } : !axi4.port<32, 64, 4>
  %xbar1 = axi4.xbar %clk mgrs %mgr : (!axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 4>
  %xbar2 = axi4.xbar %clk mgrs %xbar1 : (!axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 4>
  // %xbar1 fans out to both a direct subordinate and the chained %xbar2.
  axi4.subordinate_port %xbar1, %clk node %snode_a {
    port_mapping = #axi4.port_wires<"clk", "s0">,
    access = [#axi4.window<base = 0, size = 4096, burst_specs = [<incr, len = 4>]>],
    outstanding_requests = 4 : ui32
  } : !axi4.port<32, 64, 4>
  axi4.subordinate_port %xbar2, %clk node %snode_b {
    port_mapping = #axi4.port_wires<"clk", "s0">,
    access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<incr, len = 4>]>],
    outstanding_requests = 4 : ui32
  } : !axi4.port<32, 64, 4>
  hw.output
}
