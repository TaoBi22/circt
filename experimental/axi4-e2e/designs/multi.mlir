// 2 managers -> xbar -> 2 subordinates, inside a top module. mgr_module_a
// targets sub_module5_a (base 0), mgr_module_b targets sub_module5_b
// (base 4096), proving the shared xbar's routing/arbitration/id-widening
// (4 -> 5 bits) actually work, not just single-flow correctness. See
// sim/tb_axitop_multi.sv for the waveform testbench that exercises this
// end to end.
hw.module @mgr_module_a(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1, out done : i1, out beat0 : i64, out beat1 : i64, out beat2 : i64, out beat3 : i64) {
  %f = hw.constant 0 : i1
  %t = hw.constant 1 : i1
  %c2 = hw.constant 0 : i2
  %c3 = hw.constant 0 : i3
  %c4 = hw.constant 0 : i4
  %c8 = hw.constant 0 : i8
  %c32 = hw.constant 0 : i32
  %c64 = hw.constant 0 : i64
  %araddr_a = hw.constant 0 : i32   // target sub_module5_a's window (base 0)
  %arsize = hw.constant 3 : i3
  %arlen4 = hw.constant 3 : i8        // arlen = beats - 1 -> 4-beat burst
  %arburst_incr = hw.constant 1 : i2  // INCR

  %clock = seq.to_clock %clk

  // Single-shot AR issue: arvalid until accepted, then never again.
  %state_q = seq.compreg %state_next, %clock reset %f, %f : i1
  %issuing = comb.xor %state_q, %t : i1
  %ar_accept = comb.and %issuing, %m_axi_m0_arready : i1
  %state_next = comb.mux %ar_accept, %t, %state_q : i1

  // Accept every beat of the burst response (rready held high throughout);
  // done fires once the final (rlast) beat is accepted.
  %r_last_beat = comb.and %m_axi_m0_rvalid, %m_axi_m0_rlast : i1
  %done_next = comb.mux %r_last_beat, %t, %done_q : i1
  %done_q = seq.compreg %done_next, %clock reset %f, %f : i1

  // Beat counter selects which beatN register captures the incoming beat.
  %bc0 = hw.constant 0 : i2
  %bc1 = hw.constant 1 : i2
  %bc2 = hw.constant 2 : i2
  %bc3 = hw.constant 3 : i2
  %beat_count_next = comb.mux %m_axi_m0_rvalid, %beat_count_inc, %beat_count_q : i2
  %beat_count_q = seq.compreg %beat_count_next, %clock reset %f, %bc0 : i2
  %beat_count_inc = comb.add %beat_count_q, %bc1 : i2

  %is_beat0 = comb.icmp eq %beat_count_q, %bc0 : i2
  %is_beat1 = comb.icmp eq %beat_count_q, %bc1 : i2
  %is_beat2 = comb.icmp eq %beat_count_q, %bc2 : i2
  %is_beat3 = comb.icmp eq %beat_count_q, %bc3 : i2
  %en_beat0 = comb.and %m_axi_m0_rvalid, %is_beat0 : i1
  %en_beat1 = comb.and %m_axi_m0_rvalid, %is_beat1 : i1
  %en_beat2 = comb.and %m_axi_m0_rvalid, %is_beat2 : i1
  %en_beat3 = comb.and %m_axi_m0_rvalid, %is_beat3 : i1

  %beat0_next = comb.mux %en_beat0, %m_axi_m0_rdata, %beat0_q : i64
  %beat0_q = seq.compreg %beat0_next, %clock reset %f, %c64 : i64
  %beat1_next = comb.mux %en_beat1, %m_axi_m0_rdata, %beat1_q : i64
  %beat1_q = seq.compreg %beat1_next, %clock reset %f, %c64 : i64
  %beat2_next = comb.mux %en_beat2, %m_axi_m0_rdata, %beat2_q : i64
  %beat2_q = seq.compreg %beat2_next, %clock reset %f, %c64 : i64
  %beat3_next = comb.mux %en_beat3, %m_axi_m0_rdata, %beat3_q : i64
  %beat3_q = seq.compreg %beat3_next, %clock reset %f, %c64 : i64

  hw.output %c4, %araddr_a, %c8, %c3, %c2, %f, %c4, %c3, %c4, %c4, %f,
            %c64, %c8, %f, %f,
            %f,
            %c4, %araddr_a, %arlen4, %arsize, %arburst_incr, %f, %c4, %c3, %c4, %c4, %issuing,
            %t,
            %done_q, %beat0_q, %beat1_q, %beat2_q, %beat3_q
    : i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i64, i8, i1, i1, i1,
      i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i1, i1, i64, i64, i64, i64
}
hw.module @mgr_module_b(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1, out done : i1, out beat0 : i64, out beat1 : i64, out beat2 : i64, out beat3 : i64) {
  %f = hw.constant 0 : i1
  %t = hw.constant 1 : i1
  %c2 = hw.constant 0 : i2
  %c3 = hw.constant 0 : i3
  %c4 = hw.constant 0 : i4
  %c8 = hw.constant 0 : i8
  %c32 = hw.constant 0 : i32
  %c64 = hw.constant 0 : i64
  %araddr_b = hw.constant 4096 : i32   // target sub_module5_b's window (base 4096)
  %arsize = hw.constant 3 : i3
  %arlen4 = hw.constant 3 : i8        // arlen = beats - 1 -> 4-beat burst
  %arburst_incr = hw.constant 1 : i2  // INCR

  %clock = seq.to_clock %clk

  %state_q = seq.compreg %state_next, %clock reset %f, %f : i1
  %issuing = comb.xor %state_q, %t : i1
  %ar_accept = comb.and %issuing, %m_axi_m0_arready : i1
  %state_next = comb.mux %ar_accept, %t, %state_q : i1

  %r_last_beat = comb.and %m_axi_m0_rvalid, %m_axi_m0_rlast : i1
  %done_next = comb.mux %r_last_beat, %t, %done_q : i1
  %done_q = seq.compreg %done_next, %clock reset %f, %f : i1

  %bc0 = hw.constant 0 : i2
  %bc1 = hw.constant 1 : i2
  %bc2 = hw.constant 2 : i2
  %bc3 = hw.constant 3 : i2
  %beat_count_next = comb.mux %m_axi_m0_rvalid, %beat_count_inc, %beat_count_q : i2
  %beat_count_q = seq.compreg %beat_count_next, %clock reset %f, %bc0 : i2
  %beat_count_inc = comb.add %beat_count_q, %bc1 : i2

  %is_beat0 = comb.icmp eq %beat_count_q, %bc0 : i2
  %is_beat1 = comb.icmp eq %beat_count_q, %bc1 : i2
  %is_beat2 = comb.icmp eq %beat_count_q, %bc2 : i2
  %is_beat3 = comb.icmp eq %beat_count_q, %bc3 : i2
  %en_beat0 = comb.and %m_axi_m0_rvalid, %is_beat0 : i1
  %en_beat1 = comb.and %m_axi_m0_rvalid, %is_beat1 : i1
  %en_beat2 = comb.and %m_axi_m0_rvalid, %is_beat2 : i1
  %en_beat3 = comb.and %m_axi_m0_rvalid, %is_beat3 : i1

  %beat0_next = comb.mux %en_beat0, %m_axi_m0_rdata, %beat0_q : i64
  %beat0_q = seq.compreg %beat0_next, %clock reset %f, %c64 : i64
  %beat1_next = comb.mux %en_beat1, %m_axi_m0_rdata, %beat1_q : i64
  %beat1_q = seq.compreg %beat1_next, %clock reset %f, %c64 : i64
  %beat2_next = comb.mux %en_beat2, %m_axi_m0_rdata, %beat2_q : i64
  %beat2_q = seq.compreg %beat2_next, %clock reset %f, %c64 : i64
  %beat3_next = comb.mux %en_beat3, %m_axi_m0_rdata, %beat3_q : i64
  %beat3_q = seq.compreg %beat3_next, %clock reset %f, %c64 : i64

  hw.output %c4, %araddr_b, %c8, %c3, %c2, %f, %c4, %c3, %c4, %c4, %f,
            %c64, %c8, %f, %f,
            %f,
            %c4, %araddr_b, %arlen4, %arsize, %arburst_incr, %f, %c4, %c3, %c4, %c4, %issuing,
            %t,
            %done_q, %beat0_q, %beat1_q, %beat2_q, %beat3_q
    : i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i64, i8, i1, i1, i1,
      i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i1, i1, i64, i64, i64, i64
}
hw.module @sub_module5_a(in %clk : i1, in %s_axi_s0_awid : i5, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i5, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i5, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i5, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1) {
  %false = hw.constant false
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c0_i5 = hw.constant 0 : i5
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

  %rid_next = comb.mux %accept, %s_axi_s0_arid, %rid_q : i5
  %rid_q = seq.compreg %rid_next, %clock reset %false, %c0_i5 : i5

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

  hw.output %false, %false, %c0_i5, %c0_i2, %false, %idle, %rid_q, %selected, %c0_i2, %is_last_beat, %state_q
    : i1, i1, i5, i2, i1, i1, i5, i64, i2, i1, i1
}
hw.module @sub_module5_b(in %clk : i1, in %s_axi_s0_awid : i5, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i5, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i5, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i5, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1) {
  %false = hw.constant false
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c0_i5 = hw.constant 0 : i5
  %c0_i8 = hw.constant 0 : i8
  %c1_i2 = hw.constant 1 : i2
  %c1_i8 = hw.constant 1 : i8
  %c0_i64 = hw.constant 0 : i64

  %clock = seq.to_clock %clk

  // Distinct ROM contents from sub_module5_a's, so the waveform/self-check
  // can prove which subordinate answered which manager.
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

  %rid_next = comb.mux %accept, %s_axi_s0_arid, %rid_q : i5
  %rid_q = seq.compreg %rid_next, %clock reset %false, %c0_i5 : i5

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

  hw.output %false, %false, %c0_i5, %c0_i2, %false, %idle, %rid_q, %selected, %c0_i2, %is_last_beat, %state_q
    : i1, i1, i5, i2, i1, i1, i5, i64, i2, i1, i1
}

hw.module @AXITop(in %clk_i : i1) {
  %clk = builtin.unrealized_conversion_cast %clk_i : i1 to !axi4.clock
  %mnode1 = axi4.node @mgr_module_a : !axi4.node
  %mnode2 = axi4.node @mgr_module_b : !axi4.node
  %snode1 = axi4.node @sub_module5_a : !axi4.node
  %snode2 = axi4.node @sub_module5_b : !axi4.node
  %mgr1 = axi4.manager_port %clk node %mnode1 {
    port_mapping = #axi4.port_wires<"clk", "m0">,
    access = [#axi4.window<base = 0, size = 8192, burst_specs = [<incr, len = 4>]>],
    outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
  } : !axi4.port<32, 64, 4>
  %mgr2 = axi4.manager_port %clk node %mnode2 {
    port_mapping = #axi4.port_wires<"clk", "m0">,
    access = [#axi4.window<base = 0, size = 8192, burst_specs = [<incr, len = 4>]>],
    outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
  } : !axi4.port<32, 64, 4>
  %xbar = axi4.xbar %clk mgrs %mgr1, %mgr2 : (!axi4.port<32, 64, 4>, !axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 5>
  axi4.subordinate_port %xbar, %clk node %snode1 {
    port_mapping = #axi4.port_wires<"clk", "s0">,
    access = [#axi4.window<base = 0, size = 4096, burst_specs = [<incr, len = 4>]>],
    outstanding_requests = 4 : ui32
  } : !axi4.port<32, 64, 5>
  axi4.subordinate_port %xbar, %clk node %snode2 {
    port_mapping = #axi4.port_wires<"clk", "s0">,
    access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<incr, len = 4>]>],
    outstanding_requests = 4 : ui32
  } : !axi4.port<32, 64, 5>
  hw.output
}
