// 2 managers -> xbar -> 2 subordinates, inside a top module. mgr_module_a
// targets sub_module5_a (base 0), mgr_module_b targets sub_module5_b
// (base 4096), proving the shared xbar's routing/arbitration/id-widening
// (4 -> 5 bits) actually work, not just single-flow correctness. Each pair
// runs the same 7-phase read-write-read sequence as single.mlir's
// mgr_module/sub_module (concurrently, with distinct addresses/write data
// per pair), proving read-after-write consistency through the shared xbar
// while two independent flows are in progress at once. See
// sim/tb_axitop_multi.sv for the waveform testbench that exercises this
// end to end.
hw.module @mgr_module_a(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1, out done : i1, out beat0 : i64, out beat1 : i64, out beat2 : i64, out beat3 : i64, out after_beat0 : i64, out after_beat1 : i64, out after_beat2 : i64, out after_beat3 : i64) {
  %f = hw.constant 0 : i1
  %t = hw.constant 1 : i1
  %c2 = hw.constant 0 : i2
  %c3 = hw.constant 0 : i3
  %c4 = hw.constant 0 : i4
  %c8 = hw.constant 0 : i8
  %c32 = hw.constant 0 : i32
  %c64 = hw.constant 0 : i64
  %araddr_a = hw.constant 0 : i32     // target sub_module5_a's window (base 0)
  %arsize = hw.constant 3 : i3
  %arlen4 = hw.constant 3 : i8         // arlen = beats - 1 -> 4-beat read bursts
  %awlen2 = hw.constant 1 : i8         // awlen = beats - 1 -> 2-beat write burst
  %arburst_incr = hw.constant 1 : i2   // INCR
  %awaddr_w_a = hw.constant 8 : i32    // word 1's byte address within a's window
  %wstrb_full = hw.constant 0xFF : i8  // full-word writes only (stopgap, see README)
  %new_a_word1 = hw.constant 0xAAAAAAAA11111111 : i64
  %new_a_word2 = hw.constant 0xBBBBBBBB22222222 : i64

  %clock = seq.to_clock %clk

  // Explicit 7-phase sequencer, identical shape to single.mlir's mgr_module
  // (see that file for the full phase table / mixed_fanout.mlir lesson this
  // follows -- each transition below is gated on that phase's own icmp-eq
  // predicate, never a signal shared across phases).
  %p0 = hw.constant 0 : i3
  %p1 = hw.constant 1 : i3
  %p2 = hw.constant 2 : i3
  %p3 = hw.constant 3 : i3
  %p4 = hw.constant 4 : i3
  %p5 = hw.constant 5 : i3
  %p6 = hw.constant 6 : i3
  %phase_q = seq.compreg %phase_next, %clock reset %f, %p0 : i3

  %in_p0 = comb.icmp eq %phase_q, %p0 : i3
  %in_p1 = comb.icmp eq %phase_q, %p1 : i3
  %in_p2 = comb.icmp eq %phase_q, %p2 : i3
  %in_p3 = comb.icmp eq %phase_q, %p3 : i3
  %in_p4 = comb.icmp eq %phase_q, %p4 : i3
  %in_p5 = comb.icmp eq %phase_q, %p5 : i3
  %in_p6 = comb.icmp eq %phase_q, %p6 : i3

  %arvalid = comb.or %in_p0, %in_p5 : i1
  %ar_accept_r1 = comb.and %in_p0, %m_axi_m0_arready : i1
  %ar_accept_r2 = comb.and %in_p5, %m_axi_m0_arready : i1
  %any_ar_accept = comb.or %ar_accept_r1, %ar_accept_r2 : i1
  %r_last_beat = comb.and %m_axi_m0_rvalid, %m_axi_m0_rlast : i1

  %aw_accept = comb.and %in_p2, %m_axi_m0_awready : i1
  %w_beat_accept = comb.and %in_p3, %m_axi_m0_wready : i1
  %w_last_accept = comb.and %w_beat_accept, %w_beat_idx_q : i1
  %b_accept = comb.and %in_p4, %m_axi_m0_bvalid : i1

  %r1_advance = comb.and %in_p1, %r_last_beat : i1
  %phase_after_ar1 = comb.mux %ar_accept_r1, %p1, %phase_q : i3
  %phase_after_r1last = comb.mux %r1_advance, %p2, %phase_after_ar1 : i3
  %phase_after_aw = comb.mux %aw_accept, %p3, %phase_after_r1last : i3
  %phase_after_wlast = comb.mux %w_last_accept, %p4, %phase_after_aw : i3
  %phase_after_b = comb.mux %b_accept, %p5, %phase_after_wlast : i3
  %phase_next = comb.mux %ar_accept_r2, %p6, %phase_after_b : i3

  %r2_advance = comb.and %in_p6, %r_last_beat : i1
  %done_next = comb.mux %r2_advance, %t, %done_q : i1
  %done_q = seq.compreg %done_next, %clock reset %f, %f : i1

  // Write-beat index: 0 selects new_a_word1/not-last, 1 selects new_a_word2/last.
  %w_beat_idx_next = comb.mux %aw_accept, %f, %w_beat_idx_after : i1
  %w_beat_idx_after = comb.mux %w_beat_accept, %t, %w_beat_idx_q : i1
  %w_beat_idx_q = seq.compreg %w_beat_idx_next, %clock reset %f, %f : i1
  %wdata_sel = comb.mux %w_beat_idx_q, %new_a_word2, %new_a_word1 : i64

  // Beat counter (shared by both read bursts -- mutually exclusive in time)
  // resets on either burst's AR accept, then increments per accepted beat.
  %bc0 = hw.constant 0 : i2
  %bc1 = hw.constant 1 : i2
  %bc2 = hw.constant 2 : i2
  %bc3 = hw.constant 3 : i2
  %beat_count_after_recv = comb.mux %m_axi_m0_rvalid, %beat_count_inc, %beat_count_q : i2
  %beat_count_next = comb.mux %any_ar_accept, %bc0, %beat_count_after_recv : i2
  %beat_count_q = seq.compreg %beat_count_next, %clock reset %f, %bc0 : i2
  %beat_count_inc = comb.add %beat_count_q, %bc1 : i2

  %is_beat0 = comb.icmp eq %beat_count_q, %bc0 : i2
  %is_beat1 = comb.icmp eq %beat_count_q, %bc1 : i2
  %is_beat2 = comb.icmp eq %beat_count_q, %bc2 : i2
  %is_beat3 = comb.icmp eq %beat_count_q, %bc3 : i2
  %recv_r1 = comb.and %m_axi_m0_rvalid, %in_p1 : i1
  %recv_r2 = comb.and %m_axi_m0_rvalid, %in_p6 : i1
  %en_beat0 = comb.and %recv_r1, %is_beat0 : i1
  %en_beat1 = comb.and %recv_r1, %is_beat1 : i1
  %en_beat2 = comb.and %recv_r1, %is_beat2 : i1
  %en_beat3 = comb.and %recv_r1, %is_beat3 : i1
  %en_after0 = comb.and %recv_r2, %is_beat0 : i1
  %en_after1 = comb.and %recv_r2, %is_beat1 : i1
  %en_after2 = comb.and %recv_r2, %is_beat2 : i1
  %en_after3 = comb.and %recv_r2, %is_beat3 : i1

  %beat0_next = comb.mux %en_beat0, %m_axi_m0_rdata, %beat0_q : i64
  %beat0_q = seq.compreg %beat0_next, %clock reset %f, %c64 : i64
  %beat1_next = comb.mux %en_beat1, %m_axi_m0_rdata, %beat1_q : i64
  %beat1_q = seq.compreg %beat1_next, %clock reset %f, %c64 : i64
  %beat2_next = comb.mux %en_beat2, %m_axi_m0_rdata, %beat2_q : i64
  %beat2_q = seq.compreg %beat2_next, %clock reset %f, %c64 : i64
  %beat3_next = comb.mux %en_beat3, %m_axi_m0_rdata, %beat3_q : i64
  %beat3_q = seq.compreg %beat3_next, %clock reset %f, %c64 : i64
  %after_beat0_next = comb.mux %en_after0, %m_axi_m0_rdata, %after_beat0_q : i64
  %after_beat0_q = seq.compreg %after_beat0_next, %clock reset %f, %c64 : i64
  %after_beat1_next = comb.mux %en_after1, %m_axi_m0_rdata, %after_beat1_q : i64
  %after_beat1_q = seq.compreg %after_beat1_next, %clock reset %f, %c64 : i64
  %after_beat2_next = comb.mux %en_after2, %m_axi_m0_rdata, %after_beat2_q : i64
  %after_beat2_q = seq.compreg %after_beat2_next, %clock reset %f, %c64 : i64
  %after_beat3_next = comb.mux %en_after3, %m_axi_m0_rdata, %after_beat3_q : i64
  %after_beat3_q = seq.compreg %after_beat3_next, %clock reset %f, %c64 : i64

  hw.output %c4, %awaddr_w_a, %awlen2, %arsize, %arburst_incr, %f, %c4, %c3, %c4, %c4, %in_p2,
            %wdata_sel, %wstrb_full, %w_beat_idx_q, %in_p3,
            %in_p4,
            %c4, %araddr_a, %arlen4, %arsize, %arburst_incr, %f, %c4, %c3, %c4, %c4, %arvalid,
            %t,
            %done_q, %beat0_q, %beat1_q, %beat2_q, %beat3_q, %after_beat0_q, %after_beat1_q, %after_beat2_q, %after_beat3_q
    : i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i64, i8, i1, i1, i1,
      i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i1, i1, i64, i64, i64, i64, i64, i64, i64, i64
}
hw.module @mgr_module_b(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1, out done : i1, out beat0 : i64, out beat1 : i64, out beat2 : i64, out beat3 : i64, out after_beat0 : i64, out after_beat1 : i64, out after_beat2 : i64, out after_beat3 : i64) {
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
  %arlen4 = hw.constant 3 : i8         // arlen = beats - 1 -> 4-beat read bursts
  %awlen2 = hw.constant 1 : i8         // awlen = beats - 1 -> 2-beat write burst
  %arburst_incr = hw.constant 1 : i2   // INCR
  %awaddr_w_b = hw.constant 4104 : i32 // word 1's byte address within b's window (4096 + 8)
  %wstrb_full = hw.constant 0xFF : i8  // full-word writes only (stopgap, see README)
  %new_b_word1 = hw.constant 0xCCCCCCCC33333333 : i64
  %new_b_word2 = hw.constant 0xDDDDDDDD44444444 : i64

  %clock = seq.to_clock %clk

  // Explicit 7-phase sequencer, identical shape to single.mlir's mgr_module.
  %p0 = hw.constant 0 : i3
  %p1 = hw.constant 1 : i3
  %p2 = hw.constant 2 : i3
  %p3 = hw.constant 3 : i3
  %p4 = hw.constant 4 : i3
  %p5 = hw.constant 5 : i3
  %p6 = hw.constant 6 : i3
  %phase_q = seq.compreg %phase_next, %clock reset %f, %p0 : i3

  %in_p0 = comb.icmp eq %phase_q, %p0 : i3
  %in_p1 = comb.icmp eq %phase_q, %p1 : i3
  %in_p2 = comb.icmp eq %phase_q, %p2 : i3
  %in_p3 = comb.icmp eq %phase_q, %p3 : i3
  %in_p4 = comb.icmp eq %phase_q, %p4 : i3
  %in_p5 = comb.icmp eq %phase_q, %p5 : i3
  %in_p6 = comb.icmp eq %phase_q, %p6 : i3

  %arvalid = comb.or %in_p0, %in_p5 : i1
  %ar_accept_r1 = comb.and %in_p0, %m_axi_m0_arready : i1
  %ar_accept_r2 = comb.and %in_p5, %m_axi_m0_arready : i1
  %any_ar_accept = comb.or %ar_accept_r1, %ar_accept_r2 : i1
  %r_last_beat = comb.and %m_axi_m0_rvalid, %m_axi_m0_rlast : i1

  %aw_accept = comb.and %in_p2, %m_axi_m0_awready : i1
  %w_beat_accept = comb.and %in_p3, %m_axi_m0_wready : i1
  %w_last_accept = comb.and %w_beat_accept, %w_beat_idx_q : i1
  %b_accept = comb.and %in_p4, %m_axi_m0_bvalid : i1

  %r1_advance = comb.and %in_p1, %r_last_beat : i1
  %phase_after_ar1 = comb.mux %ar_accept_r1, %p1, %phase_q : i3
  %phase_after_r1last = comb.mux %r1_advance, %p2, %phase_after_ar1 : i3
  %phase_after_aw = comb.mux %aw_accept, %p3, %phase_after_r1last : i3
  %phase_after_wlast = comb.mux %w_last_accept, %p4, %phase_after_aw : i3
  %phase_after_b = comb.mux %b_accept, %p5, %phase_after_wlast : i3
  %phase_next = comb.mux %ar_accept_r2, %p6, %phase_after_b : i3

  %r2_advance = comb.and %in_p6, %r_last_beat : i1
  %done_next = comb.mux %r2_advance, %t, %done_q : i1
  %done_q = seq.compreg %done_next, %clock reset %f, %f : i1

  // Write-beat index: 0 selects new_b_word1/not-last, 1 selects new_b_word2/last.
  %w_beat_idx_next = comb.mux %aw_accept, %f, %w_beat_idx_after : i1
  %w_beat_idx_after = comb.mux %w_beat_accept, %t, %w_beat_idx_q : i1
  %w_beat_idx_q = seq.compreg %w_beat_idx_next, %clock reset %f, %f : i1
  %wdata_sel = comb.mux %w_beat_idx_q, %new_b_word2, %new_b_word1 : i64

  // Beat counter (shared by both read bursts -- mutually exclusive in time)
  // resets on either burst's AR accept, then increments per accepted beat.
  %bc0 = hw.constant 0 : i2
  %bc1 = hw.constant 1 : i2
  %bc2 = hw.constant 2 : i2
  %bc3 = hw.constant 3 : i2
  %beat_count_after_recv = comb.mux %m_axi_m0_rvalid, %beat_count_inc, %beat_count_q : i2
  %beat_count_next = comb.mux %any_ar_accept, %bc0, %beat_count_after_recv : i2
  %beat_count_q = seq.compreg %beat_count_next, %clock reset %f, %bc0 : i2
  %beat_count_inc = comb.add %beat_count_q, %bc1 : i2

  %is_beat0 = comb.icmp eq %beat_count_q, %bc0 : i2
  %is_beat1 = comb.icmp eq %beat_count_q, %bc1 : i2
  %is_beat2 = comb.icmp eq %beat_count_q, %bc2 : i2
  %is_beat3 = comb.icmp eq %beat_count_q, %bc3 : i2
  %recv_r1 = comb.and %m_axi_m0_rvalid, %in_p1 : i1
  %recv_r2 = comb.and %m_axi_m0_rvalid, %in_p6 : i1
  %en_beat0 = comb.and %recv_r1, %is_beat0 : i1
  %en_beat1 = comb.and %recv_r1, %is_beat1 : i1
  %en_beat2 = comb.and %recv_r1, %is_beat2 : i1
  %en_beat3 = comb.and %recv_r1, %is_beat3 : i1
  %en_after0 = comb.and %recv_r2, %is_beat0 : i1
  %en_after1 = comb.and %recv_r2, %is_beat1 : i1
  %en_after2 = comb.and %recv_r2, %is_beat2 : i1
  %en_after3 = comb.and %recv_r2, %is_beat3 : i1

  %beat0_next = comb.mux %en_beat0, %m_axi_m0_rdata, %beat0_q : i64
  %beat0_q = seq.compreg %beat0_next, %clock reset %f, %c64 : i64
  %beat1_next = comb.mux %en_beat1, %m_axi_m0_rdata, %beat1_q : i64
  %beat1_q = seq.compreg %beat1_next, %clock reset %f, %c64 : i64
  %beat2_next = comb.mux %en_beat2, %m_axi_m0_rdata, %beat2_q : i64
  %beat2_q = seq.compreg %beat2_next, %clock reset %f, %c64 : i64
  %beat3_next = comb.mux %en_beat3, %m_axi_m0_rdata, %beat3_q : i64
  %beat3_q = seq.compreg %beat3_next, %clock reset %f, %c64 : i64
  %after_beat0_next = comb.mux %en_after0, %m_axi_m0_rdata, %after_beat0_q : i64
  %after_beat0_q = seq.compreg %after_beat0_next, %clock reset %f, %c64 : i64
  %after_beat1_next = comb.mux %en_after1, %m_axi_m0_rdata, %after_beat1_q : i64
  %after_beat1_q = seq.compreg %after_beat1_next, %clock reset %f, %c64 : i64
  %after_beat2_next = comb.mux %en_after2, %m_axi_m0_rdata, %after_beat2_q : i64
  %after_beat2_q = seq.compreg %after_beat2_next, %clock reset %f, %c64 : i64
  %after_beat3_next = comb.mux %en_after3, %m_axi_m0_rdata, %after_beat3_q : i64
  %after_beat3_q = seq.compreg %after_beat3_next, %clock reset %f, %c64 : i64

  hw.output %c4, %awaddr_w_b, %awlen2, %arsize, %arburst_incr, %f, %c4, %c3, %c4, %c4, %in_p2,
            %wdata_sel, %wstrb_full, %w_beat_idx_q, %in_p3,
            %in_p4,
            %c4, %araddr_b, %arlen4, %arsize, %arburst_incr, %f, %c4, %c3, %c4, %c4, %arvalid,
            %t,
            %done_q, %beat0_q, %beat1_q, %beat2_q, %beat3_q, %after_beat0_q, %after_beat1_q, %after_beat2_q, %after_beat3_q
    : i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i64, i8, i1, i1, i1,
      i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i1, i1, i64, i64, i64, i64, i64, i64, i64, i64
}
hw.module @sub_module5_a(in %clk : i1, in %s_axi_s0_awid : i5, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i5, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i5, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i5, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1) {
  %false = hw.constant false
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c2_i2 = hw.constant 2 : i2
  %c3_i2 = hw.constant 3 : i2
  %c0_i5 = hw.constant 0 : i5
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %c0_i64 = hw.constant 0 : i64

  %clock = seq.to_clock %clk

  // Preloaded RAM contents (depth 4). seq.compreg's reset operand below is
  // never applied (reset is permanently false, see README stopgaps), so the
  // ONLY thing that actually seeds these registers' simulation-time value is
  // the `initial` clause fed by seq.initial -- not the declared reset value.
  %word0_init = seq.initial () {
    %v = hw.constant 0xCAFEF00DCAFEF00D : i64
    seq.yield %v : i64
  } : () -> !seq.immutable<i64>
  %word1_init = seq.initial () {
    %v = hw.constant 0xDEADBEEFDEADBEEF : i64
    seq.yield %v : i64
  } : () -> !seq.immutable<i64>
  %word2_init = seq.initial () {
    %v = hw.constant 0xFACEFEEDFACEFEED : i64
    seq.yield %v : i64
  } : () -> !seq.immutable<i64>
  %word3_init = seq.initial () {
    %v = hw.constant 0x8BADF00D8BADF00D : i64
    seq.yield %v : i64
  } : () -> !seq.immutable<i64>

  // ---- Read side: 2-state FSM, state_q = 0 IDLE (accept AR) / 1 BURST
  // (stream R beats) -- unchanged shape from the read-only ROM version.
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

  // RAM word index, latched from araddr at AR accept, incremented per beat.
  %idx_start = comb.extract %s_axi_s0_araddr from 3 : (i32) -> i2
  %idx_inc = comb.add %idx_q, %c1_i2 : i2
  %idx_after_beat = comb.mux %r_handshake, %idx_inc, %idx_q : i2
  %idx_next = comb.mux %accept, %idx_start, %idx_after_beat : i2
  %idx_q = seq.compreg %idx_next, %clock reset %false, %c0_i2 : i2

  %idx0 = comb.extract %idx_q from 0 : (i2) -> i1
  %idx1 = comb.extract %idx_q from 1 : (i2) -> i1
  %sel_lo = comb.mux %idx0, %word1_q, %word0_q : i64
  %sel_hi = comb.mux %idx0, %word3_q, %word2_q : i64
  %selected = comb.mux %idx1, %sel_hi, %sel_lo : i64

  // ---- Write side: new, independent 3-state FSM. Each transition below is
  // gated on that state's own icmp-eq predicate, so (per the mixed_fanout.mlir
  // lesson) they're mutually exclusive by construction and safe to chain in
  // any order.
  %ws_idle = hw.constant 0 : i2
  %ws_data = hw.constant 1 : i2
  %ws_resp = hw.constant 2 : i2
  %w_state_q = seq.compreg %w_state_next, %clock reset %false, %ws_idle : i2
  %w_in_idle = comb.icmp eq %w_state_q, %ws_idle : i2
  %w_in_data = comb.icmp eq %w_state_q, %ws_data : i2
  %w_in_resp = comb.icmp eq %w_state_q, %ws_resp : i2

  %aw_accept = comb.and %s_axi_s0_awvalid, %w_in_idle : i1
  %w_beat_accept = comb.and %s_axi_s0_wvalid, %w_in_data : i1
  %w_last_accept = comb.and %w_beat_accept, %s_axi_s0_wlast : i1
  %b_accept = comb.and %s_axi_s0_bready, %w_in_resp : i1

  %w_state_after_aw = comb.mux %aw_accept, %ws_data, %w_state_q : i2
  %w_state_after_data = comb.mux %w_last_accept, %ws_resp, %w_state_after_aw : i2
  %w_state_next = comb.mux %b_accept, %ws_idle, %w_state_after_data : i2

  %bid_next = comb.mux %aw_accept, %s_axi_s0_awid, %bid_q : i5
  %bid_q = seq.compreg %bid_next, %clock reset %false, %c0_i5 : i5

  %w_idx_start = comb.extract %s_axi_s0_awaddr from 3 : (i32) -> i2
  %w_idx_inc = comb.add %w_idx_q, %c1_i2 : i2
  %w_idx_after_beat = comb.mux %w_beat_accept, %w_idx_inc, %w_idx_q : i2
  %w_idx_next = comb.mux %aw_accept, %w_idx_start, %w_idx_after_beat : i2
  %w_idx_q = seq.compreg %w_idx_next, %clock reset %false, %c0_i2 : i2

  %w_is_word0 = comb.icmp eq %w_idx_q, %c0_i2 : i2
  %w_is_word1 = comb.icmp eq %w_idx_q, %c1_i2 : i2
  %w_is_word2 = comb.icmp eq %w_idx_q, %c2_i2 : i2
  %w_is_word3 = comb.icmp eq %w_idx_q, %c3_i2 : i2
  %we0 = comb.and %w_beat_accept, %w_is_word0 : i1
  %we1 = comb.and %w_beat_accept, %w_is_word1 : i1
  %we2 = comb.and %w_beat_accept, %w_is_word2 : i1
  %we3 = comb.and %w_beat_accept, %w_is_word3 : i1

  %word0_next = comb.mux %we0, %s_axi_s0_wdata, %word0_q : i64
  %word0_q = seq.compreg %word0_next, %clock reset %false, %c0_i64 initial %word0_init : i64
  %word1_next = comb.mux %we1, %s_axi_s0_wdata, %word1_q : i64
  %word1_q = seq.compreg %word1_next, %clock reset %false, %c0_i64 initial %word1_init : i64
  %word2_next = comb.mux %we2, %s_axi_s0_wdata, %word2_q : i64
  %word2_q = seq.compreg %word2_next, %clock reset %false, %c0_i64 initial %word2_init : i64
  %word3_next = comb.mux %we3, %s_axi_s0_wdata, %word3_q : i64
  %word3_q = seq.compreg %word3_next, %clock reset %false, %c0_i64 initial %word3_init : i64

  hw.output %w_in_idle, %w_in_data, %bid_q, %c0_i2, %w_in_resp, %idle, %rid_q, %selected, %c0_i2, %is_last_beat, %state_q
    : i1, i1, i5, i2, i1, i1, i5, i64, i2, i1, i1
}
hw.module @sub_module5_b(in %clk : i1, in %s_axi_s0_awid : i5, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i5, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i5, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i5, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1) {
  %false = hw.constant false
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c2_i2 = hw.constant 2 : i2
  %c3_i2 = hw.constant 3 : i2
  %c0_i5 = hw.constant 0 : i5
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %c0_i64 = hw.constant 0 : i64

  %clock = seq.to_clock %clk

  // Distinct ROM/initial contents from sub_module5_a's, so the waveform/
  // self-check can prove which subordinate answered which manager.
  %word0_init = seq.initial () {
    %v = hw.constant 0xFEEDFACEFEEDFACE : i64
    seq.yield %v : i64
  } : () -> !seq.immutable<i64>
  %word1_init = seq.initial () {
    %v = hw.constant 0xB105F00DB105F00D : i64
    seq.yield %v : i64
  } : () -> !seq.immutable<i64>
  %word2_init = seq.initial () {
    %v = hw.constant 0x5CA1AB1E5CA1AB1E : i64
    seq.yield %v : i64
  } : () -> !seq.immutable<i64>
  %word3_init = seq.initial () {
    %v = hw.constant 0x0DEFACED0DEFACED : i64
    seq.yield %v : i64
  } : () -> !seq.immutable<i64>

  // ---- Read side: 2-state FSM, state_q = 0 IDLE (accept AR) / 1 BURST
  // (stream R beats) -- unchanged shape from the read-only ROM version.
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
  %sel_lo = comb.mux %idx0, %word1_q, %word0_q : i64
  %sel_hi = comb.mux %idx0, %word3_q, %word2_q : i64
  %selected = comb.mux %idx1, %sel_hi, %sel_lo : i64

  // ---- Write side: new, independent 3-state FSM (see sub_module5_a's comment).
  %ws_idle = hw.constant 0 : i2
  %ws_data = hw.constant 1 : i2
  %ws_resp = hw.constant 2 : i2
  %w_state_q = seq.compreg %w_state_next, %clock reset %false, %ws_idle : i2
  %w_in_idle = comb.icmp eq %w_state_q, %ws_idle : i2
  %w_in_data = comb.icmp eq %w_state_q, %ws_data : i2
  %w_in_resp = comb.icmp eq %w_state_q, %ws_resp : i2

  %aw_accept = comb.and %s_axi_s0_awvalid, %w_in_idle : i1
  %w_beat_accept = comb.and %s_axi_s0_wvalid, %w_in_data : i1
  %w_last_accept = comb.and %w_beat_accept, %s_axi_s0_wlast : i1
  %b_accept = comb.and %s_axi_s0_bready, %w_in_resp : i1

  %w_state_after_aw = comb.mux %aw_accept, %ws_data, %w_state_q : i2
  %w_state_after_data = comb.mux %w_last_accept, %ws_resp, %w_state_after_aw : i2
  %w_state_next = comb.mux %b_accept, %ws_idle, %w_state_after_data : i2

  %bid_next = comb.mux %aw_accept, %s_axi_s0_awid, %bid_q : i5
  %bid_q = seq.compreg %bid_next, %clock reset %false, %c0_i5 : i5

  %w_idx_start = comb.extract %s_axi_s0_awaddr from 3 : (i32) -> i2
  %w_idx_inc = comb.add %w_idx_q, %c1_i2 : i2
  %w_idx_after_beat = comb.mux %w_beat_accept, %w_idx_inc, %w_idx_q : i2
  %w_idx_next = comb.mux %aw_accept, %w_idx_start, %w_idx_after_beat : i2
  %w_idx_q = seq.compreg %w_idx_next, %clock reset %false, %c0_i2 : i2

  %w_is_word0 = comb.icmp eq %w_idx_q, %c0_i2 : i2
  %w_is_word1 = comb.icmp eq %w_idx_q, %c1_i2 : i2
  %w_is_word2 = comb.icmp eq %w_idx_q, %c2_i2 : i2
  %w_is_word3 = comb.icmp eq %w_idx_q, %c3_i2 : i2
  %we0 = comb.and %w_beat_accept, %w_is_word0 : i1
  %we1 = comb.and %w_beat_accept, %w_is_word1 : i1
  %we2 = comb.and %w_beat_accept, %w_is_word2 : i1
  %we3 = comb.and %w_beat_accept, %w_is_word3 : i1

  %word0_next = comb.mux %we0, %s_axi_s0_wdata, %word0_q : i64
  %word0_q = seq.compreg %word0_next, %clock reset %false, %c0_i64 initial %word0_init : i64
  %word1_next = comb.mux %we1, %s_axi_s0_wdata, %word1_q : i64
  %word1_q = seq.compreg %word1_next, %clock reset %false, %c0_i64 initial %word1_init : i64
  %word2_next = comb.mux %we2, %s_axi_s0_wdata, %word2_q : i64
  %word2_q = seq.compreg %word2_next, %clock reset %false, %c0_i64 initial %word2_init : i64
  %word3_next = comb.mux %we3, %s_axi_s0_wdata, %word3_q : i64
  %word3_q = seq.compreg %word3_next, %clock reset %false, %c0_i64 initial %word3_init : i64

  hw.output %w_in_idle, %w_in_data, %bid_q, %c0_i2, %w_in_resp, %idle, %rid_q, %selected, %c0_i2, %is_last_beat, %state_q
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
    access = [#axi4.window<base = 0, size = 8192, burst_specs = [<incr, len = 4>, <incr, len = 2>]>],
    outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
  } : !axi4.port<32, 64, 4>
  %mgr2 = axi4.manager_port %clk node %mnode2 {
    port_mapping = #axi4.port_wires<"clk", "m0">,
    access = [#axi4.window<base = 0, size = 8192, burst_specs = [<incr, len = 4>, <incr, len = 2>]>],
    outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
  } : !axi4.port<32, 64, 4>
  %xbar = axi4.xbar %clk mgrs %mgr1, %mgr2 : (!axi4.port<32, 64, 4>, !axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 5>
  axi4.subordinate_port %xbar, %clk node %snode1 {
    port_mapping = #axi4.port_wires<"clk", "s0">,
    access = [#axi4.window<base = 0, size = 4096, burst_specs = [<incr, len = 4>, <incr, len = 2>]>],
    outstanding_requests = 4 : ui32
  } : !axi4.port<32, 64, 5>
  axi4.subordinate_port %xbar, %clk node %snode2 {
    port_mapping = #axi4.port_wires<"clk", "s0">,
    access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<incr, len = 4>, <incr, len = 2>]>],
    outstanding_requests = 4 : ui32
  } : !axi4.port<32, 64, 5>
  hw.output
}
