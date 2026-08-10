// Mixed fan-out inside a top module: one crossbar feeds a subordinate AND a
// chained crossbar. Emits AXITop + two wrappers: axi_xbar_1u2d (the fan-out)
// and axi_xbar_1u1d (the leaf). mgr_module runs the same 7-phase
// read-write-read sequence used by single.mlir/multi.mlir, first against
// sub_module_a (address 0, direct via xbar1), then against sub_module_b
// (address 4096, chained via xbar2) -- a 4-beat AXI4 INCR read burst, a
// 2-beat INCR write burst overwriting words 1 and 2, then a second 4-beat
// read burst -- proving both fan-out paths support read-after-write
// consistency, not just single-flow correctness. See
// sim/tb_axitop_mixed_fanout.sv for the waveform testbench that exercises
// this end to end.
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 1, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<incr, len = 2>, <incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
// Both crossbars have a single upstream manager, so neither widens the ID.
!sub_a = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 1, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 2>, <incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!mid = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 1, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 2>, <incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub_b = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 1, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 2>, <incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

!aw = !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, atop: i6, user: i1>
!w = !hw.struct<data: i64, strb: i8, last: i1, user: i1>
!b = !hw.struct<id: i4, resp: i2, user: i1>
!ar = !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i1>
!r = !hw.struct<id: i4, data: i64, resp: i2, last: i1, user: i1>

hw.module @mgr_module(in %clk_i : !seq.clock, in %rst_ni : i1, out axi : !mgr,
                      out done : i1,
                      out a_beat0 : i64, out a_beat1 : i64, out a_beat2 : i64,
                      out a_beat3 : i64, out after_a_beat0 : i64,
                      out after_a_beat1 : i64, out after_a_beat2 : i64,
                      out after_a_beat3 : i64,
                      out b_beat0 : i64, out b_beat1 : i64, out b_beat2 : i64,
                      out b_beat3 : i64, out after_b_beat0 : i64,
                      out after_b_beat1 : i64, out after_b_beat2 : i64,
                      out after_b_beat3 : i64) {
  %f = hw.constant 0 : i1
  %t = hw.constant 1 : i1
  %c2 = hw.constant 0 : i2
  %c3 = hw.constant 0 : i3
  %c4 = hw.constant 0 : i4
  %c8 = hw.constant 0 : i8
  %atop0 = hw.constant 0 : i6      // no atomic ops
  %c64 = hw.constant 0 : i64
  %araddr_a = hw.constant 0 : i32      // sub_module_a's window (base 0, direct via xbar1)
  %araddr_b = hw.constant 4096 : i32   // sub_module_b's window (base 4096, chained via xbar2)
  %awaddr_w_a = hw.constant 8 : i32    // word 1's byte address within a's window
  %awaddr_w_b = hw.constant 4104 : i32 // word 1's byte address within b's window (4096 + 8)
  %arsize = hw.constant 3 : i3
  %arlen4 = hw.constant 3 : i8         // arlen = beats - 1 -> 4-beat read bursts
  %awlen2 = hw.constant 1 : i8         // awlen = beats - 1 -> 2-beat write burst
  %arburst_incr = hw.constant 1 : i2   // INCR
  %wstrb_full = hw.constant 0xFF : i8  // full-word writes only (stopgap, see README)
  %new_a_word1 = hw.constant 0xAAAAAAAA11111111 : i64
  %new_a_word2 = hw.constant 0xBBBBBBBB22222222 : i64
  %new_b_word1 = hw.constant 0xCCCCCCCC33333333 : i64
  %new_b_word2 = hw.constant 0xDDDDDDDD44444444 : i64

  %rst = comb.xor %rst_ni, %t : i1   // active-high reset from active-low rst_ni

  // Explicit 14-phase sequencer: single.mlir's 7-phase read-write-read
  // sequence (ISSUE_R1/RECV_R1/ISSUE_W/SEND_W/RECV_WRESP/ISSUE_R2/RECV_R2),
  // run once against sub_module_a (direct via xbar1) then once against
  // sub_module_b (chained via xbar2). As with every FSM in this harness
  // (see the mixed_fanout.mlir lesson from the read-only version), each
  // transition below is gated on that phase's own icmp-eq predicate, never a
  // signal shared across phases.
  //    0 ISSUE_R1_A:   arvalid=1, araddr=0 (4-beat read).        AR accept    -> 1
  //    1 RECV_R1_A:    rready=1, capture a_beat0..3.             rlast accept -> 2
  //    2 ISSUE_W_A:    awvalid=1, awaddr=8 (2-beat write).       AW accept    -> 3
  //    3 SEND_W_A:     wvalid=1, stream new_a_word1, new_a_word2. last W accept -> 4
  //    4 RECV_WRESP_A: bready=1.                                  B accept    -> 5
  //    5 ISSUE_R2_A:   arvalid=1, araddr=0 (re-read all 4).       AR accept    -> 6
  //    6 RECV_R2_A:    rready=1, capture after_a_beat0..3.        rlast accept -> 7
  //    7 ISSUE_R1_B:   arvalid=1, araddr=4096 (4-beat read).      AR accept    -> 8
  //    8 RECV_R1_B:    rready=1, capture b_beat0..3.              rlast accept -> 9
  //    9 ISSUE_W_B:    awvalid=1, awaddr=4104 (2-beat write).     AW accept    -> 10
  //   10 SEND_W_B:     wvalid=1, stream new_b_word1, new_b_word2. last W accept -> 11
  //   11 RECV_WRESP_B: bready=1.                                  B accept    -> 12
  //   12 ISSUE_R2_B:   arvalid=1, araddr=4096 (re-read all 4).    AR accept    -> 13
  //   13 RECV_R2_B:    rready=1, capture after_b_beat0..3.        rlast accept -> done=1, stays at 13
  %p0 = hw.constant 0 : i4
  %p1 = hw.constant 1 : i4
  %p2 = hw.constant 2 : i4
  %p3 = hw.constant 3 : i4
  %p4 = hw.constant 4 : i4
  %p5 = hw.constant 5 : i4
  %p6 = hw.constant 6 : i4
  %p7 = hw.constant 7 : i4
  %p8 = hw.constant 8 : i4
  %p9 = hw.constant 9 : i4
  %p10 = hw.constant 10 : i4
  %p11 = hw.constant 11 : i4
  %p12 = hw.constant 12 : i4
  %p13 = hw.constant 13 : i4
  %phase_q = seq.compreg %phase_next, %clk_i reset %rst, %p0 : i4

  %in_p0 = comb.icmp eq %phase_q, %p0 : i4
  %in_p1 = comb.icmp eq %phase_q, %p1 : i4
  %in_p2 = comb.icmp eq %phase_q, %p2 : i4
  %in_p3 = comb.icmp eq %phase_q, %p3 : i4
  %in_p4 = comb.icmp eq %phase_q, %p4 : i4
  %in_p5 = comb.icmp eq %phase_q, %p5 : i4
  %in_p6 = comb.icmp eq %phase_q, %p6 : i4
  %in_p7 = comb.icmp eq %phase_q, %p7 : i4
  %in_p8 = comb.icmp eq %phase_q, %p8 : i4
  %in_p9 = comb.icmp eq %phase_q, %p9 : i4
  %in_p10 = comb.icmp eq %phase_q, %p10 : i4
  %in_p11 = comb.icmp eq %phase_q, %p11 : i4
  %in_p12 = comb.icmp eq %phase_q, %p12 : i4
  %in_p13 = comb.icmp eq %phase_q, %p13 : i4

  // The AXI4 interface this manager drives. Every payload and valid is a
  // function of the phase register, so nothing loops back through the port.
  %aw = hw.struct_create (%c4, %awaddr_sel, %awlen2, %arsize, %arburst_incr, %f,
                          %c4, %c3, %c4, %c4, %atop0, %f) : !aw
  %w = hw.struct_create (%wdata_sel, %wstrb_full, %w_beat_idx_q, %f) : !w
  %ar = hw.struct_create (%c4, %araddr_sel, %arlen4, %arsize, %arburst_incr, %f,
                          %c4, %c3, %c4, %c4, %f) : !ar
  %port, %awready, %wready, %b, %bvalid, %arready, %r, %rvalid =
    axi4.channel_structs_to_port %clk_i, %rst_ni
      aw %aw, %awvalid w %w, %wvalid b %bready
      ar %ar, %arvalid_full r %t : !mgr
  %rdata = hw.struct_extract %r["data"] : !r
  %rlast = hw.struct_extract %r["last"] : !r

  // ---- AR channel: 4 read-issue phases (R1/R2 x A/B), 2 distinct addresses.
  %arvalid_ab = comb.or %in_p0, %in_p5 : i1
  %arvalid = comb.or %arvalid_ab, %in_p7 : i1
  %arvalid_full = comb.or %arvalid, %in_p12 : i1
  %is_b_read = comb.or %in_p7, %in_p12 : i1
  %araddr_sel = comb.mux %is_b_read, %araddr_b, %araddr_a : i32

  %ar_accept_r1_a = comb.and %in_p0, %arready : i1
  %ar_accept_r2_a = comb.and %in_p5, %arready : i1
  %ar_accept_r1_b = comb.and %in_p7, %arready : i1
  %ar_accept_r2_b = comb.and %in_p12, %arready : i1
  %any_ar_accept_ab = comb.or %ar_accept_r1_a, %ar_accept_r2_a : i1
  %any_ar_accept_b = comb.or %ar_accept_r1_b, %ar_accept_r2_b : i1
  %any_ar_accept = comb.or %any_ar_accept_ab, %any_ar_accept_b : i1
  %r_last_beat = comb.and %rvalid, %rlast : i1

  // ---- AW/W/B channels: 2 write-issue phases (A/B), each a 2-beat burst.
  %aw_accept_a = comb.and %in_p2, %awready : i1
  %aw_accept_b = comb.and %in_p9, %awready : i1
  %any_aw_accept = comb.or %aw_accept_a, %aw_accept_b : i1
  %awvalid = comb.or %in_p2, %in_p9 : i1
  %awaddr_sel = comb.mux %in_p9, %awaddr_w_b, %awaddr_w_a : i32

  %w_beat_accept_a = comb.and %in_p3, %wready : i1
  %w_beat_accept_b = comb.and %in_p10, %wready : i1
  %any_w_beat_accept = comb.or %w_beat_accept_a, %w_beat_accept_b : i1
  %wvalid = comb.or %in_p3, %in_p10 : i1
  %w_last_accept_a = comb.and %w_beat_accept_a, %w_beat_idx_q : i1
  %w_last_accept_b = comb.and %w_beat_accept_b, %w_beat_idx_q : i1

  %b_accept_a = comb.and %in_p4, %bvalid : i1
  %b_accept_b = comb.and %in_p11, %bvalid : i1
  %bready = comb.or %in_p4, %in_p11 : i1

  // ---- Phase transitions: each condition mutually exclusive by
  // construction (gated on a distinct phase), chained the same way as
  // single.mlir's 7-phase sequence, just extended to run twice.
  %r1_a_advance = comb.and %in_p1, %r_last_beat : i1
  %r2_a_advance = comb.and %in_p6, %r_last_beat : i1
  %r1_b_advance = comb.and %in_p8, %r_last_beat : i1
  %r2_b_advance = comb.and %in_p13, %r_last_beat : i1

  %phase_after_ar_r1_a = comb.mux %ar_accept_r1_a, %p1, %phase_q : i4
  %phase_after_r1_a = comb.mux %r1_a_advance, %p2, %phase_after_ar_r1_a : i4
  %phase_after_aw_a = comb.mux %aw_accept_a, %p3, %phase_after_r1_a : i4
  %phase_after_w_a = comb.mux %w_last_accept_a, %p4, %phase_after_aw_a : i4
  %phase_after_b_a = comb.mux %b_accept_a, %p5, %phase_after_w_a : i4
  %phase_after_ar_r2_a = comb.mux %ar_accept_r2_a, %p6, %phase_after_b_a : i4
  %phase_after_r2_a = comb.mux %r2_a_advance, %p7, %phase_after_ar_r2_a : i4
  %phase_after_ar_r1_b = comb.mux %ar_accept_r1_b, %p8, %phase_after_r2_a : i4
  %phase_after_r1_b = comb.mux %r1_b_advance, %p9, %phase_after_ar_r1_b : i4
  %phase_after_aw_b = comb.mux %aw_accept_b, %p10, %phase_after_r1_b : i4
  %phase_after_w_b = comb.mux %w_last_accept_b, %p11, %phase_after_aw_b : i4
  %phase_after_b_b = comb.mux %b_accept_b, %p12, %phase_after_w_b : i4
  %phase_next = comb.mux %ar_accept_r2_b, %p13, %phase_after_b_b : i4

  %done_next = comb.mux %r2_b_advance, %t, %done_q : i1
  %done_q = seq.compreg %done_next, %clk_i reset %rst, %f : i1

  // Write-beat index (shared by both write bursts -- mutually exclusive in
  // time): 0 selects wordN_1/not-last, 1 selects wordN_2/last.
  %w_beat_idx_next = comb.mux %any_aw_accept, %f, %w_beat_idx_after : i1
  %w_beat_idx_after = comb.mux %any_w_beat_accept, %t, %w_beat_idx_q : i1
  %w_beat_idx_q = seq.compreg %w_beat_idx_next, %clk_i reset %rst, %f : i1
  %wdata_a_sel = comb.mux %w_beat_idx_q, %new_a_word2, %new_a_word1 : i64
  %wdata_b_sel = comb.mux %w_beat_idx_q, %new_b_word2, %new_b_word1 : i64
  %wdata_sel = comb.mux %in_p10, %wdata_b_sel, %wdata_a_sel : i64

  // Beat counter (shared by all 4 read bursts -- mutually exclusive in time)
  // resets on any read burst's AR accept, then increments per accepted beat.
  %bc0 = hw.constant 0 : i2
  %bc1 = hw.constant 1 : i2
  %bc2 = hw.constant 2 : i2
  %bc3 = hw.constant 3 : i2
  %beat_count_after_recv = comb.mux %rvalid, %beat_count_inc, %beat_count_q : i2
  %beat_count_next = comb.mux %any_ar_accept, %bc0, %beat_count_after_recv : i2
  %beat_count_q = seq.compreg %beat_count_next, %clk_i reset %rst, %bc0 : i2
  %beat_count_inc = comb.add %beat_count_q, %bc1 : i2

  %is_beat0 = comb.icmp eq %beat_count_q, %bc0 : i2
  %is_beat1 = comb.icmp eq %beat_count_q, %bc1 : i2
  %is_beat2 = comb.icmp eq %beat_count_q, %bc2 : i2
  %is_beat3 = comb.icmp eq %beat_count_q, %bc3 : i2
  %recv_r1_a = comb.and %rvalid, %in_p1 : i1
  %recv_r2_a = comb.and %rvalid, %in_p6 : i1
  %recv_r1_b = comb.and %rvalid, %in_p8 : i1
  %recv_r2_b = comb.and %rvalid, %in_p13 : i1

  %en_a_beat0 = comb.and %recv_r1_a, %is_beat0 : i1
  %en_a_beat1 = comb.and %recv_r1_a, %is_beat1 : i1
  %en_a_beat2 = comb.and %recv_r1_a, %is_beat2 : i1
  %en_a_beat3 = comb.and %recv_r1_a, %is_beat3 : i1
  %en_after_a_beat0 = comb.and %recv_r2_a, %is_beat0 : i1
  %en_after_a_beat1 = comb.and %recv_r2_a, %is_beat1 : i1
  %en_after_a_beat2 = comb.and %recv_r2_a, %is_beat2 : i1
  %en_after_a_beat3 = comb.and %recv_r2_a, %is_beat3 : i1
  %en_b_beat0 = comb.and %recv_r1_b, %is_beat0 : i1
  %en_b_beat1 = comb.and %recv_r1_b, %is_beat1 : i1
  %en_b_beat2 = comb.and %recv_r1_b, %is_beat2 : i1
  %en_b_beat3 = comb.and %recv_r1_b, %is_beat3 : i1
  %en_after_b_beat0 = comb.and %recv_r2_b, %is_beat0 : i1
  %en_after_b_beat1 = comb.and %recv_r2_b, %is_beat1 : i1
  %en_after_b_beat2 = comb.and %recv_r2_b, %is_beat2 : i1
  %en_after_b_beat3 = comb.and %recv_r2_b, %is_beat3 : i1

  %a_beat0_next = comb.mux %en_a_beat0, %rdata, %a_beat0_q : i64
  %a_beat0_q = seq.compreg %a_beat0_next, %clk_i reset %rst, %c64 : i64
  %a_beat1_next = comb.mux %en_a_beat1, %rdata, %a_beat1_q : i64
  %a_beat1_q = seq.compreg %a_beat1_next, %clk_i reset %rst, %c64 : i64
  %a_beat2_next = comb.mux %en_a_beat2, %rdata, %a_beat2_q : i64
  %a_beat2_q = seq.compreg %a_beat2_next, %clk_i reset %rst, %c64 : i64
  %a_beat3_next = comb.mux %en_a_beat3, %rdata, %a_beat3_q : i64
  %a_beat3_q = seq.compreg %a_beat3_next, %clk_i reset %rst, %c64 : i64
  %after_a_beat0_next = comb.mux %en_after_a_beat0, %rdata, %after_a_beat0_q : i64
  %after_a_beat0_q = seq.compreg %after_a_beat0_next, %clk_i reset %rst, %c64 : i64
  %after_a_beat1_next = comb.mux %en_after_a_beat1, %rdata, %after_a_beat1_q : i64
  %after_a_beat1_q = seq.compreg %after_a_beat1_next, %clk_i reset %rst, %c64 : i64
  %after_a_beat2_next = comb.mux %en_after_a_beat2, %rdata, %after_a_beat2_q : i64
  %after_a_beat2_q = seq.compreg %after_a_beat2_next, %clk_i reset %rst, %c64 : i64
  %after_a_beat3_next = comb.mux %en_after_a_beat3, %rdata, %after_a_beat3_q : i64
  %after_a_beat3_q = seq.compreg %after_a_beat3_next, %clk_i reset %rst, %c64 : i64

  %b_beat0_next = comb.mux %en_b_beat0, %rdata, %b_beat0_q : i64
  %b_beat0_q = seq.compreg %b_beat0_next, %clk_i reset %rst, %c64 : i64
  %b_beat1_next = comb.mux %en_b_beat1, %rdata, %b_beat1_q : i64
  %b_beat1_q = seq.compreg %b_beat1_next, %clk_i reset %rst, %c64 : i64
  %b_beat2_next = comb.mux %en_b_beat2, %rdata, %b_beat2_q : i64
  %b_beat2_q = seq.compreg %b_beat2_next, %clk_i reset %rst, %c64 : i64
  %b_beat3_next = comb.mux %en_b_beat3, %rdata, %b_beat3_q : i64
  %b_beat3_q = seq.compreg %b_beat3_next, %clk_i reset %rst, %c64 : i64
  %after_b_beat0_next = comb.mux %en_after_b_beat0, %rdata, %after_b_beat0_q : i64
  %after_b_beat0_q = seq.compreg %after_b_beat0_next, %clk_i reset %rst, %c64 : i64
  %after_b_beat1_next = comb.mux %en_after_b_beat1, %rdata, %after_b_beat1_q : i64
  %after_b_beat1_q = seq.compreg %after_b_beat1_next, %clk_i reset %rst, %c64 : i64
  %after_b_beat2_next = comb.mux %en_after_b_beat2, %rdata, %after_b_beat2_q : i64
  %after_b_beat2_q = seq.compreg %after_b_beat2_next, %clk_i reset %rst, %c64 : i64
  %after_b_beat3_next = comb.mux %en_after_b_beat3, %rdata, %after_b_beat3_q : i64
  %after_b_beat3_q = seq.compreg %after_b_beat3_next, %clk_i reset %rst, %c64 : i64

  hw.output %port, %done_q,
            %a_beat0_q, %a_beat1_q, %a_beat2_q, %a_beat3_q,
            %after_a_beat0_q, %after_a_beat1_q, %after_a_beat2_q, %after_a_beat3_q,
            %b_beat0_q, %b_beat1_q, %b_beat2_q, %b_beat3_q,
            %after_b_beat0_q, %after_b_beat1_q, %after_b_beat2_q, %after_b_beat3_q
    : !mgr, i1, i64, i64, i64, i64, i64, i64, i64, i64,
      i64, i64, i64, i64, i64, i64, i64, i64
}

hw.module @sub_module_a(in %clk_i : !seq.clock, in %rst_ni : i1, in %axi : !sub_a) {
  %false = hw.constant false
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c2_i2 = hw.constant 2 : i2
  %c3_i2 = hw.constant 3 : i2
  %c0_i4 = hw.constant 0 : i4
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %c0_i64 = hw.constant 0 : i64

  %rst = comb.xor %rst_ni, %true : i1   // active-high reset from active-low rst_ni

  // Preloaded RAM contents (depth 4): reset seeds each word to its known
  // starting value, so the read-after-write self-check has identifiable data.
  %word0_val = hw.constant 0xCAFEF00DCAFEF00D : i64
  %word1_val = hw.constant 0xDEADBEEFDEADBEEF : i64
  %word2_val = hw.constant 0xFACEFEEDFACEFEED : i64
  %word3_val = hw.constant 0x8BADF00D8BADF00D : i64

  // The AXI4 interface this subordinate answers on. Every ready, payload and
  // valid it drives is a function of a register, so nothing loops back through
  // the port. wstrb is ignored (stopgap, see README).
  %b = hw.struct_create (%bid_q, %c0_i2, %false) : !b
  %r = hw.struct_create (%rid_q, %selected, %c0_i2, %is_last_beat, %false) : !r
  %aw, %awvalid, %w, %wvalid, %bready, %ar, %arvalid, %rready =
    axi4.port_to_channel_structs %clk_i, %rst_ni, %axi
      aw %w_in_idle w %w_in_data b %b, %w_in_resp
      ar %idle r %r, %state_q : !sub_a
  %awid = hw.struct_extract %aw["id"] : !aw
  %awaddr = hw.struct_extract %aw["addr"] : !aw
  %wdata = hw.struct_extract %w["data"] : !w
  %wlast = hw.struct_extract %w["last"] : !w
  %arid = hw.struct_extract %ar["id"] : !ar
  %araddr = hw.struct_extract %ar["addr"] : !ar
  %arlen = hw.struct_extract %ar["len"] : !ar

  // ---- Read side: 2-state FSM, state_q = 0 IDLE (accept AR) / 1 BURST
  // (stream R beats) -- unchanged shape from the read-only ROM version.
  %state_q = seq.compreg %state_next, %clk_i reset %rst, %false : i1
  %idle = comb.xor %state_q, %true : i1
  %accept = comb.and %arvalid, %idle : i1
  %r_handshake = comb.and %state_q, %rready : i1
  %is_last_beat = comb.icmp eq %remaining_q, %c0_i8 : i8
  %burst_done = comb.and %r_handshake, %is_last_beat : i1
  %back_to_idle = comb.mux %burst_done, %false, %state_q : i1
  %state_next = comb.mux %accept, %true, %back_to_idle : i1

  %rid_next = comb.mux %accept, %arid, %rid_q : i4
  %rid_q = seq.compreg %rid_next, %clk_i reset %rst, %c0_i4 : i4

  %remaining_dec = comb.sub %remaining_q, %c1_i8 : i8
  %remaining_after_beat = comb.mux %r_handshake, %remaining_dec, %remaining_q : i8
  %remaining_next = comb.mux %accept, %arlen, %remaining_after_beat : i8
  %remaining_q = seq.compreg %remaining_next, %clk_i reset %rst, %c0_i8 : i8

  %idx_start = comb.extract %araddr from 3 : (i32) -> i2
  %idx_inc = comb.add %idx_q, %c1_i2 : i2
  %idx_after_beat = comb.mux %r_handshake, %idx_inc, %idx_q : i2
  %idx_next = comb.mux %accept, %idx_start, %idx_after_beat : i2
  %idx_q = seq.compreg %idx_next, %clk_i reset %rst, %c0_i2 : i2

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
  %w_state_q = seq.compreg %w_state_next, %clk_i reset %rst, %ws_idle : i2
  %w_in_idle = comb.icmp eq %w_state_q, %ws_idle : i2
  %w_in_data = comb.icmp eq %w_state_q, %ws_data : i2
  %w_in_resp = comb.icmp eq %w_state_q, %ws_resp : i2

  %aw_accept = comb.and %awvalid, %w_in_idle : i1
  %w_beat_accept = comb.and %wvalid, %w_in_data : i1
  %w_last_accept = comb.and %w_beat_accept, %wlast : i1
  %b_accept = comb.and %bready, %w_in_resp : i1

  %w_state_after_aw = comb.mux %aw_accept, %ws_data, %w_state_q : i2
  %w_state_after_data = comb.mux %w_last_accept, %ws_resp, %w_state_after_aw : i2
  %w_state_next = comb.mux %b_accept, %ws_idle, %w_state_after_data : i2

  %bid_next = comb.mux %aw_accept, %awid, %bid_q : i4
  %bid_q = seq.compreg %bid_next, %clk_i reset %rst, %c0_i4 : i4

  %w_idx_start = comb.extract %awaddr from 3 : (i32) -> i2
  %w_idx_inc = comb.add %w_idx_q, %c1_i2 : i2
  %w_idx_after_beat = comb.mux %w_beat_accept, %w_idx_inc, %w_idx_q : i2
  %w_idx_next = comb.mux %aw_accept, %w_idx_start, %w_idx_after_beat : i2
  %w_idx_q = seq.compreg %w_idx_next, %clk_i reset %rst, %c0_i2 : i2

  %w_is_word0 = comb.icmp eq %w_idx_q, %c0_i2 : i2
  %w_is_word1 = comb.icmp eq %w_idx_q, %c1_i2 : i2
  %w_is_word2 = comb.icmp eq %w_idx_q, %c2_i2 : i2
  %w_is_word3 = comb.icmp eq %w_idx_q, %c3_i2 : i2
  %we0 = comb.and %w_beat_accept, %w_is_word0 : i1
  %we1 = comb.and %w_beat_accept, %w_is_word1 : i1
  %we2 = comb.and %w_beat_accept, %w_is_word2 : i1
  %we3 = comb.and %w_beat_accept, %w_is_word3 : i1

  %word0_next = comb.mux %we0, %wdata, %word0_q : i64
  %word0_q = seq.compreg %word0_next, %clk_i reset %rst, %word0_val : i64
  %word1_next = comb.mux %we1, %wdata, %word1_q : i64
  %word1_q = seq.compreg %word1_next, %clk_i reset %rst, %word1_val : i64
  %word2_next = comb.mux %we2, %wdata, %word2_q : i64
  %word2_q = seq.compreg %word2_next, %clk_i reset %rst, %word2_val : i64
  %word3_next = comb.mux %we3, %wdata, %word3_q : i64
  %word3_q = seq.compreg %word3_next, %clk_i reset %rst, %word3_val : i64

  hw.output
}

hw.module @sub_module_b(in %clk_i : !seq.clock, in %rst_ni : i1, in %axi : !sub_b) {
  %false = hw.constant false
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c2_i2 = hw.constant 2 : i2
  %c3_i2 = hw.constant 3 : i2
  %c0_i4 = hw.constant 0 : i4
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %c0_i64 = hw.constant 0 : i64

  %rst = comb.xor %rst_ni, %true : i1   // active-high reset from active-low rst_ni

  // Distinct starting contents from sub_module_a's, so the waveform/
  // self-check can prove which subordinate answered which burst.
  %word0_val = hw.constant 0xFEEDFACEFEEDFACE : i64
  %word1_val = hw.constant 0xB105F00DB105F00D : i64
  %word2_val = hw.constant 0x5CA1AB1E5CA1AB1E : i64
  %word3_val = hw.constant 0x0DEFACED0DEFACED : i64

  // The AXI4 interface this subordinate answers on. Every ready, payload and
  // valid it drives is a function of a register, so nothing loops back through
  // the port. wstrb is ignored (stopgap, see README).
  %b = hw.struct_create (%bid_q, %c0_i2, %false) : !b
  %r = hw.struct_create (%rid_q, %selected, %c0_i2, %is_last_beat, %false) : !r
  %aw, %awvalid, %w, %wvalid, %bready, %ar, %arvalid, %rready =
    axi4.port_to_channel_structs %clk_i, %rst_ni, %axi
      aw %w_in_idle w %w_in_data b %b, %w_in_resp
      ar %idle r %r, %state_q : !sub_b
  %awid = hw.struct_extract %aw["id"] : !aw
  %awaddr = hw.struct_extract %aw["addr"] : !aw
  %wdata = hw.struct_extract %w["data"] : !w
  %wlast = hw.struct_extract %w["last"] : !w
  %arid = hw.struct_extract %ar["id"] : !ar
  %araddr = hw.struct_extract %ar["addr"] : !ar
  %arlen = hw.struct_extract %ar["len"] : !ar

  // ---- Read side: 2-state FSM, state_q = 0 IDLE (accept AR) / 1 BURST
  // (stream R beats) -- unchanged shape from the read-only ROM version.
  %state_q = seq.compreg %state_next, %clk_i reset %rst, %false : i1
  %idle = comb.xor %state_q, %true : i1
  %accept = comb.and %arvalid, %idle : i1
  %r_handshake = comb.and %state_q, %rready : i1
  %is_last_beat = comb.icmp eq %remaining_q, %c0_i8 : i8
  %burst_done = comb.and %r_handshake, %is_last_beat : i1
  %back_to_idle = comb.mux %burst_done, %false, %state_q : i1
  %state_next = comb.mux %accept, %true, %back_to_idle : i1

  %rid_next = comb.mux %accept, %arid, %rid_q : i4
  %rid_q = seq.compreg %rid_next, %clk_i reset %rst, %c0_i4 : i4

  %remaining_dec = comb.sub %remaining_q, %c1_i8 : i8
  %remaining_after_beat = comb.mux %r_handshake, %remaining_dec, %remaining_q : i8
  %remaining_next = comb.mux %accept, %arlen, %remaining_after_beat : i8
  %remaining_q = seq.compreg %remaining_next, %clk_i reset %rst, %c0_i8 : i8

  %idx_start = comb.extract %araddr from 3 : (i32) -> i2
  %idx_inc = comb.add %idx_q, %c1_i2 : i2
  %idx_after_beat = comb.mux %r_handshake, %idx_inc, %idx_q : i2
  %idx_next = comb.mux %accept, %idx_start, %idx_after_beat : i2
  %idx_q = seq.compreg %idx_next, %clk_i reset %rst, %c0_i2 : i2

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
  %w_state_q = seq.compreg %w_state_next, %clk_i reset %rst, %ws_idle : i2
  %w_in_idle = comb.icmp eq %w_state_q, %ws_idle : i2
  %w_in_data = comb.icmp eq %w_state_q, %ws_data : i2
  %w_in_resp = comb.icmp eq %w_state_q, %ws_resp : i2

  %aw_accept = comb.and %awvalid, %w_in_idle : i1
  %w_beat_accept = comb.and %wvalid, %w_in_data : i1
  %w_last_accept = comb.and %w_beat_accept, %wlast : i1
  %b_accept = comb.and %bready, %w_in_resp : i1

  %w_state_after_aw = comb.mux %aw_accept, %ws_data, %w_state_q : i2
  %w_state_after_data = comb.mux %w_last_accept, %ws_resp, %w_state_after_aw : i2
  %w_state_next = comb.mux %b_accept, %ws_idle, %w_state_after_data : i2

  %bid_next = comb.mux %aw_accept, %awid, %bid_q : i4
  %bid_q = seq.compreg %bid_next, %clk_i reset %rst, %c0_i4 : i4

  %w_idx_start = comb.extract %awaddr from 3 : (i32) -> i2
  %w_idx_inc = comb.add %w_idx_q, %c1_i2 : i2
  %w_idx_after_beat = comb.mux %w_beat_accept, %w_idx_inc, %w_idx_q : i2
  %w_idx_next = comb.mux %aw_accept, %w_idx_start, %w_idx_after_beat : i2
  %w_idx_q = seq.compreg %w_idx_next, %clk_i reset %rst, %c0_i2 : i2

  %w_is_word0 = comb.icmp eq %w_idx_q, %c0_i2 : i2
  %w_is_word1 = comb.icmp eq %w_idx_q, %c1_i2 : i2
  %w_is_word2 = comb.icmp eq %w_idx_q, %c2_i2 : i2
  %w_is_word3 = comb.icmp eq %w_idx_q, %c3_i2 : i2
  %we0 = comb.and %w_beat_accept, %w_is_word0 : i1
  %we1 = comb.and %w_beat_accept, %w_is_word1 : i1
  %we2 = comb.and %w_beat_accept, %w_is_word2 : i1
  %we3 = comb.and %w_beat_accept, %w_is_word3 : i1

  %word0_next = comb.mux %we0, %wdata, %word0_q : i64
  %word0_q = seq.compreg %word0_next, %clk_i reset %rst, %word0_val : i64
  %word1_next = comb.mux %we1, %wdata, %word1_q : i64
  %word1_q = seq.compreg %word1_next, %clk_i reset %rst, %word1_val : i64
  %word2_next = comb.mux %we2, %wdata, %word2_q : i64
  %word2_q = seq.compreg %word2_next, %clk_i reset %rst, %word2_val : i64
  %word3_next = comb.mux %we3, %wdata, %word3_q : i64
  %word3_q = seq.compreg %word3_next, %clk_i reset %rst, %word3_val : i64

  hw.output
}

hw.module @AXITop(in %clk_i : !seq.clock, in %rst_ni : i1) {
  %mgr, %done, %a_beat0, %a_beat1, %a_beat2, %a_beat3,
  %after_a0, %after_a1, %after_a2, %after_a3,
  %b_beat0, %b_beat1, %b_beat2, %b_beat3,
  %after_b0, %after_b1, %after_b2, %after_b3 =
    hw.instance "mgr" @mgr_module(clk_i: %clk_i: !seq.clock, rst_ni: %rst_ni: i1)
      -> (axi: !mgr, done: i1,
          a_beat0: i64, a_beat1: i64, a_beat2: i64, a_beat3: i64,
          after_a_beat0: i64, after_a_beat1: i64, after_a_beat2: i64, after_a_beat3: i64,
          b_beat0: i64, b_beat1: i64, b_beat2: i64, b_beat3: i64,
          after_b_beat0: i64, after_b_beat1: i64, after_b_beat2: i64, after_b_beat3: i64)
  // xbar1 fans out to both a direct subordinate and the chained xbar2.
  %to_a, %to_xbar2 = axi4.xbar %clk_i, %rst_ni mgrs %mgr : (!mgr) -> (!sub_a, !mid)
  %to_b = axi4.xbar %clk_i, %rst_ni mgrs %to_xbar2 : (!mid) -> (!sub_b)
  hw.instance "sub_a" @sub_module_a(clk_i: %clk_i: !seq.clock, rst_ni: %rst_ni: i1,
                                    axi: %to_a: !sub_a) -> ()
  hw.instance "sub_b" @sub_module_b(clk_i: %clk_i: !seq.clock, rst_ni: %rst_ni: i1,
                                    axi: %to_b: !sub_b) -> ()
  hw.output
}
