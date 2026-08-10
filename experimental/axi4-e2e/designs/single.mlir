// manager -> xbar -> subordinate, inside a top module. Emits AXITop + one
// axi_xbar_1u1d wrapper. mgr_module runs a 7-phase sequence: (1) a 4-beat
// AXI4 INCR read burst from address 0, (2) a 2-beat INCR write burst
// overwriting words 1 and 2 (address 8) with new data, (3) a second 4-beat
// read burst re-reading all 4 words -- proving that only the 2 written
// words changed. sub_module is a real 4-word read/write RAM whose starting
// contents are seeded by reset. See sim/tb_axitop_single.sv for the
// waveform testbench that exercises this end to end.
!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 1, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 2>, <incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

!aw = !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, atop: i6, user: i1>
!w = !hw.struct<data: i64, strb: i8, last: i1, user: i1>
!b = !hw.struct<id: i4, resp: i2, user: i1>
!ar = !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i1>
!r = !hw.struct<id: i4, data: i64, resp: i2, last: i1, user: i1>

hw.module @mgr_module(in %clk_i : !seq.clock, in %rst_ni : i1, out axi : !port,
                      out done : i1, out beat0 : i64, out beat1 : i64,
                      out beat2 : i64, out beat3 : i64, out after_beat0 : i64,
                      out after_beat1 : i64, out after_beat2 : i64,
                      out after_beat3 : i64) {
  %f = hw.constant 0 : i1
  %t = hw.constant 1 : i1
  %c2 = hw.constant 0 : i2
  %c3 = hw.constant 0 : i3
  %c4 = hw.constant 0 : i4
  %c8 = hw.constant 0 : i8
  %atop0 = hw.constant 0 : i6      // no atomic ops
  %c32 = hw.constant 0 : i32
  %c64 = hw.constant 0 : i64
  %arsize = hw.constant 3 : i3
  %arlen4 = hw.constant 3 : i8         // arlen = beats - 1 -> 4-beat read bursts
  %awlen2 = hw.constant 1 : i8         // awlen = beats - 1 -> 2-beat write burst
  %arburst_incr = hw.constant 1 : i2   // INCR
  %awaddr_w = hw.constant 8 : i32      // word 1's byte address (write target)
  %wstrb_full = hw.constant 0xFF : i8  // full-word writes only (stopgap, see README)
  %new_word1 = hw.constant 0xAAAAAAAA11111111 : i64
  %new_word2 = hw.constant 0xBBBBBBBB22222222 : i64

  %rst = comb.xor %rst_ni, %t : i1   // active-high reset from active-low rst_ni

  // Explicit 7-phase sequencer. Each transition below is gated on that
  // phase's own icmp-eq predicate (never a signal shared across phases) --
  // the mixed_fanout.mlir bug came from conflating two different phases'
  // "accept" signals into one, so this design keeps every phase's handshake
  // fully self-contained.
  //   0 ISSUE_R1: arvalid=1, araddr=0 (4-beat read).       AR accept  -> 1
  //   1 RECV_R1:  rready=1, capture beat0..3.              rlast accept -> 2
  //   2 ISSUE_W:  awvalid=1, awaddr=8 (2-beat write).      AW accept  -> 3
  //   3 SEND_W:   wvalid=1, stream new_word1, new_word2.   last W accept -> 4
  //   4 RECV_B:   bready=1.                                 B accept  -> 5
  //   5 ISSUE_R2: arvalid=1, araddr=0 (re-read all 4).      AR accept  -> 6
  //   6 RECV_R2:  rready=1, capture after_beat0..3.         rlast accept -> done=1, stays at 6
  %p0 = hw.constant 0 : i3
  %p1 = hw.constant 1 : i3
  %p2 = hw.constant 2 : i3
  %p3 = hw.constant 3 : i3
  %p4 = hw.constant 4 : i3
  %p5 = hw.constant 5 : i3
  %p6 = hw.constant 6 : i3
  %phase_q = seq.compreg %phase_next, %clk_i reset %rst, %p0 : i3

  %in_p0 = comb.icmp eq %phase_q, %p0 : i3
  %in_p1 = comb.icmp eq %phase_q, %p1 : i3
  %in_p2 = comb.icmp eq %phase_q, %p2 : i3
  %in_p3 = comb.icmp eq %phase_q, %p3 : i3
  %in_p4 = comb.icmp eq %phase_q, %p4 : i3
  %in_p5 = comb.icmp eq %phase_q, %p5 : i3
  %in_p6 = comb.icmp eq %phase_q, %p6 : i3

  // The AXI4 interface this manager drives. Every payload and valid is a
  // function of the phase register, so nothing loops back through the port.
  %aw = hw.struct_create (%c4, %awaddr_w, %awlen2, %arsize, %arburst_incr, %f,
                          %c4, %c3, %c4, %c4, %atop0, %f) : !aw
  %w = hw.struct_create (%wdata_sel, %wstrb_full, %w_beat_idx_q, %f) : !w
  %ar = hw.struct_create (%c4, %c32, %arlen4, %arsize, %arburst_incr, %f,
                          %c4, %c3, %c4, %c4, %f) : !ar
  %port, %awready, %wready, %b, %bvalid, %arready, %r, %rvalid =
    axi4.channel_structs_to_port %clk_i, %rst_ni
      aw %aw, %in_p2 w %w, %in_p3 b %in_p4 ar %ar, %arvalid r %t : !port
  %rdata = hw.struct_extract %r["data"] : !r
  %rlast = hw.struct_extract %r["last"] : !r

  %arvalid = comb.or %in_p0, %in_p5 : i1
  %ar_accept_r1 = comb.and %in_p0, %arready : i1
  %ar_accept_r2 = comb.and %in_p5, %arready : i1
  %any_ar_accept = comb.or %ar_accept_r1, %ar_accept_r2 : i1
  %r_last_beat = comb.and %rvalid, %rlast : i1

  %aw_accept = comb.and %in_p2, %awready : i1
  %w_beat_accept = comb.and %in_p3, %wready : i1
  %w_last_accept = comb.and %w_beat_accept, %w_beat_idx_q : i1
  %b_accept = comb.and %in_p4, %bvalid : i1

  %r1_advance = comb.and %in_p1, %r_last_beat : i1
  %phase_after_ar1 = comb.mux %ar_accept_r1, %p1, %phase_q : i3
  %phase_after_r1last = comb.mux %r1_advance, %p2, %phase_after_ar1 : i3
  %phase_after_aw = comb.mux %aw_accept, %p3, %phase_after_r1last : i3
  %phase_after_wlast = comb.mux %w_last_accept, %p4, %phase_after_aw : i3
  %phase_after_b = comb.mux %b_accept, %p5, %phase_after_wlast : i3
  %phase_next = comb.mux %ar_accept_r2, %p6, %phase_after_b : i3

  %r2_advance = comb.and %in_p6, %r_last_beat : i1
  %done_next = comb.mux %r2_advance, %t, %done_q : i1
  %done_q = seq.compreg %done_next, %clk_i reset %rst, %f : i1

  // Write-beat index: 0 selects new_word1/not-last, 1 selects new_word2/last.
  %w_beat_idx_next = comb.mux %aw_accept, %f, %w_beat_idx_after : i1
  %w_beat_idx_after = comb.mux %w_beat_accept, %t, %w_beat_idx_q : i1
  %w_beat_idx_q = seq.compreg %w_beat_idx_next, %clk_i reset %rst, %f : i1
  %wdata_sel = comb.mux %w_beat_idx_q, %new_word2, %new_word1 : i64

  // Beat counter (shared by both read bursts -- mutually exclusive in time)
  // resets on either burst's AR accept, then increments per accepted beat.
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
  %recv_r1 = comb.and %rvalid, %in_p1 : i1
  %recv_r2 = comb.and %rvalid, %in_p6 : i1
  %en_beat0 = comb.and %recv_r1, %is_beat0 : i1
  %en_beat1 = comb.and %recv_r1, %is_beat1 : i1
  %en_beat2 = comb.and %recv_r1, %is_beat2 : i1
  %en_beat3 = comb.and %recv_r1, %is_beat3 : i1
  %en_after0 = comb.and %recv_r2, %is_beat0 : i1
  %en_after1 = comb.and %recv_r2, %is_beat1 : i1
  %en_after2 = comb.and %recv_r2, %is_beat2 : i1
  %en_after3 = comb.and %recv_r2, %is_beat3 : i1

  %beat0_next = comb.mux %en_beat0, %rdata, %beat0_q : i64
  %beat0_q = seq.compreg %beat0_next, %clk_i reset %rst, %c64 : i64
  %beat1_next = comb.mux %en_beat1, %rdata, %beat1_q : i64
  %beat1_q = seq.compreg %beat1_next, %clk_i reset %rst, %c64 : i64
  %beat2_next = comb.mux %en_beat2, %rdata, %beat2_q : i64
  %beat2_q = seq.compreg %beat2_next, %clk_i reset %rst, %c64 : i64
  %beat3_next = comb.mux %en_beat3, %rdata, %beat3_q : i64
  %beat3_q = seq.compreg %beat3_next, %clk_i reset %rst, %c64 : i64
  %after_beat0_next = comb.mux %en_after0, %rdata, %after_beat0_q : i64
  %after_beat0_q = seq.compreg %after_beat0_next, %clk_i reset %rst, %c64 : i64
  %after_beat1_next = comb.mux %en_after1, %rdata, %after_beat1_q : i64
  %after_beat1_q = seq.compreg %after_beat1_next, %clk_i reset %rst, %c64 : i64
  %after_beat2_next = comb.mux %en_after2, %rdata, %after_beat2_q : i64
  %after_beat2_q = seq.compreg %after_beat2_next, %clk_i reset %rst, %c64 : i64
  %after_beat3_next = comb.mux %en_after3, %rdata, %after_beat3_q : i64
  %after_beat3_q = seq.compreg %after_beat3_next, %clk_i reset %rst, %c64 : i64

  hw.output %port, %done_q, %beat0_q, %beat1_q, %beat2_q, %beat3_q,
            %after_beat0_q, %after_beat1_q, %after_beat2_q, %after_beat3_q
    : !port, i1, i64, i64, i64, i64, i64, i64, i64, i64
}

hw.module @sub_module(in %clk_i : !seq.clock, in %rst_ni : i1, in %axi : !port) {
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
      ar %idle r %r, %state_q : !port
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
  %mgr, %done, %beat0, %beat1, %beat2, %beat3,
  %after_beat0, %after_beat1, %after_beat2, %after_beat3 =
    hw.instance "mgr" @mgr_module(clk_i: %clk_i: !seq.clock, rst_ni: %rst_ni: i1)
      -> (axi: !port, done: i1, beat0: i64, beat1: i64, beat2: i64, beat3: i64,
          after_beat0: i64, after_beat1: i64, after_beat2: i64, after_beat3: i64)
  %sub = axi4.xbar %clk_i, %rst_ni mgrs %mgr : (!port) -> (!port)
  hw.instance "sub" @sub_module(clk_i: %clk_i: !seq.clock, rst_ni: %rst_ni: i1,
                                axi: %sub: !port) -> ()
  hw.output
}
