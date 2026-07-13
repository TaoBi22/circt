// manager -> xbar -> subordinate, inside a top module. Emits AXITop + one
// axi_xbar_1u1d wrapper. mgr_module issues a single AXI4 read to address 0;
// sub_module is a tiny 4-word ROM. See sim/tb_axitop.sv for the waveform
// testbench that exercises this end to end.
hw.module @mgr_module(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1, out done : i1, out captured_rdata : i64) {
  %f = hw.constant 0 : i1
  %t = hw.constant 1 : i1
  %c2 = hw.constant 0 : i2
  %c3 = hw.constant 0 : i3
  %c4 = hw.constant 0 : i4
  %c8 = hw.constant 0 : i8
  %c32 = hw.constant 0 : i32
  %c64 = hw.constant 0 : i64
  %arsize = hw.constant 3 : i3

  %clock = seq.to_clock %clk

  // Single-shot AR issue: arvalid until accepted, then never again.
  %state_q = seq.compreg %state_next, %clock reset %f, %f : i1
  %issuing = comb.xor %state_q, %t : i1
  %ar_accept = comb.and %issuing, %m_axi_m0_arready : i1
  %state_next = comb.mux %ar_accept, %t, %state_q : i1

  // Capture the R response once accepted.
  %r_accept = comb.and %state_q, %m_axi_m0_rvalid : i1
  %done_next = comb.mux %r_accept, %t, %done_q : i1
  %done_q = seq.compreg %done_next, %clock reset %f, %f : i1
  %rdata_next = comb.mux %r_accept, %m_axi_m0_rdata, %captured_rdata_q : i64
  %captured_rdata_q = seq.compreg %rdata_next, %clock reset %f, %c64 : i64

  hw.output %c4, %c32, %c8, %c3, %c2, %f, %c4, %c3, %c4, %c4, %f,
            %c64, %c8, %f, %f,
            %f,
            %c4, %c32, %c8, %arsize, %c2, %f, %c4, %c3, %c4, %c4, %issuing,
            %state_q,
            %done_q, %captured_rdata_q
    : i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i64, i8, i1, i1, i1,
      i4, i32, i8, i3, i2, i1, i4, i3, i4, i4, i1, i1, i1, i64
}
hw.module @sub_module(in %clk : i1, in %s_axi_s0_awid : i4, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i4, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i4, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i4, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1) {
  %false = hw.constant false
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c0_i4 = hw.constant 0 : i4
  %c0_i64 = hw.constant 0 : i64

  %clock = seq.to_clock %clk

  // ROM contents (depth 4, indexed by araddr[4:3]); only word 0 is ever
  // fetched by mgr_module - the rest just prove it's a real memory.
  %rom0 = hw.constant 0xCAFEF00DCAFEF00D : i64
  %rom1 = hw.constant 0xDEADBEEFDEADBEEF : i64
  %rom2 = hw.constant 0xFACEFEEDFACEFEED : i64
  %rom3 = hw.constant 0x8BADF00D8BADF00D : i64

  // 2-state FSM: state_q = 0 IDLE (accept AR) / 1 RESP (drive R channel).
  %state_q = seq.compreg %state_next, %clock reset %false, %false : i1
  %idle = comb.xor %state_q, %true : i1
  %accept = comb.and %s_axi_s0_arvalid, %idle : i1
  %handshake_r = comb.and %state_q, %s_axi_s0_rready : i1
  %back_to_idle = comb.mux %handshake_r, %false, %state_q : i1
  %state_next = comb.mux %accept, %true, %back_to_idle : i1

  %idx = comb.extract %s_axi_s0_araddr from 3 : (i32) -> i2
  %idx0 = comb.extract %idx from 0 : (i2) -> i1
  %idx1 = comb.extract %idx from 1 : (i2) -> i1
  %sel_lo = comb.mux %idx0, %rom1, %rom0 : i64
  %sel_hi = comb.mux %idx0, %rom3, %rom2 : i64
  %selected = comb.mux %idx1, %sel_hi, %sel_lo : i64

  %rid_next = comb.mux %accept, %s_axi_s0_arid, %rid_q : i4
  %rid_q = seq.compreg %rid_next, %clock reset %false, %c0_i4 : i4
  %rdata_next = comb.mux %accept, %selected, %rdata_q : i64
  %rdata_q = seq.compreg %rdata_next, %clock reset %false, %c0_i64 : i64

  hw.output %false, %false, %c0_i4, %c0_i2, %false, %idle, %rid_q, %rdata_q, %c0_i2, %true, %state_q
    : i1, i1, i4, i2, i1, i1, i4, i64, i2, i1, i1
}

hw.module @AXITop(in %clk_i : i1) {
  %clk = builtin.unrealized_conversion_cast %clk_i : i1 to !axi4.clock
  %mnode = axi4.node @mgr_module : !axi4.node
  %snode = axi4.node @sub_module : !axi4.node
  %mgr = axi4.manager_port %clk node %mnode {
    port_mapping = #axi4.port_wires<"clk", "m0">,
    access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
    outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
  } : !axi4.port<32, 64, 4>
  %xbar = axi4.xbar %clk mgrs %mgr : (!axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 4>
  axi4.subordinate_port %xbar, %clk node %snode {
    port_mapping = #axi4.port_wires<"clk", "s0">,
    access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
    outstanding_requests = 4 : ui32
  } : !axi4.port<32, 64, 4>
  hw.output
}
