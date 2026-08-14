// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true | FileCheck %s

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 2>
!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 2>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 2>

// CHECK:       hw.module.extern @axi_demux_2d_a32_d64_i4(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    out sub1_rready : i1)
// CHECK-SAME:    attributes {source = @axi_demux_2d_a32_d64_i4.sv}
// CHECK:      sv.verbatim.source @axi_demux_2d_a32_d64_i4.sv

// Every face carries the same widths, so one set of channel types serves them all
// CHECK-SAME:   typedef logic [4-1:0] axi_demux_2d_a32_d64_i4_wid_t;\0A
// CHECK-SAME:   `AXI_TYPEDEF_REQ_T(axi_demux_2d_a32_d64_i4_req_t,

// A face per downstream port, bridged to the array PULP drives
// CHECK-SAME:   module axi_demux_2d_a32_d64_i4 (\0A
// CHECK-SAME:     output axi_demux_2d_a32_d64_i4_sub_aw_t sub0_aw,\0A
// CHECK-SAME:     output axi_demux_2d_a32_d64_i4_sub_aw_t sub1_aw,\0A
// CHECK-SAME:     output logic sub1_rready\0A);
// CHECK-SAME:   axi_demux_2d_a32_d64_i4_req_t  slv_req;\0A
// CHECK-SAME:   axi_demux_2d_a32_d64_i4_req_t  [2-1:0] mst_req;\0A
// CHECK-SAME:   assign sub1_aw = '{id: mst_req[1].aw.id,

// PULP is handed a port index per request rather than an address map, so the
// wrapper decodes the windows into one. A window's last address is inclusive and
// PULP's end address is not, so the rules end one past the window.
// CHECK-SAME:   typedef struct packed { int unsigned idx; axi_demux_2d_a32_d64_i4_addr_t start_addr; axi_demux_2d_a32_d64_i4_addr_t end_addr; } rule_t;\0A
// CHECK-SAME:   localparam rule_t [2-1:0] AddrMap = '{\0A
// CHECK-SAME:     '{idx: 0, start_addr: 32'h0, end_addr: 32'h1000},\0A
// CHECK-SAME:     '{idx: 1, start_addr: 32'h1000, end_addr: 32'h2000}\0A
// CHECK-SAME:   };\0A
// CHECK-SAME:   typedef logic [1-1:0] select_t;\0A
// CHECK-SAME:   select_t aw_select, ar_select;\0A

// A decoder per address channel, defaulting to port 0 for an address no window
// covers
// CHECK-SAME:   addr_decode #(\0A
// CHECK-SAME:     .NoIndices        (2),\0A
// CHECK-SAME:     .NoRules          (2),\0A
// CHECK-SAME:     .rule_t           (rule_t)\0A
// CHECK-SAME:   ) i_aw_decode (\0A
// CHECK-SAME:     .addr_i           (slv_req.aw.addr),\0A
// CHECK-SAME:     .addr_map_i       (AddrMap),\0A
// CHECK-SAME:     .idx_o            (aw_select),\0A
// CHECK-SAME:     .en_default_idx_i (1'b1),\0A
// CHECK-SAME:     .default_idx_i    ('0)\0A
// CHECK-SAME:   ) i_ar_decode (\0A
// CHECK-SAME:     .addr_i           (slv_req.ar.addr),\0A
// CHECK-SAME:     .idx_o            (ar_select),\0A

// CHECK-SAME:   axi_demux #(\0A
// CHECK-SAME:     .AxiIdWidth  (4),\0A
// CHECK-SAME:     .AtopSupport (1'b1),\0A
// CHECK-SAME:     .aw_chan_t  (axi_demux_2d_a32_d64_i4_aw_chan_t),\0A
// CHECK-SAME:     .NoMstPorts  (2),\0A
// CHECK-SAME:     .MaxTrans    (4),\0A
// CHECK-SAME:     .AxiLookBits (4),\0A
// CHECK-SAME:     .UniqueIds   (1'b0)\0A
// CHECK-SAME:   ) i_demux (\0A
// CHECK-SAME:     .slv_aw_select_i (aw_select),\0A
// CHECK-SAME:     .slv_ar_select_i (ar_select),\0A
// CHECK-SAME:     .mst_reqs_o      (mst_req),\0A
// CHECK-SAME:     .mst_resps_i     (mst_resp)\0A

// CHECK-SAME:  output_file = #hw.output_file<"axi_demux_2d_a32_d64_i4.sv">
// CHECK-SAME:  verilogName = "axi_demux_2d_a32_d64_i4"

// CHECK-LABEL: hw.module @Demux(
// CHECK:         hw.instance "demux0" @axi_demux_2d_a32_d64_i4(
hw.module @Demux(in %clk : !seq.clock, in %rst_ni : i1, in %upstream : !mgr,
                 out lo : !lo, out hi : !hi) {
  %a, %b = axi4.demux %clk, %rst_ni, %upstream : (!mgr) -> (!lo, !hi)
  hw.output %a, %b : !lo, !hi
}
