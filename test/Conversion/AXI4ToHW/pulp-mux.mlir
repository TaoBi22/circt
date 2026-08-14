// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true | FileCheck %s

!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 2>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 2>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 4>

// CHECK:       hw.module.extern @axi_mux_2u_a32_d64_i4_o5(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    out sub0_rready : i1)
// CHECK-SAME:    attributes {source = @axi_mux_2u_a32_d64_i4_o5.sv}
// CHECK:      sv.verbatim.source @axi_mux_2u_a32_d64_i4_o5.sv

// The managers and the subordinate route over an ID width each, so the wrapper
// carries a channel and req/resp type per side
// CHECK-SAME:   typedef logic [4-1:0] axi_mux_2u_a32_d64_i4_o5_slv_id_t;\0A
// CHECK-SAME:   typedef logic [5-1:0] axi_mux_2u_a32_d64_i4_o5_mst_id_t;\0A
// CHECK-SAME:   `AXI_TYPEDEF_ALL(axi_mux_2u_a32_d64_i4_o5_slv,
// CHECK-SAME:   `AXI_TYPEDEF_ALL(axi_mux_2u_a32_d64_i4_o5_mst,

// A face per manager, and one for the subordinate
// CHECK-SAME:   module axi_mux_2u_a32_d64_i4_o5 (\0A
// CHECK-SAME:     input  logic clk_i,\0A
// CHECK-SAME:     input  logic rst_ni,\0A
// CHECK-SAME:     input  axi_mux_2u_a32_d64_i4_o5_mgr_aw_t mgr0_aw,\0A
// CHECK-SAME:     input  axi_mux_2u_a32_d64_i4_o5_mgr_aw_t mgr1_aw,\0A
// CHECK-SAME:     output axi_mux_2u_a32_d64_i4_o5_sub_aw_t sub0_aw,\0A
// CHECK-SAME:     output logic sub0_rready\0A);

// CHECK-SAME:   axi_mux_2u_a32_d64_i4_o5_slv_req_t  [2-1:0] slv_req;\0A
// CHECK-SAME:   axi_mux_2u_a32_d64_i4_o5_mst_req_t  [1-1:0] mst_req;\0A
// CHECK-SAME:   assign slv_req[1].aw = '{id: mgr1_aw.id,
// CHECK-SAME:   assign sub0_aw = '{id: mst_req[0].aw.id,

// PULP tags the managers itself, over the wider IDs of its one downstream port
// CHECK-SAME:   axi_mux #(\0A
// CHECK-SAME:     .SlvAxiIDWidth (4),\0A
// CHECK-SAME:     .slv_aw_chan_t (axi_mux_2u_a32_d64_i4_o5_slv_aw_chan_t),\0A
// CHECK-SAME:     .mst_resp_t    (axi_mux_2u_a32_d64_i4_o5_mst_resp_t),\0A
// CHECK-SAME:     .NoSlvPorts    (2),\0A
// CHECK-SAME:     .MaxWTrans     (4),\0A
// CHECK-SAME:     .FallThrough   (1'b0)\0A
// CHECK-SAME:   ) i_mux (\0A
// CHECK-SAME:     .test_i      (1'b0),\0A
// CHECK-SAME:     .slv_reqs_i  (slv_req),\0A
// CHECK-SAME:     .slv_resps_o (slv_resp),\0A
// CHECK-SAME:     .mst_req_o   (mst_req[0]),\0A
// CHECK-SAME:     .mst_resp_i  (mst_resp[0])\0A

// CHECK-SAME:  output_file = #hw.output_file<"axi_mux_2u_a32_d64_i4_o5.sv">
// CHECK-SAME:  verilogName = "axi_mux_2u_a32_d64_i4_o5"

// CHECK-LABEL: hw.module @Mux(
// CHECK:         hw.instance "mux0" @axi_mux_2u_a32_d64_i4_o5(
hw.module @Mux(in %clk : !seq.clock, in %rst_ni : i1, in %a : !lo, in %b : !hi,
               out downstream : !sub) {
  %downstream = axi4.mux %clk, %rst_ni, %a, %b : (!lo, !hi) -> !sub
  hw.output %downstream : !sub
}
