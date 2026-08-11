// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true | FileCheck %s

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !port)
hw.module.extern @Subordinate(in %axi : !port)

// The external module is unchanged by the option - the wrapper is verilog
// hanging off it, in its own file
// CHECK:       hw.module.extern @axi_cdc_a32_d64_i4(
// CHECK-SAME:    in %src_clk_i : !seq.clock, in %dst_clk_i : !seq.clock,
// CHECK-SAME:    in %rst_ni : i1,
// CHECK-SAME:    out sub0_rready : i1)
// CHECK-SAME:    attributes {source = @axi_cdc_a32_d64_i4.sv}
// CHECK:      sv.verbatim.source @axi_cdc_a32_d64_i4.sv

// The wrapper takes a clock per side, and the single reset
// CHECK-SAME:   module axi_cdc_a32_d64_i4 (\0A
// CHECK-SAME:     input  logic src_clk_i,\0A
// CHECK-SAME:     input  logic dst_clk_i,\0A
// CHECK-SAME:     input  logic rst_ni,\0A
// CHECK-SAME:     input  axi_cdc_a32_d64_i4_mgr_aw_t mgr0_aw,\0A
// CHECK-SAME:     output logic sub0_rready\0A);

// One req/resp pair per side, bridged to the ports of that side
// CHECK-SAME:   assign slv_req.aw = '{id: mgr0_aw.id,
// CHECK-SAME:   assign sub0_aw = '{id: mst_req.aw.id,

// The upstream face is the source domain and the downstream face the
// destination, and both sides take the one reset - a crossing must not reset one
// side without the other
// CHECK-SAME:   axi_cdc #(\0A
// CHECK-SAME:     .LogDepth   (1),\0A
// CHECK-SAME:     .SyncStages (2)\0A
// CHECK-SAME:   ) i_cdc (\0A
// CHECK-SAME:     .src_clk_i  (src_clk_i),\0A
// CHECK-SAME:     .src_rst_ni (rst_ni),\0A
// CHECK-SAME:     .src_req_i  (slv_req),\0A
// CHECK-SAME:     .dst_clk_i  (dst_clk_i),\0A
// CHECK-SAME:     .dst_rst_ni (rst_ni),\0A
// CHECK-SAME:     .dst_req_o  (mst_req),\0A

// CHECK-SAME:  output_file = #hw.output_file<"axi_cdc_a32_d64_i4.sv">
// CHECK-SAME:  verilogName = "axi_cdc_a32_d64_i4"

// CHECK-LABEL: hw.module @Crossing(
// CHECK:         hw.instance "cdc0" @axi_cdc_a32_d64_i4(
hw.module @Crossing(in %clk : !seq.clock, in %other_clk : !seq.clock,
                    in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !port)
  %cdc = axi4.cdc from %clk to %other_clk, %rst_ni, %m : !port
  hw.instance "sub" @Subordinate(axi: %cdc: !port) -> ()
}
