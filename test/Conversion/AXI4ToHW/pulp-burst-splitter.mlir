// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true | FileCheck %s

!burstty = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 2>
!beats = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>>, outstanding_writes = 16, outstanding_reads = 8>

// CHECK:       hw.module.extern @axi_burst_splitter_a32_d64_i4(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    out sub0_rready : i1)
// CHECK-SAME:    attributes {source = @axi_burst_splitter_a32_d64_i4.sv}
// CHECK:      sv.verbatim.source @axi_burst_splitter_a32_d64_i4.sv

// CHECK-SAME:   `AXI_TYPEDEF_REQ_T(axi_burst_splitter_a32_d64_i4_req_t, axi_burst_splitter_a32_d64_i4_aw_chan_t,
// CHECK-SAME:   module axi_burst_splitter_a32_d64_i4 (\0A
// CHECK-SAME:     input  logic clk_i,\0A
// CHECK-SAME:     input  logic rst_ni,\0A
// CHECK-SAME:     input  axi_burst_splitter_a32_d64_i4_mgr_aw_t mgr0_aw,\0A
// CHECK-SAME:     output logic sub0_rready\0A);

// CHECK-SAME:   assign slv_req.aw = '{id: mgr0_aw.id,
// CHECK-SAME:   assign sub0_aw = '{id: mst_req.aw.id,

// CHECK-SAME:   axi_burst_splitter #(\0A
// CHECK-SAME:     .MaxReadTxns  (2),\0A
// CHECK-SAME:     .MaxWriteTxns (4),\0A
// CHECK-SAME:     .AddrWidth    (32),\0A
// CHECK-SAME:     .DataWidth    (64),\0A
// CHECK-SAME:     .IdWidth      (4),\0A
// CHECK-SAME:     .UserWidth    (1),\0A
// CHECK-SAME:   ) i_burst_splitter (\0A
// CHECK-SAME:     .slv_req_i  (slv_req),\0A
// CHECK-SAME:     .mst_resp_i (mst_resp)\0A

// CHECK-SAME:  output_file = #hw.output_file<"axi_burst_splitter_a32_d64_i4.sv">
// CHECK-SAME:  verilogName = "axi_burst_splitter_a32_d64_i4"

// CHECK-LABEL: hw.module @BurstSplitter(
// CHECK:         hw.instance "burst_splitter0" @axi_burst_splitter_a32_d64_i4(
hw.module @BurstSplitter(in %clk : !seq.clock, in %rst_ni : i1,
                         in %upstream : !burstty, out downstream : !beats) {
  %split = axi4.burst_splitter %clk, %rst_ni, %upstream : (!burstty) -> !beats
  hw.output %split : !beats
}
