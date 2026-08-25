// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true | FileCheck %s

!mem_port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>, <fixed, len = 1>>>>, outstanding_writes = 4, outstanding_reads = 4>

// CHECK:       hw.module.extern @axi_to_mem_a32_d64_i4(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mem_rvalid_i : i1, in %mem_rdata_i : i64,
// CHECK-SAME:    out mem_we_o : i1)
// CHECK-SAME:    attributes {source = @axi_to_mem_a32_d64_i4.sv}
// CHECK:      sv.verbatim.source @axi_to_mem_a32_d64_i4.sv

// The memory is not an AXI face, so only the upstream manager's channels get
// payload structs - the last of them runs straight into the module header
// CHECK-SAME:   } axi_to_mem_a32_d64_i4_mgr_r_t;\0A\0Amodule axi_to_mem_a32_d64_i4 (\0A
// CHECK-SAME:     input  logic clk_i,\0A
// CHECK-SAME:     input  logic mem_rvalid_i,\0A
// CHECK-SAME:     input  axi_to_mem_a32_d64_i4_data_t mem_rdata_i,\0A
// CHECK-SAME:     output logic mem_req_o,\0A
// CHECK-SAME:     output axi_to_mem_a32_d64_i4_addr_t mem_addr_o,\0A
// CHECK-SAME:     output axi_to_mem_a32_d64_i4_data_t mem_wdata_o,\0A
// CHECK-SAME:     output axi_to_mem_a32_d64_i4_strb_t mem_strb_o,\0A
// CHECK-SAME:     output logic mem_we_o,\0A
// CHECK-SAME:     input  axi_to_mem_a32_d64_i4_mgr_aw_t mgr0_aw,\0A
// CHECK-SAME:     input  logic mgr0_rready\0A);

// CHECK-SAME:   assign slv_req.aw = '{id: mgr0_aw.id,

// One bank carries the port's full width, the memory grants every request, and
// the atomics PULP derives are dropped
// CHECK-SAME:   axi_to_mem #(\0A
// CHECK-SAME:     .AddrWidth  (32),\0A
// CHECK-SAME:     .DataWidth  (64),\0A
// CHECK-SAME:     .IdWidth    (4),\0A
// CHECK-SAME:     .NumBanks   (1),\0A
// CHECK-SAME:     .BufDepth   (1)\0A
// CHECK-SAME:   ) i_to_mem (\0A
// CHECK-SAME:     .mem_gnt_i    (1'b1),\0A
// CHECK-SAME:     .mem_atop_o   (),\0A
// CHECK-SAME:     .mem_rdata_i  (mem_rdata_i)\0A

// CHECK-SAME:  output_file = #hw.output_file<"axi_to_mem_a32_d64_i4.sv">
// CHECK-SAME:  verilogName = "axi_to_mem_a32_d64_i4"

// CHECK-LABEL: hw.module @ToMem(
// CHECK:         hw.instance "to_mem0" @axi_to_mem_a32_d64_i4(
// CHECK-SAME:      mem_rvalid_i: %rvalid: i1, mem_rdata_i: %rdata: i64,
hw.module @ToMem(in %clk : !seq.clock, in %rst_ni : i1, in %port : !mem_port,
                 in %rvalid : i1, in %rdata : i64,
                 out valid : i1, out addr : i32, out wdata : i64,
                 out strb : i8, out we : i1) {
  %valid, %addr, %wdata, %strb, %we = axi4.to_mem %clk, %rst_ni, %port read %rvalid, %rdata : !mem_port
  hw.output %valid, %addr, %wdata, %strb, %we : i1, i32, i64, i8, i1
}
