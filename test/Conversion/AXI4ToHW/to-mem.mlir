// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s --implicit-check-not=axi4.

!mem_port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Only the port face explodes; the memory interface is carried through as the
// signals it already is, taken and driven around it
// CHECK-LABEL: hw.module.extern @axi_to_mem_a32_d64_i4(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mem_rvalid_i : i1, in %mem_rdata_i : i64,
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %mgr0_rready : i1,
// CHECK-SAME:    out mgr0_awready : i1,
// CHECK-SAME:    out mgr0_rvalid : i1,
// CHECK-SAME:    out mem_req_o : i1, out mem_addr_o : i32,
// CHECK-SAME:    out mem_wdata_o : i64, out mem_strb_o : i8,
// CHECK-SAME:    out mem_we_o : i1)

// The enclosing module's own port explodes to mirror the instance's
// CHECK-LABEL: hw.module @ToMem(
// CHECK-SAME:    in %port_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    in %rvalid : i1, in %rdata : i64,
// CHECK-SAME:    out port_awready : i1,
// CHECK-SAME:    out valid : i1, out addr : i32, out wdata : i64,
// CHECK-SAME:    out strb : i8, out we : i1)
hw.module @ToMem(in %clk : !seq.clock, in %rst_ni : i1, in %port : !mem_port,
                 in %rvalid : i1, in %rdata : i64,
                 out valid : i1, out addr : i32, out wdata : i64,
                 out strb : i8, out we : i1) {
  // CHECK: %to_mem0.mgr0_awready, {{.*}} = hw.instance "to_mem0" @axi_to_mem_a32_d64_i4(
  // CHECK-SAME: clk_i: %clk: !seq.clock, rst_ni: %rst_ni: i1
  // CHECK-SAME: mem_rvalid_i: %rvalid: i1, mem_rdata_i: %rdata: i64
  // CHECK-SAME: mgr0_aw: %port_aw:
  %valid, %addr, %wdata, %strb, %we = axi4.to_mem %clk, %rst_ni, %port read %rvalid, %rdata : !mem_port

  // CHECK: hw.output %to_mem0.mgr0_awready,
  // CHECK-SAME: %to_mem0.mem_req_o, %to_mem0.mem_addr_o,
  hw.output %valid, %addr, %wdata, %strb, %we : i1, i32, i64, i8, i1
}
