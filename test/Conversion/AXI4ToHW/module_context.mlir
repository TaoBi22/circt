// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s --implicit-check-not=unrealized_conversion_cast --implicit-check-not=axi4.
// RUN: circt-opt %s --lower-axi4-to-hw --export-verilog | FileCheck %s --check-prefix=VERILOG

// A network living inside an hw.module lowers in place: the instances land in the
// module body and the module's i1 clock port feeds them directly (the
// i1 -> !axi4.clock cast is short-circuited, leaving no unrealized casts), so the
// whole design is export-verilog-able - not just the xbar wrapper.

// CHECK-LABEL: hw.module @AXITop(in %clk_i : i1)
// CHECK-DAG: hw.instance "mgr_module{{.*}}" @mgr_module(clk: %clk_i
// CHECK-DAG: hw.instance "sub_module{{.*}}" @sub_module(clk: %clk_i
// CHECK-DAG: hw.instance "xbar_{{[0-9]+}}" @axi_xbar_1u1d_a32_d64_i4_o4(clk_i: %clk_i
// CHECK: hw.output

// The full design (top module + wrapper) emits as SystemVerilog.
// VERILOG: module AXITop(
// VERILOG: input clk_i
// VERILOG-DAG: mgr_module mgr_module{{.*}} (
// VERILOG-DAG: sub_module sub_module{{.*}} (
// VERILOG-DAG: axi_xbar_1u1d_a32_d64_i4_o4 xbar_{{[0-9]+}} (
// VERILOG: module axi_xbar_1u1d_a32_d64_i4_o4 (

hw.module.extern @mgr_module(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module(in %clk : i1, in %s_axi_s0_awid : i4, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i4, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i4, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i4, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)

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
