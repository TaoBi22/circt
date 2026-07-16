// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s

// Crossbar address-map derivation for chained and fan-out topologies. Each
// crossbar lowers to a PULP axi_xbar wrapper whose baked-in AddrMap has one
// rule per downstream master port; a chained crossbar contributes the union of
// the ranges reachable through it. (The full wrapper/connectivity form is
// checked in xbar.mlir; here we only assert the derived address rules.)

// Scenario 1 - chained xbars: mgr(i4) -> A(i4->i5) -> B(i5->i6) -> sub@[0,0x1000).
// Both crossbars route the subordinate's window; the upstream crossbar (A) finds
// its range only by recursing through the downstream crossbar (B).
// CHECK-DAG: @axi_xbar_1u1d_a32_d64_i4_o5_source {{.*}}AddrMap{{.*}}idx: 0, start_addr: 32'h0, end_addr: 32'h1000}
// CHECK-DAG: @axi_xbar_1u1d_a32_d64_i5_o6_source {{.*}}AddrMap{{.*}}idx: 0, start_addr: 32'h0, end_addr: 32'h1000}

// Scenario 2 - fan-out: mgr(i4) -> M(i4->i5) fans out to a direct subordinate
// (@[0,0x1000)) and a chained crossbar N(i5->i5) -> sub@[0x2000,0x3000). M gets
// one disjoint rule per master port; N routes its own subordinate's window.
// CHECK-DAG: @axi_xbar_1u2d_a32_d64_i4_o5_source {{.*}}idx: 0, start_addr: 32'h0, end_addr: 32'h1000},{{.*}}idx: 1, start_addr: 32'h2000, end_addr: 32'h3000}
// CHECK-DAG: @axi_xbar_1u1d_a32_d64_i5_o5_source {{.*}}AddrMap{{.*}}idx: 0, start_addr: 32'h2000, end_addr: 32'h3000}

hw.module.extern @mgr_i4_m0(in %clk : i1, in %rst_ni : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awuser : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wuser : i4, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_buser : i4, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_aruser : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_ruser : i4, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_i6_s0(in %clk : i1, in %rst_ni : i1, in %s_axi_s0_awid : i6, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awuser : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wuser : i4, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i6, out s_axi_s0_bresp : i2, out s_axi_s0_buser : i4, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i6, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_aruser : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i6, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_ruser : i4, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)
hw.module.extern @sub_i5_s0(in %clk : i1, in %rst_ni : i1, in %s_axi_s0_awid : i5, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awuser : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wuser : i4, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i5, out s_axi_s0_bresp : i2, out s_axi_s0_buser : i4, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i5, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_aruser : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i5, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_ruser : i4, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)
hw.module.extern @sub_i5_s1(in %clk : i1, in %rst_ni : i1, in %s_axi_s1_awid : i5, in %s_axi_s1_awaddr : i32, in %s_axi_s1_awlen : i8, in %s_axi_s1_awsize : i3, in %s_axi_s1_awburst : i2, in %s_axi_s1_awlock : i1, in %s_axi_s1_awcache : i4, in %s_axi_s1_awprot : i3, in %s_axi_s1_awqos : i4, in %s_axi_s1_awregion : i4, in %s_axi_s1_awuser : i4, in %s_axi_s1_awvalid : i1, out s_axi_s1_awready : i1, in %s_axi_s1_wdata : i64, in %s_axi_s1_wstrb : i8, in %s_axi_s1_wlast : i1, in %s_axi_s1_wuser : i4, in %s_axi_s1_wvalid : i1, out s_axi_s1_wready : i1, out s_axi_s1_bid : i5, out s_axi_s1_bresp : i2, out s_axi_s1_buser : i4, out s_axi_s1_bvalid : i1, in %s_axi_s1_bready : i1, in %s_axi_s1_arid : i5, in %s_axi_s1_araddr : i32, in %s_axi_s1_arlen : i8, in %s_axi_s1_arsize : i3, in %s_axi_s1_arburst : i2, in %s_axi_s1_arlock : i1, in %s_axi_s1_arcache : i4, in %s_axi_s1_arprot : i3, in %s_axi_s1_arqos : i4, in %s_axi_s1_arregion : i4, in %s_axi_s1_aruser : i4, in %s_axi_s1_arvalid : i1, out s_axi_s1_arready : i1, out s_axi_s1_rid : i5, out s_axi_s1_rdata : i64, out s_axi_s1_rresp : i2, out s_axi_s1_rlast : i1, out s_axi_s1_ruser : i4, out s_axi_s1_rvalid : i1, in %s_axi_s1_rready : i1)

// Scenario 1: chained crossbars.
%c0 = unrealized_conversion_cast to !axi4.clock
%r0 = unrealized_conversion_cast to !axi4.reset
%mn0 = axi4.node @mgr_i4_m0 : !axi4.node
%sn0 = axi4.node @sub_i6_s0 : !axi4.node
%mgr0 = axi4.manager_port %c0, %r0 node %mn0 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
%xa = axi4.xbar %c0, %r0 mgrs %mgr0 : (!axi4.port<32, 64, 4, 4, 4>) -> !axi4.port<32, 64, 5, 5, 4>
%xb = axi4.xbar %c0, %r0 mgrs %xa : (!axi4.port<32, 64, 5, 5, 4>) -> !axi4.port<32, 64, 6, 6, 4>
axi4.subordinate_port %xb, %c0, %r0 node %sn0 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 6, 6, 4>

// Scenario 2: fan-out to a direct subordinate and a chained crossbar. The
// chained branch is declared first so the direct subordinate is master port 0.
%mn1 = axi4.node @mgr_i4_m0 : !axi4.node
%sn1 = axi4.node @sub_i5_s0 : !axi4.node
%sn2 = axi4.node @sub_i5_s1 : !axi4.node
%mgr1 = axi4.manager_port %c0, %r0 node %mn1 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
%xm = axi4.xbar %c0, %r0 mgrs %mgr1 : (!axi4.port<32, 64, 4, 4, 4>) -> !axi4.port<32, 64, 5, 5, 4>
%xn = axi4.xbar %c0, %r0 mgrs %xm : (!axi4.port<32, 64, 5, 5, 4>) -> !axi4.port<32, 64, 5, 5, 4>
axi4.subordinate_port %xn, %c0, %r0 node %sn2 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s1">,
  access = [#axi4.window<base = 8192, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 5, 5, 4>
axi4.subordinate_port %xm, %c0, %r0 node %sn1 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 5, 5, 4>
