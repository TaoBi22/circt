// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s

// A single manager->subordinate node pair, wired directly (no crossbar). The
// port type carries a single user width (4), which flows through the flat
// port_wires ports on every channel.

// CHECK: hw.module.extern @mgr_module
// CHECK: hw.module.extern @sub_module

// The B response payload the manager consumes is unpacked from a struct
// (%[[BSTRUCT]], packed from the subordinate's outputs further down). Its user
// field is i4, matching the port's user width.
// CHECK: %[[BID:.+]] = hw.struct_extract %[[BSTRUCT:.+]]["id"] : !hw.struct<id: i4, resp: i2, user: i4>
// CHECK: %[[BUSER:.+]] = hw.struct_extract %[[BSTRUCT]]["user"] : !hw.struct<id: i4, resp: i2, user: i4>

// The R payload is unpacked likewise; its user field is i4.
// CHECK: hw.struct_extract %{{.+}}["user"] : !hw.struct<id: i4, data: i64, resp: i2, last: i1, user: i4>

// The abstract clock and reset are each lowered to a shared i1 module port.
// CHECK-DAG: %[[CLK:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.clock to i1
// CHECK-DAG: %[[RST:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.reset to i1

// The manager instance takes the shared clock/reset; request-ready comes back
// from the subordinate (awready <- sub), and the response arrives from the
// subordinate: valid straight through (bvalid <- sub) and payload via the
// unpacked B struct above (bid/buser <- %[[BID]]/%[[BUSER]]).
// CHECK: hw.instance "mgr_module_0" @mgr_module(
// CHECK-SAME: clk: %[[CLK]]: i1
// CHECK-SAME: rst_ni: %[[RST]]: i1
// CHECK-SAME: m_axi_m0_awready: %sub_module_1.s_axi_s0_awready: i1
// CHECK-SAME: m_axi_m0_bid: %[[BID]]: i4
// CHECK-SAME: m_axi_m0_buser: %[[BUSER]]: i4
// CHECK-SAME: m_axi_m0_bvalid: %sub_module_1.s_axi_s0_bvalid: i1

// The AW request payload is packed from the manager's flat outputs (user: i4)
// then unpacked to drive the subordinate.
// CHECK: %[[AW:.+]] = hw.struct_create (%mgr_module_0.m_axi_m0_awid, {{.*}}%mgr_module_0.m_axi_m0_awuser) : !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i4>
// CHECK: %[[AWID:.+]] = hw.struct_extract %[[AW]]["id"]
// CHECK: %[[AWUSER:.+]] = hw.struct_extract %[[AW]]["user"]

// The subordinate instance takes the shared clock/reset; the request arrives
// from the manager: payload via the unpacked AW struct (awid/awuser <-
// %[[AWID]]/%[[AWUSER]]) and valid straight through (awvalid <- mgr). Response-
// ready comes back from the manager (bready <- mgr).
// CHECK: hw.instance "sub_module_1" @sub_module(
// CHECK-SAME: clk: %[[CLK]]: i1
// CHECK-SAME: rst_ni: %[[RST]]: i1
// CHECK-SAME: s_axi_s0_awid: %[[AWID]]: i4
// CHECK-SAME: s_axi_s0_awuser: %[[AWUSER]]: i4
// CHECK-SAME: s_axi_s0_awvalid: %mgr_module_0.m_axi_m0_awvalid: i1
// CHECK-SAME: s_axi_s0_bready: %mgr_module_0.m_axi_m0_bready: i1

// The B response struct the manager unpacked is packed here from the
// subordinate's outputs, closing the response path.
// CHECK: %[[BSTRUCT]] = hw.struct_create (%sub_module_1.s_axi_s0_bid, %sub_module_1.s_axi_s0_bresp, %sub_module_1.s_axi_s0_buser) : !hw.struct<id: i4, resp: i2, user: i4>

// CHECK-NOT: axi4.node
// CHECK-NOT: axi4.manager_port
// CHECK-NOT: axi4.subordinate_port

hw.module.extern @mgr_module(in %clk : i1, in %rst_ni : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awuser : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wuser : i4, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_buser : i4, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_aruser : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_ruser : i4, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module(in %clk : i1, in %rst_ni : i1, in %s_axi_s0_awid : i4, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awuser : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wuser : i4, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i4, out s_axi_s0_bresp : i2, out s_axi_s0_buser : i4, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i4, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_aruser : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i4, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_ruser : i4, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mnode = axi4.node @mgr_module : !axi4.node
%snode = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mnode {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
axi4.subordinate_port %mgr, %clk, %rst node %snode {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
