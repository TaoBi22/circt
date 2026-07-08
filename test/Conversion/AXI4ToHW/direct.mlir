// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s

// A single manager->subordinate node pair, wired directly (no crossbar).

// CHECK: hw.module.extern @mgr_module
// CHECK: hw.module.extern @sub_module

// Response payloads (B/R) flow subordinate->manager: unpacked from a struct and
// fed to the manager's flat response inputs.
// CHECK: %[[BID:[a-zA-Z0-9_]+]] = hw.struct_extract %[[BSTRUCT:[a-zA-Z0-9_]+]]["id"] : !hw.struct<id: i4, resp: i2>
// CHECK: %[[BRESP:[a-zA-Z0-9_]+]] = hw.struct_extract %[[BSTRUCT]]["resp"] : !hw.struct<id: i4, resp: i2>

// The abstract clock is lowered to a single shared i1 module clock port.
// CHECK: %[[CLK:[a-zA-Z0-9_]+]] = unrealized_conversion_cast %{{.+}} : !axi4.clock to i1

// The manager instance takes request-ready and response valid/payload from the
// subordinate (payload via the extracts above).
// CHECK: hw.instance "mgr_module_0" @mgr_module(
// CHECK-SAME: clk: %[[CLK]]: i1
// CHECK-SAME: m_axi_m0_awready: %[[SUB:[a-zA-Z0-9_]+]].s_axi_s0_awready: i1
// CHECK-SAME: m_axi_m0_bid: %[[BID]]: i4
// CHECK-SAME: m_axi_m0_bresp: %[[BRESP]]: i2
// CHECK-SAME: m_axi_m0_bvalid: %[[SUB]].s_axi_s0_bvalid: i1

// Request payloads (AW/W/AR) are packed from the manager's flat outputs...
// CHECK: %[[AW:[a-zA-Z0-9_]+]] = hw.struct_create (%[[MGR:[a-zA-Z0-9_]+]].m_axi_m0_awid, %[[MGR]].m_axi_m0_awaddr, %[[MGR]].m_axi_m0_awlen, %[[MGR]].m_axi_m0_awsize, %[[MGR]].m_axi_m0_awburst, %[[MGR]].m_axi_m0_awlock, %[[MGR]].m_axi_m0_awcache, %[[MGR]].m_axi_m0_awprot, %[[MGR]].m_axi_m0_awqos, %[[MGR]].m_axi_m0_awregion) : !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4>
// CHECK: %[[W:[a-zA-Z0-9_]+]] = hw.struct_create (%[[MGR]].m_axi_m0_wdata, %[[MGR]].m_axi_m0_wstrb, %[[MGR]].m_axi_m0_wlast) : !hw.struct<data: i64, strb: i8, last: i1>

// ...then unpacked to feed the subordinate.
// CHECK: %[[AWID:[a-zA-Z0-9_]+]] = hw.struct_extract %[[AW]]["id"]

// The subordinate instance takes request valid/payload and response-ready from
// the manager (request payload via the extract above).
// CHECK: hw.instance "sub_module_1" @sub_module(
// CHECK-SAME: clk: %[[CLK]]: i1
// CHECK-SAME: s_axi_s0_awid: %[[AWID]]: i4
// CHECK-SAME: s_axi_s0_awvalid: %[[MGR]].m_axi_m0_awvalid: i1
// CHECK-SAME: s_axi_s0_bready: %[[MGR]].m_axi_m0_bready: i1

// The response payload consumed by the manager is packed from the
// subordinate's outputs.
// CHECK: %[[BSTRUCT]] = hw.struct_create (%[[SUB]].s_axi_s0_bid, %[[SUB]].s_axi_s0_bresp) : !hw.struct<id: i4, resp: i2>

// CHECK-NOT: axi4.node
// CHECK-NOT: axi4.manager_port
// CHECK-NOT: axi4.subordinate_port

hw.module.extern @mgr_module(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module(in %clk : i1, in %s_axi_s0_awid : i4, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i4, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i4, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i4, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)

%clk = unrealized_conversion_cast to !axi4.clock
%mnode = axi4.node @mgr_module : !axi4.node
%snode = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk node %mnode {
  port_mapping = #axi4.port_wires<"clk", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate_port %mgr, %clk node %snode {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
