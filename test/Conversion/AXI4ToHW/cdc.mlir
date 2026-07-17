// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s

// A manager -> cdc -> subordinate network spanning two clock domains. The cdc
// lowers to a generated SystemVerilog wrapper (sv.verbatim.source) instantiating
// PULP's axi_cdc, plus a typed sv.verbatim.module the hw.instance targets. Like
// the cut it is symmetric (widths preserved across the crossing, write/read IDs
// independent), but it has a source and a destination clock/reset pair.

//===--------------------------------------------------------------------===//
// The generated wrapper source.
//===--------------------------------------------------------------------===//

// CHECK: sv.verbatim.source @axi_cdc_a32_d64_iw4_ir6_u2_source
// CHECK-SAME: `include \22axi/typedef.svh\22
// CHECK-SAME: typedef logic [4-1:0] {{.*}}wid_t;
// CHECK-SAME: typedef logic [6-1:0] {{.*}}rid_t;

// The write-path channel structs use wid_t; the read-path structs use rid_t.
// CHECK-SAME: {{.*}}wid_t id;{{.*}} } {{.*}}sub_aw_t;
// CHECK-SAME: {{.*}}rid_t id;{{.*}} } {{.*}}sub_ar_t;

// The boundary bridges match the cut's (atop tied to 0 on AW; user copied).
// CHECK-SAME: assign slv_req.aw = '{{[{]}}{{.*}}atop: '0, user: sub0_aw.user};
// CHECK-SAME: assign mgr0_aw = '{{[{]}}{{.*}}user: mst_req.aw.user};

// The clock-domain-crossing FIFO: source side fed the wrapper's src clk/reset
// and slv req/resp, destination side the dst clk/reset and mst req/resp.
// CHECK-SAME: axi_cdc #(
// CHECK-SAME: .LogDepth   (1),
// CHECK-SAME: .SyncStages (2)
// CHECK-SAME: ) i_cdc (
// CHECK-SAME: .src_clk_i  (src_clk_i),
// CHECK-SAME: .src_req_i  (slv_req),
// CHECK-SAME: .dst_clk_i  (dst_clk_i),
// CHECK-SAME: .dst_req_o  (mst_req),
// CHECK-SAME: .dst_resp_i (mst_resp)

//===--------------------------------------------------------------------===//
// The typed interface the hw.instance targets: a src and dst clock/reset, then
// the channel-split faces with per-channel write/read id widths.
//===--------------------------------------------------------------------===//

// CHECK: sv.verbatim.module @axi_cdc_a32_d64_iw4_ir6_u2
// CHECK-SAME: in %src_clk_i : i1, in %src_rst_ni : i1, in %dst_clk_i : i1, in %dst_rst_ni : i1,
// CHECK-SAME: in %sub0_aw : !hw.struct<id: i4,
// CHECK-SAME: in %sub0_ar : !hw.struct<id: i6,
// CHECK-SAME: out mgr0_aw : !hw.struct<id: i4,

//===--------------------------------------------------------------------===//
// Connectivity: manager in the source domain, subordinate in the destination
// domain, each clock/reset materialized independently.
//===--------------------------------------------------------------------===//

// CHECK: %[[CLKA:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.clock to i1
// CHECK: %[[RSTA:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.reset to i1

// The manager node is in the source domain.
// CHECK: {{.+}} @mgr_module(clk: %[[CLKA]]: i1, rst_ni: %[[RSTA]]: i1,
// CHECK-SAME: m_axi_m0_awready: %[[CDC:cdc_[0-9]+]].sub0_aw_ready: i1,

// The destination domain's clock/reset are materialized separately.
// CHECK: %[[CLKB:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.clock to i1
// CHECK: %[[RSTB:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.reset to i1

// The subordinate node is in the destination domain.
// CHECK: {{.+}} @sub_module(clk: %[[CLKB]]: i1, rst_ni: %[[RSTB]]: i1,
// CHECK-SAME: s_axi_s0_awvalid: %[[CDC]].mgr0_aw_valid: i1,

// The cdc instance bridges the two: source clk/reset from the manager domain,
// destination clk/reset from the subordinate domain.
// CHECK: %[[CDC]].sub0_aw_ready, {{.+}} = hw.instance "[[CDC]]" @axi_cdc_a32_d64_iw4_ir6_u2(src_clk_i: %[[CLKA]]: i1, src_rst_ni: %[[RSTA]]: i1, dst_clk_i: %[[CLKB]]: i1, dst_rst_ni: %[[RSTB]]: i1,

hw.module.extern @mgr_module(in %clk : i1, in %rst_ni : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awuser : i2, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wuser : i2, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_buser : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i6, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_aruser : i2, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i6, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_ruser : i2, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module(in %clk : i1, in %rst_ni : i1, in %s_axi_s0_awid : i4, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awuser : i2, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wuser : i2, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i4, out s_axi_s0_bresp : i2, out s_axi_s0_buser : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i6, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_aruser : i2, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i6, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_ruser : i2, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)

%clkA = unrealized_conversion_cast to !axi4.clock
%rstA = unrealized_conversion_cast to !axi4.reset
%clkB = unrealized_conversion_cast to !axi4.clock
%rstB = unrealized_conversion_cast to !axi4.reset
%mnode = axi4.node @mgr_module : !axi4.node
%snode = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clkA, %rstA node %mnode {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 6, 2>
%cdc = axi4.cdc %mgr from [%clkA, %rstA] to [%clkB, %rstB] : !axi4.port<32, 64, 4, 6, 2>
axi4.subordinate_port %cdc, %clkB, %rstB node %snode {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 6, 2>
