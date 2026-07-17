// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s

// A manager -> cut -> subordinate network. The cut lowers to a generated
// SystemVerilog wrapper (sv.verbatim.source) instantiating PULP's axi_cut, plus
// a typed sv.verbatim.module the hw.instance targets. The wrapper's boundary is
// the canonical channel-split form (a struct payload plus valid/ready per
// channel per face); PULP's req/resp packing lives in the generated SV.
//
// The port type uses different write (4) and read (6) ID widths. Unlike the
// crossbar, axi_cut never inspects IDs, so the two are kept independent: the
// write path (AW, B) uses wid_t and the read path (AR, R) uses rid_t.

//===--------------------------------------------------------------------===//
// The generated wrapper source.
//===--------------------------------------------------------------------===//

// CHECK: sv.verbatim.source @axi_cut_a32_d64_iw4_ir6_u2_source
// CHECK-SAME: `include \22axi/typedef.svh\22
// CHECK-SAME: typedef logic [2-1:0] {{.*}}user_t;
// CHECK-SAME: typedef logic [4-1:0] {{.*}}wid_t;
// CHECK-SAME: typedef logic [6-1:0] {{.*}}rid_t;

// The write-path channel structs use wid_t; the read-path structs use rid_t.
// CHECK-SAME: {{.*}}wid_t id;{{.*}} } {{.*}}sub_aw_t;
// CHECK-SAME: {{.*}}rid_t id;{{.*}} } {{.*}}sub_ar_t;

// The canonical user copies straight across PULP's user_t in both directions
// (atop tied to 0 on AW); no zero-extend or slice is needed.
// CHECK-SAME: assign slv_req.aw = '{{[{]}}{{.*}}atop: '0, user: sub0_aw.user};
// CHECK-SAME: assign sub0_b = '{{[{]}}id: slv_resp.b.id, resp: slv_resp.b.resp, user: slv_resp.b.user};
// CHECK-SAME: assign mgr0_aw = '{{[{]}}{{.*}}user: mst_req.aw.user};

// The register slice itself, fed the wrapper's clk/reset and req/resp structs.
// CHECK-SAME: axi_cut #(
// CHECK-SAME: .Bypass     (1'b0),
// CHECK-SAME: ) i_cut (
// CHECK-SAME: .clk_i      (clk_i),
// CHECK-SAME: .rst_ni     (rst_ni),
// CHECK-SAME: .slv_req_i  (slv_req),
// CHECK-SAME: .mst_resp_i (mst_resp)

//===--------------------------------------------------------------------===//
// The typed interface the hw.instance targets mirrors the channel-split form,
// with the port's user width and the per-channel write/read id widths.
//===--------------------------------------------------------------------===//

// CHECK: sv.verbatim.module @axi_cut_a32_d64_iw4_ir6_u2
// CHECK-SAME: in %sub0_aw : !hw.struct<id: i4,
// CHECK-SAME: in %sub0_ar : !hw.struct<id: i6,
// CHECK-SAME: out mgr0_aw : !hw.struct<id: i4,
// CHECK-SAME: in %mgr0_r : !hw.struct<id: i6,

//===--------------------------------------------------------------------===//
// Connectivity: the three instances are wired manager -> cut -> subordinate,
// sharing one materialized clock and reset.
//===--------------------------------------------------------------------===//

// CHECK: %[[CLK:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.clock to i1
// CHECK: %[[RST:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.reset to i1

// The manager node drives the cut's upstream (sub0) request channels and is
// fed the cut's upstream ready/response signals plus the shared clk/reset.
// CHECK: %[[MGR:mgr_module_[0-9]+]].m_axi_m0_awid, {{.+}} @mgr_module(clk: %[[CLK]]: i1, rst_ni: %[[RST]]: i1,
// CHECK-SAME: m_axi_m0_awready: %[[CUT:cut_[0-9]+]].sub0_aw_ready: i1,

// The subordinate node is driven from the cut's downstream (mgr0) request
// channels, again sharing clk/reset.
// CHECK: %[[SUB:sub_module_[0-9]+]].s_axi_s0_awready, {{.+}} @sub_module(clk: %[[CLK]]: i1, rst_ni: %[[RST]]: i1,
// CHECK-SAME: s_axi_s0_awvalid: %[[CUT]].mgr0_aw_valid: i1,

// The cut instance ties it together: upstream fed by the manager, downstream
// fed by the subordinate, clk/reset shared with both.
// CHECK: %[[CUT]].sub0_aw_ready, {{.+}} = hw.instance "[[CUT]]" @axi_cut_a32_d64_iw4_ir6_u2(clk_i: %[[CLK]]: i1, rst_ni: %[[RST]]: i1,
// CHECK-SAME: sub0_aw_valid: %[[MGR]].m_axi_m0_awvalid: i1,
// CHECK-SAME: mgr0_r_valid: %[[SUB]].s_axi_s0_rvalid: i1)

hw.module.extern @mgr_module(in %clk : i1, in %rst_ni : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awuser : i2, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wuser : i2, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_buser : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i6, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_aruser : i2, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i6, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_ruser : i2, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module(in %clk : i1, in %rst_ni : i1, in %s_axi_s0_awid : i4, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awuser : i2, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wuser : i2, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i4, out s_axi_s0_bresp : i2, out s_axi_s0_buser : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i6, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_aruser : i2, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i6, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_ruser : i2, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mnode = axi4.node @mgr_module : !axi4.node
%snode = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mnode {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 6, 2>
%cut = axi4.cut %clk, %rst at %mgr : !axi4.port<32, 64, 4, 6, 2>
axi4.subordinate_port %cut, %clk, %rst node %snode {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 6, 2>
