// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s

// A manager -> xbar -> subordinate network. The xbar lowers to a generated
// SystemVerilog wrapper (sv.verbatim.source) instantiating PULP's axi_xbar, plus
// a typed sv.verbatim.module the hw.instance targets. The wrapper's boundary is
// the canonical channel-split form (a struct payload plus valid/ready per
// channel per face); PULP's req/resp packing lives in the generated SV.

//===--------------------------------------------------------------------===//
// The generated wrapper source.
//===--------------------------------------------------------------------===//

// The port type carries a single user width (4), so PULP's user_t and every
// boundary channel's user field are all 4 bits. The AW/AR id typedefs are the
// slave/master id widths (4 upstream, 5 down).
// CHECK: sv.verbatim.source @axi_xbar_1u1d_a32_d64_i4_o5_source
// CHECK-SAME: `include \22axi/typedef.svh\22
// CHECK-SAME: typedef logic [4-1:0] {{.*}}user_t;
// CHECK-SAME: typedef logic [4-1:0] {{.*}}slv_id_t;
// CHECK-SAME: typedef logic [5-1:0] {{.*}}mst_id_t;

// The boundary channel structs carry the port's user width, which equals the
// PULP user_t.
// CHECK-SAME: logic [4-1:0] user; } {{.*}}sub_aw_t;
// CHECK-SAME: logic [4-1:0] user; } {{.*}}sub_r_t;

// The canonical user copies straight across PULP's user_t in both directions
// (atop tied to 0); no zero-extend or slice is needed.
// CHECK-SAME: assign slv_req[0].aw = '{{[{]}}{{.*}}atop: '0, user: sub0_aw.user};
// CHECK-SAME: assign sub0_b = '{{[{]}}id: slv_resp[0].b.id, resp: slv_resp[0].b.resp, user: slv_resp[0].b.user};
// CHECK-SAME: assign sub0_r = '{{[{]}}{{.*}}user: slv_resp[0].r.user};
// CHECK-SAME: assign mgr0_aw = '{{[{]}}{{.*}}user: mst_req[0].aw.user};
// CHECK-SAME: assign mst_resp[0].b = '{{[{]}}id: mgr0_b.id, resp: mgr0_b.resp, user: mgr0_b.user};

// The crossbar config and address map come straight from the port type and the
// subordinate's access window.
// CHECK-SAME: AxiIdWidthSlvPorts: 4,
// CHECK-SAME: AxiAddrWidth:       32,
// CHECK-SAME: AxiDataWidth:       64,
// CHECK-SAME: '{{[{]}}idx: 0, start_addr: 32'h0, end_addr: 32'h1000}

// The real crossbar is fed the wrapper's clk/reset directly.
// CHECK-SAME: .clk_i{{.*}}(clk_i),
// CHECK-SAME: .rst_ni{{.*}}(rst_ni),

//===--------------------------------------------------------------------===//
// The typed interface the hw.instance targets mirrors the channel-split form,
// with the port's user width and the slave/master id widths per face.
//===--------------------------------------------------------------------===//

// CHECK: sv.verbatim.module @axi_xbar_1u1d_a32_d64_i4_o5
// CHECK-SAME: in %sub0_aw : !hw.struct<id: i4,
// CHECK-SAME: out sub0_r : !hw.struct<id: i4, data: i64, resp: i2, last: i1, user: i4>
// CHECK-SAME: out mgr0_aw : !hw.struct<id: i5,

//===--------------------------------------------------------------------===//
// Connectivity: the three instances are wired manager -> xbar -> subordinate,
// sharing one materialized clock and reset.
//===--------------------------------------------------------------------===//

// CHECK: %[[CLK:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.clock to i1
// CHECK: %[[RST:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.reset to i1

// The manager node drives the xbar's upstream (sub0) request channels and is
// fed the xbar's upstream ready/response signals plus the shared clk/reset.
// CHECK: %[[MGR:mgr_module_[0-9]+]].m_axi_m0_awid, {{.+}} @mgr_module(clk: %[[CLK]]: i1, rst_ni: %[[RST]]: i1,
// CHECK-SAME: m_axi_m0_awready: %[[XBAR:xbar_[0-9]+]].sub0_aw_ready: i1,
// CHECK-SAME: m_axi_m0_bvalid: %[[XBAR]].sub0_b_valid: i1,

// The manager's AW outputs are packed into the channel struct feeding the xbar.
// CHECK: %[[MGRAW:.+]] = hw.struct_create (%[[MGR]].m_axi_m0_awid, {{.+}}) : !hw.struct<id: i4, addr: i32,

// The subordinate node is driven from the xbar's downstream (mgr0) request
// channels, again sharing clk/reset.
// CHECK: %[[SUB:sub_module_[0-9]+]].s_axi_s0_awready, {{.+}} @sub_module(clk: %[[CLK]]: i1, rst_ni: %[[RST]]: i1,
// CHECK-SAME: s_axi_s0_awvalid: %[[XBAR]].mgr0_aw_valid: i1,

// The subordinate's B response is packed into the struct feeding the xbar's
// mgr0 face (master-side id width 5).
// CHECK: %[[SUBB:.+]] = hw.struct_create (%[[SUB]].s_axi_s0_bid, {{.+}}) : !hw.struct<id: i5, resp: i2, user: i4>

// The xbar instance ties it together: upstream fed by the manager, downstream
// fed by the subordinate, clk/reset shared with both.
// CHECK: %[[XBAR]].sub0_aw_ready, {{.+}} = hw.instance "[[XBAR]]" @axi_xbar_1u1d_a32_d64_i4_o5(clk_i: %[[CLK]]: i1, rst_ni: %[[RST]]: i1,
// CHECK-SAME: sub0_aw: %[[MGRAW]]: !hw.struct<id: i4,
// CHECK-SAME: sub0_aw_valid: %[[MGR]].m_axi_m0_awvalid: i1,
// CHECK-SAME: mgr0_b: %[[SUBB]]: !hw.struct<id: i5, resp: i2, user: i4>,
// CHECK-SAME: mgr0_r_valid: %[[SUB]].s_axi_s0_rvalid: i1)

hw.module.extern @mgr_module(in %clk : i1, in %rst_ni : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awuser : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wuser : i4, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_buser : i4, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_aruser : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_ruser : i4, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module(in %clk : i1, in %rst_ni : i1, in %s_axi_s0_awid : i5, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awuser : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wuser : i4, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i5, out s_axi_s0_bresp : i2, out s_axi_s0_buser : i4, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i5, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_aruser : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i5, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_ruser : i4, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mnode = axi4.node @mgr_module : !axi4.node
%snode = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mnode {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
%xbar = axi4.xbar %clk, %rst mgrs %mgr : (!axi4.port<32, 64, 4, 4, 4>) -> !axi4.port<32, 64, 5, 5, 4>
axi4.subordinate_port %xbar, %clk, %rst node %snode {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 5, 5, 4>
