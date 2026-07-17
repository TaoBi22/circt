// RUN: circt-opt %s --lower-axi4-to-hw | FileCheck %s

// A manager -> data_width_converter -> subordinate network. The converter lowers
// to a generated SystemVerilog wrapper (sv.verbatim.source) instantiating PULP's
// axi_dw_converter, plus a typed sv.verbatim.module the hw.instance targets.
// Unlike the cut it is asymmetric: only the data (and strobe) width differs
// between the upstream (slave, 64) and downstream (master, 32) sides; the ID,
// address, and user widths are preserved. Single clock/reset domain.

//===--------------------------------------------------------------------===//
// The generated wrapper source.
//===--------------------------------------------------------------------===//

// CHECK: sv.verbatim.source @axi_dw_converter_a32_i4_u2_sd64_md32_source
// CHECK-SAME: `include \22axi/typedef.svh\22

// Shared address/id/user; split data/strb per side.
// CHECK-SAME: typedef logic [4-1:0] {{.*}}id_t;
// CHECK-SAME: typedef logic [64-1:0] {{.*}}slv_data_t;
// CHECK-SAME: typedef logic [32-1:0] {{.*}}mst_data_t;

// The W/R boundary structs use each side's data typedef; AW is shared.
// CHECK-SAME: {{.*}}slv_data_t data;{{.*}} } {{.*}}sub_w_t;
// CHECK-SAME: {{.*}}mst_data_t data;{{.*}} } {{.*}}mgr_w_t;

// The converter, fed the wrapper's clk/reset and the slv/mst req/resp structs.
// CHECK-SAME: axi_dw_converter #(
// CHECK-SAME: .AxiSlvPortDataWidth (64),
// CHECK-SAME: .AxiMstPortDataWidth (32),
// CHECK-SAME: .AxiAddrWidth        (32),
// CHECK-SAME: .AxiIdWidth          (4),
// CHECK-SAME: ) i_dw_converter (
// CHECK-SAME: .clk_i      (clk_i),
// CHECK-SAME: .slv_req_i  (slv_req),
// CHECK-SAME: .mst_resp_i (mst_resp)

//===--------------------------------------------------------------------===//
// The typed interface: shared id/addr on AW, per-side data on W/R.
//===--------------------------------------------------------------------===//

// CHECK: sv.verbatim.module @axi_dw_converter_a32_i4_u2_sd64_md32
// CHECK-SAME: in %sub0_w : !hw.struct<data: i64,
// CHECK-SAME: out mgr0_w : !hw.struct<data: i32,

//===--------------------------------------------------------------------===//
// Connectivity: manager -> converter -> subordinate, one shared clk/reset.
//===--------------------------------------------------------------------===//

// CHECK: %[[CLK:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.clock to i1
// CHECK: %[[RST:.+]] = unrealized_conversion_cast %{{.+}} : !axi4.reset to i1

// CHECK: %[[MGR:mgr_module_[0-9]+]].m_axi_m0_awid, {{.+}} @mgr_module(clk: %[[CLK]]: i1, rst_ni: %[[RST]]: i1,
// CHECK-SAME: m_axi_m0_awready: %[[DWC:dw_converter_[0-9]+]].sub0_aw_ready: i1,

// CHECK: %[[SUB:sub_module_[0-9]+]].s_axi_s0_awready, {{.+}} @sub_module(clk: %[[CLK]]: i1, rst_ni: %[[RST]]: i1,
// CHECK-SAME: s_axi_s0_awvalid: %[[DWC]].mgr0_aw_valid: i1,

// CHECK: %[[DWC]].sub0_aw_ready, {{.+}} = hw.instance "[[DWC]]" @axi_dw_converter_a32_i4_u2_sd64_md32(clk_i: %[[CLK]]: i1, rst_ni: %[[RST]]: i1,
// CHECK-SAME: sub0_aw_valid: %[[MGR]].m_axi_m0_awvalid: i1,
// CHECK-SAME: mgr0_r_valid: %[[SUB]].s_axi_s0_rvalid: i1)

hw.module.extern @mgr_module(in %clk : i1, in %rst_ni : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awuser : i2, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wuser : i2, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_buser : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_aruser : i2, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_ruser : i2, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module(in %clk : i1, in %rst_ni : i1, in %s_axi_s0_awid : i4, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awuser : i2, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i32, in %s_axi_s0_wstrb : i4, in %s_axi_s0_wlast : i1, in %s_axi_s0_wuser : i2, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i4, out s_axi_s0_bresp : i2, out s_axi_s0_buser : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i4, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_aruser : i2, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i4, out s_axi_s0_rdata : i32, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_ruser : i2, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mnode = axi4.node @mgr_module : !axi4.node
%snode = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mnode {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 2>
%dwc = axi4.data_width_converter %clk, %rst, %mgr : (!axi4.port<32, 64, 4, 4, 2>) -> !axi4.port<32, 32, 4, 4, 2>
axi4.subordinate_port %dwc, %clk, %rst node %snode {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 32, 4, 4, 2>
