// RUN: circt-opt %s --split-input-file --lower-axi4-to-hw | FileCheck %s

// A manager -> xbar -> subordinate network. The xbar lowers to a generated
// SystemVerilog wrapper (sv.verbatim.source) instantiating PULP's axi_xbar, plus
// a typed sv.verbatim.module interface the hw.instance targets. Every channel
// carries a `user` sideband and AW a 6-bit `atop`, both tied to 0.

// The wrapper source bakes in the AXI typedefs, the Cfg, the address map (from
// the subordinate's access window), and ties off reset/test/default-port ports.
// The typedefs are name-prefixed and widths track the port type (a32/d64/i4);
// the AW/AR id typedefs are the slave/master id widths.
// CHECK: sv.verbatim.source @axi_xbar_1u1d_a32_d64_i4_o4_source
// CHECK-SAME: `include \22axi/typedef.svh\22
// CHECK-SAME: typedef logic [32-1:0] axi_xbar_1u1d_a32_d64_i4_o4_addr_t;
// CHECK-SAME: typedef logic [64-1:0] axi_xbar_1u1d_a32_d64_i4_o4_data_t;
// CHECK-SAME: typedef logic [64/8-1:0] axi_xbar_1u1d_a32_d64_i4_o4_strb_t;
// CHECK-SAME: typedef logic [1-1:0] axi_xbar_1u1d_a32_d64_i4_o4_user_t;
// CHECK-SAME: typedef logic [4-1:0] axi_xbar_1u1d_a32_d64_i4_o4_slv_id_t;
// CHECK-SAME: typedef logic [4-1:0] axi_xbar_1u1d_a32_d64_i4_o4_mst_id_t;
// CHECK-SAME: `AXI_TYPEDEF_ALL(axi_xbar_1u1d_a32_d64_i4_o4_slv, axi_xbar_1u1d_a32_d64_i4_o4_addr_t, axi_xbar_1u1d_a32_d64_i4_o4_slv_id_t,
// CHECK-SAME: `AXI_TYPEDEF_ALL(axi_xbar_1u1d_a32_d64_i4_o4_mst, axi_xbar_1u1d_a32_d64_i4_o4_addr_t, axi_xbar_1u1d_a32_d64_i4_o4_mst_id_t,

// The wrapper's own port list uses the PULP req/resp typedefs (bit-identical to
// the hw.array-of-struct interface below).
// CHECK-SAME: module axi_xbar_1u1d_a32_d64_i4_o4 (
// CHECK-SAME: input  logic clk_i,
// CHECK-SAME: input  axi_xbar_1u1d_a32_d64_i4_o4_slv_req_t  [1-1:0] slv_ports_req_i,
// CHECK-SAME: output axi_xbar_1u1d_a32_d64_i4_o4_slv_resp_t [1-1:0] slv_ports_resp_o,
// CHECK-SAME: output axi_xbar_1u1d_a32_d64_i4_o4_mst_req_t  [1-1:0] mst_ports_req_o,
// CHECK-SAME: input  axi_xbar_1u1d_a32_d64_i4_o4_mst_resp_t [1-1:0] mst_ports_resp_i

// Every Cfg field is populated; unset/version-specific fields fall back to 0.
// CHECK-SAME: localparam axi_pkg::xbar_cfg_t Cfg = '{
// CHECK-SAME: NoSlvPorts:{{ +}}1,
// CHECK-SAME: NoMstPorts:{{ +}}1,
// CHECK-SAME: MaxMstTrans:{{ +}}8,
// CHECK-SAME: MaxSlvTrans:{{ +}}8,
// CHECK-SAME: FallThrough:{{ +}}1'b0,
// CHECK-SAME: LatencyMode:{{ +}}axi_pkg::CUT_ALL_AX,
// CHECK-SAME: AxiIdWidthSlvPorts:{{ +}}4,
// CHECK-SAME: AxiIdUsedSlvPorts:{{ +}}4,
// CHECK-SAME: UniqueIds:{{ +}}1'b0,
// CHECK-SAME: AxiAddrWidth:{{ +}}32,
// CHECK-SAME: AxiDataWidth:{{ +}}64,
// CHECK-SAME: NoAddrRules:{{ +}}1,
// CHECK-SAME: default:{{ +}}'0

// The address map has one rule covering the subordinate's [0, 0x1000) window.
// CHECK-SAME: localparam axi_pkg::xbar_rule_32_t [1-1:0] AddrMap = '{
// CHECK-SAME: '{idx: 0, start_addr: 32'h0, end_addr: 32'h1000}

// The axi_xbar instance binds every type parameter and ties off the control
// ports (reset high, test/default off).
// CHECK-SAME: axi_xbar #(
// CHECK-SAME: .Cfg{{ +}}(Cfg),
// CHECK-SAME: .ATOPs{{ +}}(1'b1),
// CHECK-SAME: .Connectivity{{ +}}('1),
// CHECK-SAME: .w_chan_t{{ +}}(axi_xbar_1u1d_a32_d64_i4_o4_slv_w_chan_t),
// CHECK-SAME: .slv_req_t{{ +}}(axi_xbar_1u1d_a32_d64_i4_o4_slv_req_t),
// CHECK-SAME: .mst_resp_t{{ +}}(axi_xbar_1u1d_a32_d64_i4_o4_mst_resp_t),
// CHECK-SAME: .rule_t{{ +}}(axi_pkg::xbar_rule_32_t)
// CHECK-SAME: ) i_xbar (
// CHECK-SAME: .clk_i{{ +}}(clk_i),
// CHECK-SAME: .rst_ni{{ +}}(1'b1),
// CHECK-SAME: .test_i{{ +}}(1'b0),
// CHECK-SAME: .addr_map_i{{ +}}(AddrMap),
// CHECK-SAME: .en_default_mst_port_i ('0),
// CHECK-SAME: .default_mst_port_i    ('0)
// CHECK-SAME: verilogName = "axi_xbar_1u1d_a32_d64_i4_o4"

// The typed interface carries the 5 array-of-struct data ports, points at the
// source, and is targeted by the hw.instance via verilogName.
// CHECK: sv.verbatim.module @axi_xbar_1u1d_a32_d64_i4_o4(in %clk_i : i1
// CHECK-SAME: %slv_ports_req_i : !hw.array<1xstruct<aw: !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, atop: i6, user: i1>
// CHECK-SAME: source = @axi_xbar_1u1d_a32_d64_i4_o4_source
// CHECK-SAME: verilogName = "axi_xbar_1u1d_a32_d64_i4_o4"

// The manager's request channels pack into a req struct - AW bridged with atop
// and user tied to 0 - and array into the slave request port; the subordinate's
// responses pack into the master response port.
// CHECK-DAG: %[[ATOP:.+]] = hw.constant 0 : i6
// CHECK-DAG: %[[AWCHAN:.+]] = hw.struct_create ({{.*}}, %[[ATOP]], %{{.+}}) : !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, atop: i6, user: i1>
// CHECK-DAG: %[[REQ:.+]] = hw.struct_create (%[[AWCHAN]], {{.*}}) : !hw.struct<aw: {{.*}}, r_ready: i1>
// CHECK-DAG: %[[SLVREQ:.+]] = hw.array_create %[[REQ]] : !hw.struct<aw: {{.*}}, r_ready: i1>
// CHECK-DAG: %[[MSTRESP:.+]] = hw.array_create %{{.+}} : !hw.struct<aw_ready: i1, {{.*}}>

// CHECK: hw.instance "xbar_2" @axi_xbar_1u1d_a32_d64_i4_o4(clk_i: %{{.+}}: i1, slv_ports_req_i: %[[SLVREQ]]: {{.*}}, mst_ports_resp_i: %[[MSTRESP]]: {{.*}}) ->

// The slave response and master request arrays unpack per element; the master AW
// is bridged back to canonical, dropping atop and user.
// CHECK-DAG: %[[SLVRESP:.+]] = hw.array_get %xbar_2.slv_ports_resp_o[%{{.+}}]
// CHECK-DAG: %{{.+}} = hw.struct_extract %[[SLVRESP]]["aw_ready"]
// CHECK-DAG: %[[MSTREQ:.+]] = hw.array_get %xbar_2.mst_ports_req_o[%{{.+}}]
// CHECK-DAG: %[[MSTAW:.+]] = hw.struct_extract %[[MSTREQ]]["aw"] : !hw.struct<aw: {{.*}}, r_ready: i1>
// CHECK-DAG: %{{.+}} = hw.struct_extract %[[MSTAW]]["id"] : !hw.struct<id: i4, {{.*}}, atop: i6, user: i1>

// CHECK-NOT: axi4.node
// CHECK-NOT: axi4.manager_port
// CHECK-NOT: axi4.subordinate_port
// CHECK-NOT: axi4.xbar

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
%xbar = axi4.xbar %clk mgrs %mgr : (!axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 4>
axi4.subordinate_port %xbar, %clk node %snode {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Chained crossbars: manager -> xbar -> xbar -> subordinate. The two xbars route
// differently - the upstream xbar sees the whole address space, the downstream
// xbar the subordinate's window - so each gets its own wrapper. The downstream
// xbar is lowered second, so it takes the _0-suffixed name.

// CHECK: sv.verbatim.source @axi_xbar_1u1d_a32_d64_i4_o4_0_source
// CHECK-SAME: '{idx: 0, start_addr: 32'h0, end_addr: 32'h1000}
// CHECK: sv.verbatim.source @axi_xbar_1u1d_a32_d64_i4_o4_source
// CHECK-SAME: '{idx: 0, start_addr: 32'h0, end_addr: 32'hFFFFFFFF}

// The upstream xbar (xbar_2, full-range) master request feeds the downstream
// xbar (xbar_3, windowed) slave request: its AW is unpacked (atop/user dropped)
// then repacked with atop/user tied to 0.
// CHECK: hw.instance "xbar_2" @axi_xbar_1u1d_a32_d64_i4_o4(
// CHECK: %[[X1MSTREQ:.+]] = hw.array_get %xbar_2.mst_ports_req_o[%{{.+}}]
// CHECK: %[[X1AW:.+]] = hw.struct_extract %[[X1MSTREQ]]["aw"]
// CHECK: %[[X1AWID:.+]] = hw.struct_extract %[[X1AW]]["id"] : !hw.struct<id: i4, {{.*}}, atop: i6, user: i1>
// CHECK: hw.instance "xbar_3" @axi_xbar_1u1d_a32_d64_i4_o4_0(

// CHECK-NOT: axi4.xbar

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
%xbar1 = axi4.xbar %clk mgrs %mgr : (!axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 4>
%xbar2 = axi4.xbar %clk mgrs %xbar1 : (!axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 4>
axi4.subordinate_port %xbar2, %clk node %snode {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// A 2-manager -> xbar -> 2-subordinate network exercises multi-port shapes: the
// req/resp arrays are sized to the port counts, the Cfg tracks them, and the
// address map carries one rule per downstream window.

// CHECK: sv.verbatim.source @axi_xbar_2u2d_a32_d64_i4_o5_source
// CHECK-SAME: input  axi_xbar_2u2d_a32_d64_i4_o5_slv_req_t  [2-1:0] slv_ports_req_i,
// CHECK-SAME: output axi_xbar_2u2d_a32_d64_i4_o5_mst_req_t  [2-1:0] mst_ports_req_o,
// CHECK-SAME: NoSlvPorts:{{ +}}2,
// CHECK-SAME: NoMstPorts:{{ +}}2,
// CHECK-SAME: NoAddrRules:{{ +}}2,
// One rule per downstream, indexed by master-port order (the xbar result's
// use-list order, here reverse of definition).
// CHECK-SAME: axi_pkg::xbar_rule_32_t [2-1:0] AddrMap
// CHECK-DAG: '{idx: 0, start_addr: 32'h1000, end_addr: 32'h2000}
// CHECK-DAG: '{idx: 1, start_addr: 32'h0, end_addr: 32'h1000}

// Both upstream managers and both downstream subordinates wire through the sized
// req/resp arrays.
// CHECK: hw.instance "xbar_4" @axi_xbar_2u2d_a32_d64_i4_o5(clk_i: %{{.+}}: i1, slv_ports_req_i: %{{.+}}: !hw.array<2xstruct<{{.*}}>>, mst_ports_resp_i: %{{.+}}: !hw.array<2xstruct<{{.*}}>>)

// CHECK-NOT: axi4.xbar

hw.module.extern @mgr_module(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
// The subordinates carry the widened 5-bit id, so they attach to a module with
// 5-bit id ports.
hw.module.extern @sub_module5(in %clk : i1, in %s_axi_s0_awid : i5, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i5, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i5, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i5, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)

%clk = unrealized_conversion_cast to !axi4.clock
%mnode1 = axi4.node @mgr_module : !axi4.node
%mnode2 = axi4.node @mgr_module : !axi4.node
%snode1 = axi4.node @sub_module5 : !axi4.node
%snode2 = axi4.node @sub_module5 : !axi4.node
%mgr1 = axi4.manager_port %clk node %mnode1 {
  port_mapping = #axi4.port_wires<"clk", "m0">,
  access = [#axi4.window<base = 0, size = 8192, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
%mgr2 = axi4.manager_port %clk node %mnode2 {
  port_mapping = #axi4.port_wires<"clk", "m0">,
  access = [#axi4.window<base = 0, size = 8192, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
// Two managers widen the downstream id by ceil(log2(2)) = 1 bit (4 -> 5).
%xbar = axi4.xbar %clk mgrs %mgr1, %mgr2 : (!axi4.port<32, 64, 4>, !axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 5>
axi4.subordinate_port %xbar, %clk node %snode1 {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 5>
axi4.subordinate_port %xbar, %clk node %snode2 {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 5>
