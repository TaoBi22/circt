// RUN: circt-opt %s --split-input-file --lower-axi4-to-hw | FileCheck %s

// A manager -> xbar -> subordinate network. The xbar lowers to a generated
// SystemVerilog wrapper (sv.verbatim.source) instantiating PULP's axi_xbar, plus
// a typed sv.verbatim.module interface the hw.instance targets. The wrapper's
// boundary mirrors the internal channel-split form (a struct payload plus
// valid/ready per channel per face); PULP's req/resp packing - and tying its
// atop/user fields to 0 - lives entirely in the generated SystemVerilog.

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

// Canonical (non-PULP) per-channel payload typedefs for the boundary ports, one
// set per face role. No atop/user - those are PULP-only and stay internal.
// CHECK-SAME: typedef struct packed { axi_xbar_1u1d_a32_d64_i4_o4_slv_id_t id; axi_xbar_1u1d_a32_d64_i4_o4_addr_t addr; axi_pkg::len_t len; axi_pkg::size_t size; axi_pkg::burst_t burst; logic lock; axi_pkg::cache_t cache; axi_pkg::prot_t prot; axi_pkg::qos_t qos; axi_pkg::region_t region; } axi_xbar_1u1d_a32_d64_i4_o4_sub_aw_t;
// CHECK-SAME: typedef struct packed { axi_xbar_1u1d_a32_d64_i4_o4_data_t data; axi_xbar_1u1d_a32_d64_i4_o4_strb_t strb; logic last; } axi_xbar_1u1d_a32_d64_i4_o4_sub_w_t;
// CHECK-SAME: } axi_xbar_1u1d_a32_d64_i4_o4_mgr_r_t;

// The wrapper's port list is per-channel struct/valid/ready per face, matching
// the typed interface below. Requests are received on the subordinate face and
// driven on the manager face; responses flow the other way.
// CHECK-SAME: module axi_xbar_1u1d_a32_d64_i4_o4 (
// CHECK-SAME: input  logic clk_i,
// CHECK-SAME: input  axi_xbar_1u1d_a32_d64_i4_o4_sub_aw_t sub0_aw,
// CHECK-SAME: input  logic sub0_aw_valid,
// CHECK-SAME: output logic sub0_aw_ready,
// CHECK-SAME: output axi_xbar_1u1d_a32_d64_i4_o4_sub_b_t sub0_b,
// CHECK-SAME: output logic sub0_b_valid,
// CHECK-SAME: input  logic sub0_b_ready,
// CHECK-SAME: output axi_xbar_1u1d_a32_d64_i4_o4_mgr_aw_t mgr0_aw,
// CHECK-SAME: output logic mgr0_aw_valid,
// CHECK-SAME: input  logic mgr0_aw_ready,
// CHECK-SAME: input  axi_xbar_1u1d_a32_d64_i4_o4_mgr_b_t mgr0_b,

// Internal PULP-shaped req/resp arrays, wired to the boundary ports by assigns.
// CHECK-SAME: axi_xbar_1u1d_a32_d64_i4_o4_slv_req_t  [1-1:0] slv_req;
// CHECK-SAME: axi_xbar_1u1d_a32_d64_i4_o4_mst_resp_t [1-1:0] mst_resp;

// The request payload bridges canonical -> PULP with atop/user tied to 0; the
// response bridges PULP -> canonical, dropping them.
// CHECK-SAME: assign slv_req[0].aw = '{id: sub0_aw.id, addr: sub0_aw.addr, len: sub0_aw.len, size: sub0_aw.size, burst: sub0_aw.burst, lock: sub0_aw.lock, cache: sub0_aw.cache, prot: sub0_aw.prot, qos: sub0_aw.qos, region: sub0_aw.region, atop: '0, user: '0};
// CHECK-SAME: assign slv_req[0].aw_valid = sub0_aw_valid;
// CHECK-SAME: assign sub0_aw_ready = slv_resp[0].aw_ready;
// CHECK-SAME: assign sub0_b = '{id: slv_resp[0].b.id, resp: slv_resp[0].b.resp};
// CHECK-SAME: assign mgr0_aw = '{id: mst_req[0].aw.id, addr: mst_req[0].aw.addr, len: mst_req[0].aw.len, size: mst_req[0].aw.size, burst: mst_req[0].aw.burst, lock: mst_req[0].aw.lock, cache: mst_req[0].aw.cache, prot: mst_req[0].aw.prot, qos: mst_req[0].aw.qos, region: mst_req[0].aw.region};
// CHECK-SAME: assign mst_resp[0].b = '{id: mgr0_b.id, resp: mgr0_b.resp, user: '0};

// Every Cfg field is populated; unset/version-specific fields fall back to 0.
// CHECK-SAME: localparam axi_pkg::xbar_cfg_t Cfg = '{
// CHECK-SAME: NoSlvPorts:{{ +}}1,
// CHECK-SAME: NoMstPorts:{{ +}}1,
// CHECK-SAME: NoAddrRules:{{ +}}1,
// CHECK-SAME: default:{{ +}}'0

// The address map has one rule covering the subordinate's [0, 0x1000) window.
// CHECK-SAME: localparam axi_pkg::xbar_rule_32_t [1-1:0] AddrMap = '{
// CHECK-SAME: '{idx: 0, start_addr: 32'h0, end_addr: 32'h1000}

// The axi_xbar instance binds every type parameter, wires to the internal
// req/resp arrays, and ties off the control ports (reset high, test/default off).
// CHECK-SAME: axi_xbar #(
// CHECK-SAME: .Cfg{{ +}}(Cfg),
// CHECK-SAME: .slv_req_t{{ +}}(axi_xbar_1u1d_a32_d64_i4_o4_slv_req_t),
// CHECK-SAME: .rule_t{{ +}}(axi_pkg::xbar_rule_32_t)
// CHECK-SAME: ) i_xbar (
// CHECK-SAME: .clk_i{{ +}}(clk_i),
// CHECK-SAME: .rst_ni{{ +}}(1'b1),
// CHECK-SAME: .slv_ports_req_i       (slv_req),
// CHECK-SAME: .mst_ports_resp_i      (mst_resp),
// CHECK-SAME: .addr_map_i{{ +}}(AddrMap),
// CHECK-SAME: verilogName = "axi_xbar_1u1d_a32_d64_i4_o4"

// The typed interface carries the per-channel struct/valid/ready ports, points
// at the source, and is targeted by the hw.instance via verilogName.
// CHECK: sv.verbatim.module @axi_xbar_1u1d_a32_d64_i4_o4(in %clk_i : i1
// CHECK-SAME: in %sub0_aw : !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4>
// CHECK-SAME: in %sub0_aw_valid : i1, out sub0_aw_ready : i1
// CHECK-SAME: out mgr0_aw : !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4>
// CHECK-SAME: source = @axi_xbar_1u1d_a32_d64_i4_o4_source
// CHECK-SAME: verilogName = "axi_xbar_1u1d_a32_d64_i4_o4"

// The manager-face request outputs are structs; the subordinate node unpacks
// them per field (emitted during node lowering, before the xbar instance).
// CHECK: hw.struct_extract %xbar_2.mgr0_aw["id"] : !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4>

// The manager's request channels feed the subordinate-face struct ports (each
// channel one struct value); mgr0_aw is a struct output.
// CHECK: hw.instance "xbar_2" @axi_xbar_1u1d_a32_d64_i4_o4(clk_i: %{{[^:]+}}: i1, sub0_aw: %{{[^:]+}}: !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4>, sub0_aw_valid: %{{[^:]+}}: i1
// CHECK-SAME: -> (sub0_aw_ready: i1, {{.*}}, mgr0_aw: !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4>

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

// Chained crossbars: manager -> xbar -> xbar -> subordinate. The upstream xbar's
// reachable range is computed from the downstream xbar's subordinate, so both
// xbars route the same [0, 0x1000) window and share one wrapper (not full-range).

// Exactly one wrapper is emitted, routing the subordinate's window.
// CHECK: sv.verbatim.source @axi_xbar
// CHECK-SAME: '{idx: 0, start_addr: 32'h0, end_addr: 32'h1000}
// CHECK-NOT: sv.verbatim.source
// CHECK: sv.verbatim.module @axi_xbar_1u1d_a32_d64_i4_o4(

// Both crossbars instantiate the shared wrapper.
// CHECK: hw.instance "xbar_2" @axi_xbar_1u1d_a32_d64_i4_o4(

// The two wrappers speak the same canonical channel form, so the upstream xbar's
// manager-face request output feeds the downstream xbar's subordinate-face input
// directly - no unpack/repack (and no atop/user round-trip) between them.
// CHECK: hw.instance "xbar_3" @axi_xbar_1u1d_a32_d64_i4_o4(clk_i: %{{[^:]+}}: i1, sub0_aw: %xbar_2.mgr0_aw: !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4>

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

// A 2-manager -> xbar -> 2-subordinate network exercises multi-port shapes: each
// face gets its own numbered set of boundary ports, the internal req/resp arrays
// are sized to the port counts, the Cfg tracks them, and the address map carries
// one rule per downstream window.

// CHECK: sv.verbatim.source @axi_xbar_2u2d_a32_d64_i4_o5_source
// Two subordinate faces (sub0/sub1) and two manager faces (mgr0/mgr1) on the
// boundary; the internal PULP arrays are sized [2-1:0].
// CHECK-SAME: input  logic sub1_aw_valid,
// CHECK-SAME: output logic mgr1_aw_valid,
// CHECK-SAME: axi_xbar_2u2d_a32_d64_i4_o5_slv_req_t  [2-1:0] slv_req;
// CHECK-SAME: NoSlvPorts:{{ +}}2,
// CHECK-SAME: NoMstPorts:{{ +}}2,
// CHECK-SAME: NoAddrRules:{{ +}}2,
// One rule per downstream, indexed by master-port order (the xbar result's
// use-list order, here reverse of definition).
// CHECK-SAME: axi_pkg::xbar_rule_32_t [2-1:0] AddrMap
// CHECK-DAG: '{idx: 0, start_addr: 32'h1000, end_addr: 32'h2000}
// CHECK-DAG: '{idx: 1, start_addr: 32'h0, end_addr: 32'h1000}

// Both upstream managers and both downstream subordinates wire through their own
// numbered faces.
// CHECK: hw.instance "xbar_4" @axi_xbar_2u2d_a32_d64_i4_o5(clk_i: %{{[^:]+}}: i1, sub0_aw: %{{.+}}

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

// -----

// Mixed fan-out: one crossbar feeds a subordinate AND a chained crossbar at once.
// The chained downstream is routed the range reachable through it (computed from
// its subordinate, not full-range), so it coexists with the sibling subordinate.

// The leaf xbar routes its subordinate's [0x1000, 0x2000) window.
// CHECK: sv.verbatim.source @axi_xbar_1u1d_a32_d64_i4_o4_source
// CHECK-SAME: '{idx: 0, start_addr: 32'h1000, end_addr: 32'h2000}

// The fan-out xbar has two master ports: idx 0 -> the direct subordinate, idx 1
// -> the chained xbar's computed reachable range (not full-range).
// CHECK: sv.verbatim.source @axi_xbar_1u2d_a32_d64_i4_o4_source
// CHECK-DAG: '{idx: 0, start_addr: 32'h0, end_addr: 32'h1000}
// CHECK-DAG: '{idx: 1, start_addr: 32'h1000, end_addr: 32'h2000}

// CHECK: hw.instance "xbar_{{[0-9]+}}" @axi_xbar_1u2d_a32_d64_i4_o4(
// CHECK: hw.instance "xbar_{{[0-9]+}}" @axi_xbar_1u1d_a32_d64_i4_o4(

// CHECK-NOT: axi4.xbar

hw.module.extern @mgr_module(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module(in %clk : i1, in %s_axi_s0_awid : i4, in %s_axi_s0_awaddr : i32, in %s_axi_s0_awlen : i8, in %s_axi_s0_awsize : i3, in %s_axi_s0_awburst : i2, in %s_axi_s0_awlock : i1, in %s_axi_s0_awcache : i4, in %s_axi_s0_awprot : i3, in %s_axi_s0_awqos : i4, in %s_axi_s0_awregion : i4, in %s_axi_s0_awvalid : i1, out s_axi_s0_awready : i1, in %s_axi_s0_wdata : i64, in %s_axi_s0_wstrb : i8, in %s_axi_s0_wlast : i1, in %s_axi_s0_wvalid : i1, out s_axi_s0_wready : i1, out s_axi_s0_bid : i4, out s_axi_s0_bresp : i2, out s_axi_s0_bvalid : i1, in %s_axi_s0_bready : i1, in %s_axi_s0_arid : i4, in %s_axi_s0_araddr : i32, in %s_axi_s0_arlen : i8, in %s_axi_s0_arsize : i3, in %s_axi_s0_arburst : i2, in %s_axi_s0_arlock : i1, in %s_axi_s0_arcache : i4, in %s_axi_s0_arprot : i3, in %s_axi_s0_arqos : i4, in %s_axi_s0_arregion : i4, in %s_axi_s0_arvalid : i1, out s_axi_s0_arready : i1, out s_axi_s0_rid : i4, out s_axi_s0_rdata : i64, out s_axi_s0_rresp : i2, out s_axi_s0_rlast : i1, out s_axi_s0_rvalid : i1, in %s_axi_s0_rready : i1)

%clk = unrealized_conversion_cast to !axi4.clock
%mnode = axi4.node @mgr_module : !axi4.node
%snode_a = axi4.node @sub_module : !axi4.node
%snode_b = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk node %mnode {
  port_mapping = #axi4.port_wires<"clk", "m0">,
  access = [#axi4.window<base = 0, size = 8192, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
%xbar1 = axi4.xbar %clk mgrs %mgr : (!axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 4>
%xbar2 = axi4.xbar %clk mgrs %xbar1 : (!axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 4>
// %xbar1 fans out to both a direct subordinate and the chained %xbar2.
axi4.subordinate_port %xbar1, %clk node %snode_a {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate_port %xbar2, %clk node %snode_b {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
