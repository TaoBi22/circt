// RUN: circt-opt %s --split-input-file --lower-axi4-to-hw | FileCheck %s

// A manager -> xbar -> subordinate network. The xbar lowers to PULP `axi_xbar`'s
// interface: the slave ports face the upstream managers, the master ports the
// downstream subordinates, each side an array of packed req/resp structs. Every
// channel carries a `user` sideband and the AW channel a 6-bit `atop` field our
// canonical form lacks (both tied to 0).

// CHECK-LABEL: hw.module.extern @axi_xbar_1u1d_a32_d64_i4_o4(
// CHECK-SAME: in %clk_i : i1
// CHECK-SAME: in %slv_ports_req_i : !hw.array<1xstruct<aw: !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, atop: i6, user: i1>, aw_valid: i1, w: !hw.struct<data: i64, strb: i8, last: i1, user: i1>, w_valid: i1, b_ready: i1, ar: !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i1>, ar_valid: i1, r_ready: i1>>
// CHECK-SAME: out slv_ports_resp_o : !hw.array<1xstruct<aw_ready: i1, ar_ready: i1, w_ready: i1, b_valid: i1, b: !hw.struct<id: i4, resp: i2, user: i1>, r_valid: i1, r: !hw.struct<id: i4, data: i64, resp: i2, last: i1, user: i1>>>
// CHECK-SAME: out mst_ports_req_o : !hw.array<1xstruct<aw: !hw.struct<{{.*}}, atop: i6, user: i1>, aw_valid: i1, {{.*}}, r_ready: i1>>
// CHECK-SAME: in %mst_ports_resp_i : !hw.array<1xstruct<aw_ready: i1, {{.*}}, r: !hw.struct<id: i4, data: i64, resp: i2, last: i1, user: i1>>>

// The upstream manager's request channels are packed into a req struct - AW
// first bridged to `aw_chan_t` with atop and user tied to 0 - and arrayed into
// the slave request port. The subordinate's responses are likewise packed into
// the master response port.
// CHECK-DAG: %[[ATOP:.+]] = hw.constant 0 : i6
// CHECK-DAG: %[[AWCHAN:.+]] = hw.struct_create ({{.*}}, %[[ATOP]], %{{.+}}) : !hw.struct<id: i4, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, atop: i6, user: i1>
// CHECK-DAG: %[[REQ:.+]] = hw.struct_create (%[[AWCHAN]], {{.*}}) : !hw.struct<aw: {{.*}}, r_ready: i1>
// CHECK-DAG: %[[SLVREQ:.+]] = hw.array_create %[[REQ]] : !hw.struct<aw: {{.*}}, r_ready: i1>
// CHECK-DAG: %[[MSTRESP:.+]] = hw.array_create %{{.+}} : !hw.struct<aw_ready: i1, {{.*}}>

// CHECK: hw.instance "xbar_2" @axi_xbar_1u1d_a32_d64_i4_o4(clk_i: %{{.+}}: i1, slv_ports_req_i: %[[SLVREQ]]: {{.*}}, mst_ports_resp_i: %[[MSTRESP]]: {{.*}}) ->

// The slave response and master request arrays are unpacked per element; the
// master AW is bridged back to our canonical payload, dropping atop.
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

// Chained crossbars: manager -> xbar -> xbar -> subordinate. Both xbars share
// one extern, and the xbar->xbar edge packs the first's master request array
// back into the second's slave request array.

// One extern is emitted for both crossbars.
// CHECK-LABEL: hw.module.extern @axi_xbar_1u1d_a32_d64_i4_o4(
// CHECK-NOT: hw.module.extern @axi_xbar

// CHECK: hw.instance "xbar_2" @axi_xbar_1u1d_a32_d64_i4_o4(

// The first xbar's master request feeds the second xbar's slave request: its AW
// is unpacked (atop dropped) then repacked with atop tied to 0.
// CHECK: %[[X1MSTREQ:.+]] = hw.array_get %xbar_2.mst_ports_req_o[%{{.+}}]
// CHECK: %[[X1AW:.+]] = hw.struct_extract %[[X1MSTREQ]]["aw"]
// CHECK: %[[X1AWID:.+]] = hw.struct_extract %[[X1AW]]["id"] : !hw.struct<id: i4, {{.*}}, atop: i6, user: i1>

// CHECK: hw.instance "xbar_3" @axi_xbar_1u1d_a32_d64_i4_o4(

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
