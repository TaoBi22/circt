// RUN: circt-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect --lower-axi4-to-hw

// Multiple clock domains.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk0 = unrealized_conversion_cast to !axi4.clock
%clk1 = unrealized_conversion_cast to !axi4.clock
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk0 node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
// expected-error @below {{multiple clock domains are not yet supported}}
axi4.subordinate_port %mgr, %clk1 node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "axi_out">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// axi4 result used by a non-axi4 op.
hw.module.extern @mgr_module()
%clk = unrealized_conversion_cast to !axi4.clock
%mgr_node = axi4.node @mgr_module : !axi4.node
// expected-error @below {{results of axi4 operations may only be used by axi4 operations}}
%mgr = axi4.manager_port %clk node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
"test.foo"(%mgr) : (!axi4.port<32, 64, 4>) -> ()

// -----

// Nodeless port.
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{nodeless ports are not yet supported}}
%mgr = axi4.manager_port %clk {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate_port %mgr, %clk node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "axi_out">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Manager port with no uses.
hw.module.extern @mgr_module()
%clk = unrealized_conversion_cast to !axi4.clock
%mgr_node = axi4.node @mgr_module : !axi4.node
// expected-error @below {{axi4.manager_ports with no uses are not yet supported}}
%mgr = axi4.manager_port %clk node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Unsupported req_resp_structs mapping.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{only the 'port_wires' port_mapping is currently supported}}
%mgr = axi4.manager_port %clk node %mgr_node {
  port_mapping = #axi4.req_resp_structs<"clk", "axi_req_i", "axi_resp_o">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate_port %mgr, %clk node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "axi_out">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Unsupported port_interface mapping.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
// expected-error @below {{only the 'port_wires' port_mapping is currently supported}}
axi4.subordinate_port %mgr, %clk node %sub_node {
  port_mapping = #axi4.port_interface<"clk", "axi_if">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Node module missing a port required by the port_wires mapping.
hw.module.extern @mgr_module(in %clk : i1)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{referenced module has no port 'm_axi_m0_awid' required by the port_wires mapping}}
%mgr = axi4.manager_port %clk node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate_port %mgr, %clk node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Node module port declared with the wrong direction.
hw.module.extern @mgr_module(in %clk : i1, in %m_axi_m0_awid : i4)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{module port 'm_axi_m0_awid' has the wrong direction}}
%mgr = axi4.manager_port %clk node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate_port %mgr, %clk node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Node module missing the clock port named by the port_wires mapping.
hw.module.extern @mgr_module(out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{referenced module has no clock port 'clk'}}
%mgr = axi4.manager_port %clk node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate_port %mgr, %clk node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Node module clock port is not lowerable (not i1).
hw.module.extern @mgr_module(in %clk : i2, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{unsupported clock port type}}
%mgr = axi4.manager_port %clk node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate_port %mgr, %clk node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Node module has an input port not driven by any AXI4 port.
hw.module.extern @mgr_module(in %clk : i1, in %extra : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
// expected-error @below {{module input port 'extra' is not driven by any AXI4 port}}
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate_port %mgr, %clk node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Xbar lowering is not yet implemented.
hw.module.extern @mgr_module()
%clk = unrealized_conversion_cast to !axi4.clock
%mgr_node = axi4.node @mgr_module : !axi4.node
%mgr = axi4.manager_port %clk node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
// expected-error @below {{xbar lowering is not yet implemented}}
%xbar = axi4.xbar %clk mgrs %mgr : (!axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 4>
