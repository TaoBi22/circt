// RUN: circt-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect --lower-axi4-to-hw

// Multiple clock domains.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk0 = unrealized_conversion_cast to !axi4.clock
%clk1 = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk0, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
// expected-error @below {{multiple clock domains are not yet supported}}
axi4.subordinate_port %mgr, %clk1, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_out">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

// Multiple reset domains.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst0 = unrealized_conversion_cast to !axi4.reset
%rst1 = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst0 node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
// expected-error @below {{multiple reset domains are not yet supported}}
axi4.subordinate_port %mgr, %clk, %rst1 node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_out">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

// axi4 result used by a non-axi4 op.
hw.module.extern @mgr_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
// expected-error @below {{results of axi4 operations may only be used by axi4 operations}}
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
"test.foo"(%mgr) : (!axi4.port<32, 64, 4, 4, 0>) -> ()

// -----

// Nodeless port.
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{nodeless ports are not yet supported}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %mgr, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_out">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

// Manager port with no uses.
hw.module.extern @mgr_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
// expected-error @below {{axi4.manager_ports with no uses are not yet supported}}
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

// Unsupported req_resp_structs mapping.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{only the 'port_wires' port_mapping is supported}}
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.req_resp_structs<"clk", "rst_ni", "axi_req_i", "axi_resp_o">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %mgr, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_out">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

// Unsupported port_interface mapping.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
// expected-error @below {{only the 'port_wires' port_mapping is supported}}
axi4.subordinate_port %mgr, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_interface<"clk", "rst_ni", "axi_if">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

// Node module missing a port required by the port_wires mapping.
hw.module.extern @mgr_module(in %clk : i1, in %rst_ni : i1)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{referenced module has no port 'm_axi_m0_awid' required by the port_wires mapping}}
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
axi4.subordinate_port %mgr, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>

// -----

// Node module port declared with the wrong direction.
hw.module.extern @mgr_module(in %clk : i1, in %rst_ni : i1, in %m_axi_m0_awid : i4)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{module port 'm_axi_m0_awid' has the wrong direction}}
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
axi4.subordinate_port %mgr, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>

// -----

// Network operations split across regions. The whole network lowers into one
// block, so ops in a different region (here, a stray node nested in a second
// hw.module) are rejected.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %mgr, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
hw.module @Other() {
  // expected-error @below {{all axi4 network operations must be in the same region}}
  %other_node = axi4.node @mgr_module : !axi4.node
  hw.output
}

// -----

// Node module missing the reset port named by the port_wires mapping.
hw.module.extern @mgr_module(in %clk : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awuser : i2, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wuser : i3, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_buser : i1, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_aruser : i2, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_ruser : i4, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{referenced module has no reset port 'rst_ni'}}
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
axi4.subordinate_port %mgr, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>

// -----

// Node module missing the clock port named by the port_wires mapping.
hw.module.extern @mgr_module(in %rst_ni : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awuser : i2, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wuser : i3, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_buser : i1, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_aruser : i2, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_ruser : i4, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{referenced module has no clock port 'clk'}}
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
axi4.subordinate_port %mgr, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>

// -----

// Node module clock port is not lowerable (not i1).
hw.module.extern @mgr_module(in %clk : i2, in %rst_ni : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awuser : i2, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wuser : i3, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_buser : i1, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_aruser : i2, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_ruser : i4, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
// expected-error @below {{unsupported clock port type}}
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
axi4.subordinate_port %mgr, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>

// -----

// Node module has an input port not driven by any AXI4 port.
hw.module.extern @mgr_module(in %clk : i1, in %rst_ni : i1, in %extra : i1, out m_axi_m0_awid : i4, out m_axi_m0_awaddr : i32, out m_axi_m0_awlen : i8, out m_axi_m0_awsize : i3, out m_axi_m0_awburst : i2, out m_axi_m0_awlock : i1, out m_axi_m0_awcache : i4, out m_axi_m0_awprot : i3, out m_axi_m0_awqos : i4, out m_axi_m0_awregion : i4, out m_axi_m0_awuser : i2, out m_axi_m0_awvalid : i1, in %m_axi_m0_awready : i1, out m_axi_m0_wdata : i64, out m_axi_m0_wstrb : i8, out m_axi_m0_wlast : i1, out m_axi_m0_wuser : i3, out m_axi_m0_wvalid : i1, in %m_axi_m0_wready : i1, in %m_axi_m0_bid : i4, in %m_axi_m0_bresp : i2, in %m_axi_m0_buser : i1, in %m_axi_m0_bvalid : i1, out m_axi_m0_bready : i1, out m_axi_m0_arid : i4, out m_axi_m0_araddr : i32, out m_axi_m0_arlen : i8, out m_axi_m0_arsize : i3, out m_axi_m0_arburst : i2, out m_axi_m0_arlock : i1, out m_axi_m0_arcache : i4, out m_axi_m0_arprot : i3, out m_axi_m0_arqos : i4, out m_axi_m0_arregion : i4, out m_axi_m0_aruser : i2, out m_axi_m0_arvalid : i1, in %m_axi_m0_arready : i1, in %m_axi_m0_rid : i4, in %m_axi_m0_rdata : i64, in %m_axi_m0_rresp : i2, in %m_axi_m0_rlast : i1, in %m_axi_m0_ruser : i4, in %m_axi_m0_rvalid : i1, out m_axi_m0_rready : i1)
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{module input port 'extra' is not driven by any AXI4 port}}
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
axi4.subordinate_port %mgr, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>

// -----

// Crossbar with overlapping downstream address windows.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node0 = axi4.node @sub_module : !axi4.node
%sub_node1 = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
// expected-error @below {{overlapping address windows in the crossbar address map}}
%xbar = axi4.xbar %clk, %rst mgrs %mgr : (!axi4.port<32, 64, 4, 4, 0>) -> !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %xbar, %clk, %rst node %sub_node0 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %xbar, %clk, %rst node %sub_node1 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 2048, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

// Crossbar address width wider than the PULP rule type supports.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<128, 64, 4, 4, 0>
// expected-error @below {{address widths wider than 64 bits are not supported}}
%xbar = axi4.xbar %clk, %rst mgrs %mgr : (!axi4.port<128, 64, 4, 4, 0>) -> !axi4.port<128, 64, 4, 4, 0>
axi4.subordinate_port %xbar, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<128, 64, 4, 4, 0>

// -----

// Crossbar port with mismatched write and read ID widths (PULP has one ID
// width per side).
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 5, 0>
// expected-error @below {{the PULP axi_xbar uses a single ID width per side, so the upstream write ID width (4) and read ID width (5) must match}}
%xbar = axi4.xbar %clk, %rst mgrs %mgr : (!axi4.port<32, 64, 4, 5, 0>) -> !axi4.port<32, 64, 4, 5, 0>
axi4.subordinate_port %xbar, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 5, 0>

// -----

// Access window that runs past the top of the port's address space.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
%xbar = axi4.xbar %clk, %rst mgrs %mgr : (!axi4.port<32, 64, 4, 4, 0>) -> !axi4.port<32, 64, 4, 4, 0>
// expected-error @below {{access window [base 4294963200, size 8192) does not fit the 32-bit address space}}
axi4.subordinate_port %xbar, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 4294963200, size = 8192, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

// Crossbar with mismatched upstream and downstream user widths (PULP routes
// user unchanged over a single user width).
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr_node = axi4.node @mgr_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node
%mgr = axi4.manager_port %clk, %rst node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 4>
// expected-error @below {{the PULP axi_xbar routes user unchanged over a single user width, so the upstream user width (4) and master user width (2) must match}}
%xbar = axi4.xbar %clk, %rst mgrs %mgr : (!axi4.port<32, 64, 4, 4, 4>) -> !axi4.port<32, 64, 4, 4, 2>
axi4.subordinate_port %xbar, %clk, %rst node %sub_node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 2>
