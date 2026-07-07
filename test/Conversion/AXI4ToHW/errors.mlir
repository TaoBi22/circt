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
