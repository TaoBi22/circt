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
// expected-error @below {{only the 'port_wires' port_mapping is supported}}
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
// expected-error @below {{only the 'port_wires' port_mapping is supported}}
axi4.subordinate_port %mgr, %clk node %sub_node {
  port_mapping = #axi4.port_interface<"clk", "axi_if">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
