// RUN: circt-opt %s --allow-unregistered-dialect | circt-opt --allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// CHECK: unrealized_conversion_cast to !axi4.clock
%c = unrealized_conversion_cast to !axi4.clock

// CHECK: unrealized_conversion_cast to !axi4.port<32, 64, 4>
%p = unrealized_conversion_cast to !axi4.port<32, 64, 4>

// CHECK: unrealized_conversion_cast to !axi4.node
%n = unrealized_conversion_cast to !axi4.node

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// CHECK: #axi4.burst_spec<fixed>
"test.attrs"() {a = #axi4.burst_spec<fixed>} : () -> ()
// CHECK: #axi4.burst_spec<incr, len = 256>
"test.attrs"() {a = #axi4.burst_spec<incr, len = 256>} : () -> ()
// CHECK: #axi4.burst_spec<wrap, len = 16>
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 16>} : () -> ()

// CHECK: #axi4.window<base = 16384, size = 256, burst_specs = [<fixed>]>
"test.attrs"() {a = #axi4.window<base = 0x4000, size = 0x100, burst_specs = [<fixed>]>} : () -> ()
// CHECK: #axi4.window<base = 16384, size = 256, burst_specs = [<wrap, len = 256>, <incr, len = 256>]>
"test.attrs"() {a = #axi4.window<base = 0x4000, size = 0x100, burst_specs = [<wrap, len = 256>, <incr, len = 256>]>} : () -> ()

// CHECK: #axi4.port_wires<"clk", "axi_in">
"test.attrs"() {a = #axi4.port_wires<"clk", "axi_in">} : () -> ()

// CHECK: #axi4.req_resp_structs<"clk", "axi_sub_req_i", "axi_sub_resp_o">
"test.attrs"() {a = #axi4.req_resp_structs<"clk", "axi_sub_req_i", "axi_sub_resp_o">} : () -> ()

// CHECK: #axi4.port_interface<"clk", "axi_in_if">
"test.attrs"() {a = #axi4.port_interface<"clk", "axi_in_if">} : () -> ()

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

hw.module.extern @mgr_module()
hw.module.extern @sub_module()

// CHECK: %[[MGRNODE:.+]] = axi4.node @mgr_module : !axi4.node
%mgr_node = axi4.node @mgr_module : !axi4.node
// CHECK: %[[SUBNODE:.+]] = axi4.node @sub_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node

// CHECK: %[[MGR0:.+]] = axi4.manager_port %[[CLK:.+]] node %[[MGRNODE]] {access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>], outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32, port_mapping = #axi4.port_wires<"clk", "axi_in0">} : !axi4.port<32, 64, 4>
%mgr0 = axi4.manager_port %c node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "axi_in0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// CHECK: %[[MGR1:.+]] = axi4.manager_port %[[CLK]] node %[[MGRNODE]] {access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<fixed>]>], outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32, port_mapping = #axi4.port_wires<"clk", "axi_in1">} : !axi4.port<32, 64, 4>
%mgr1 = axi4.manager_port %c node %mgr_node {
  port_mapping = #axi4.port_wires<"clk", "axi_in1">,
  access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// CHECK: %[[XBAR:.+]] = axi4.xbar %[[CLK]] mgrs %[[MGR0]], %[[MGR1]] : (!axi4.port<32, 64, 4>, !axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 5>
%xbar = axi4.xbar %c mgrs %mgr0, %mgr1 : (!axi4.port<32, 64, 4>, !axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 5>

// CHECK: axi4.subordinate_port %[[XBAR]], %[[CLK]] node %[[SUBNODE]] {access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>], outstanding_requests = 4 : ui32, port_mapping = #axi4.req_resp_structs<"clk", "axi_sub_req_i", "axi_sub_resp_o">} : !axi4.port<32, 64, 5>
axi4.subordinate_port %xbar, %c node %sub_node {
  port_mapping = #axi4.req_resp_structs<"clk", "axi_sub_req_i", "axi_sub_resp_o">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 5>

//===----------------------------------------------------------------------===//
// Nodeless ports
//===----------------------------------------------------------------------===//

// CHECK: %[[NODELESS_MGR:.+]] = axi4.manager_port %[[CLK]] {access = [#axi4.window<base = 8192, size = 4096, burst_specs = [<fixed>]>], outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32} : !axi4.port<32, 64, 4>
%nodeless_mgr = axi4.manager_port %c {
  access = [#axi4.window<base = 8192, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// CHECK: axi4.subordinate_port %[[NODELESS_MGR]], %[[CLK]] {access = [#axi4.window<base = 8192, size = 4096, burst_specs = [<fixed>]>], outstanding_requests = 4 : ui32} : !axi4.port<32, 64, 4>
axi4.subordinate_port %nodeless_mgr, %c {
  access = [#axi4.window<base = 8192, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
