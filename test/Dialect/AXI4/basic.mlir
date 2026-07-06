// RUN: circt-opt %s --allow-unregistered-dialect | circt-opt --allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// CHECK: unrealized_conversion_cast to !axi4.clock
%c = unrealized_conversion_cast to !axi4.clock

// CHECK: unrealized_conversion_cast to !axi4.reset
%r = unrealized_conversion_cast to !axi4.reset

// CHECK: unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0>
%p = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0>

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

// CHECK: #axi4.port_struct<"clk", "rst_ni", "axi_in">
"test.attrs"() {a = #axi4.port_struct<"clk", "rst_ni", "axi_in">} : () -> ()

// CHECK: #axi4.req_resp_structs<"clk", "rst_ni", "axi_sub_req_i", "axi_sub_resp_o">
"test.attrs"() {a = #axi4.req_resp_structs<"clk", "rst_ni", "axi_sub_req_i", "axi_sub_resp_o">} : () -> ()

// CHECK: #axi4.port_interface<"clk", "rst_ni", "axi_in_if">
"test.attrs"() {a = #axi4.port_interface<"clk", "rst_ni", "axi_in_if">} : () -> ()

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

hw.module.extern @my_module()

// CHECK: %[[NODE:.+]] = axi4.node @my_module : !axi4.node
%node = axi4.node @my_module : !axi4.node
