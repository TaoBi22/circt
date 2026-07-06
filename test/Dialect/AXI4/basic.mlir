// RUN: circt-opt %s --allow-unregistered-dialect | circt-opt --allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// CHECK: unrealized_conversion_cast to !axi4.clock
%c = unrealized_conversion_cast to !axi4.clock

// CHECK: unrealized_conversion_cast to !axi4.reset
%r = unrealized_conversion_cast to !axi4.reset

// CHECK: unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0, 0, 0>
%p = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0, 0, 0>

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

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

hw.module.extern @mgr_module()
hw.module.extern @sub_module()

// CHECK: %[[MGRNODE:.+]] = axi4.node @mgr_module : !axi4.node
%mgr_node = axi4.node @mgr_module : !axi4.node
// CHECK: %[[SUBNODE:.+]] = axi4.node @sub_module : !axi4.node
%sub_node = axi4.node @sub_module : !axi4.node

// CHECK: %[[MGR:.+]] = axi4.manager_port %[[MGRNODE]] %[[CLK:.+]], %[[RST:.+]] {access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>], outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32} : !axi4.port<32, 64, 4, 4, 0, 0, 0>
%mgr = axi4.manager_port %mgr_node %c, %r {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0, 0, 0>

// CHECK: axi4.subordinate_port %[[MGR]] %[[SUBNODE]] %[[CLK]], %[[RST]] {access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>], outstanding_requests = 4 : ui32} : !axi4.port<32, 64, 4, 4, 0, 0, 0>
axi4.subordinate_port %mgr %sub_node %c, %r {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0, 0, 0>
