// RUN: circt-opt %s --allow-unregistered-dialect | circt-opt --allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// CHECK: unrealized_conversion_cast to !axi4.port<32, 64, 4>
%p = unrealized_conversion_cast to !axi4.port<32, 64, 4>

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

// CHECK: %[[MGR:.+]] = axi4.manager @manager_core {access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>], outstanding_reads = 4 : ui32, outstanding_writes = 4 : ui32} : !axi4.port<32, 64, 4>
%mgr = axi4.manager @manager_core {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// CHECK: axi4.subordinate %[[MGR]] @subordinate_peripheral {access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>], outstanding_requests = 4 : ui32} : !axi4.port<32, 64, 4>
axi4.subordinate %mgr @subordinate_peripheral {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
