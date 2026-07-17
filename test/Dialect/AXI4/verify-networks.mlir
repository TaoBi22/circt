// RUN: circt-opt %s --verify-axi4-networks | FileCheck %s

// Verification-only pass: valid networks pass through unchanged.

hw.module.extern @mgr_module()
hw.module.extern @sub_module()

// A manager driving a single subordinate whose window matches exactly.
// CHECK: axi4.manager @mgr_module
%m1 = axi4.manager @mgr_module {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate %m1 @sub_module {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// Fan-out: an xbar output feeds two subordinates that partition the manager's
// address space, with matching incrementing bursts.
// CHECK: axi4.xbar
%m2 = axi4.manager @mgr_module {
  access = [#axi4.window<base = 0, size = 8192, burst_specs = [<incr, len = 16>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
%x = axi4.xbar %m2 : !axi4.port<32, 64, 4>
axi4.subordinate %x @sub_module {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<incr, len = 16>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate %x @sub_module {
  access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<incr, len = 16>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// Fan-in: two managers with disjoint windows merge through an xbar into a
// subordinate that covers their union.
// CHECK: axi4.subordinate
%ma = axi4.manager @mgr_module {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
%mb = axi4.manager @mgr_module {
  access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
%xm = axi4.xbar %ma, %mb : !axi4.port<32, 64, 4>
axi4.subordinate %xm @sub_module {
  access = [#axi4.window<base = 0, size = 8192, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
