// RUN: circt-opt %s --verify-axi4-networks | FileCheck %s

// Verification-only pass: valid networks pass through unchanged.

hw.module.extern @mgr_module()
hw.module.extern @sub_module()

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset

// A manager driving a single subordinate whose window matches exactly.
// CHECK: axi4.manager_port
%mn1 = axi4.node @mgr_module : !axi4.node
%sn1 = axi4.node @sub_module : !axi4.node
%m1 = axi4.manager_port %clk, %rst node %mn1 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %m1, %clk, %rst node %sn1 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// Fan-out: an xbar output feeds two subordinates that partition the manager's
// address space, with matching incrementing bursts.
// CHECK: axi4.xbar
%mn2 = axi4.node @mgr_module : !axi4.node
%sn2 = axi4.node @sub_module : !axi4.node
%sn3 = axi4.node @sub_module : !axi4.node
%m2 = axi4.manager_port %clk, %rst node %mn2 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 8192, burst_specs = [<incr, len = 16>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
%x = axi4.xbar %clk, %rst mgrs %m2 : (!axi4.port<32, 64, 4, 4, 0>) -> !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %x, %clk, %rst node %sn2 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<incr, len = 16>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %x, %clk, %rst node %sn3 {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<incr, len = 16>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// Fan-in: two managers with disjoint windows merge through an xbar into a
// subordinate that covers their union.
// CHECK: axi4.subordinate_port
%mna = axi4.node @mgr_module : !axi4.node
%mnb = axi4.node @mgr_module : !axi4.node
%sna = axi4.node @sub_module : !axi4.node
%ma = axi4.manager_port %clk, %rst node %mna {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
%mb = axi4.manager_port %clk, %rst node %mnb {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "m0">,
  access = [#axi4.window<base = 4096, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
%xm = axi4.xbar %clk, %rst mgrs %ma, %mb : (!axi4.port<32, 64, 4, 4, 0>, !axi4.port<32, 64, 4, 4, 0>) -> !axi4.port<32, 64, 5, 5, 0>
axi4.subordinate_port %xm, %clk, %rst node %sna {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "s0">,
  access = [#axi4.window<base = 0, size = 8192, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 5, 5, 0>
