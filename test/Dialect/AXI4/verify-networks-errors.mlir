// RUN: circt-opt %s --verify-axi4-networks --split-input-file --verify-diagnostics

// A subordinate handling addresses outside what any reaching manager issues to.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%m = axi4.manager @mgr_module {
  access = [#axi4.window<base = 0, size = 16, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
// expected-error @below {{handles addresses [0, 32) that no manager issues to}}
axi4.subordinate %m @sub_module {
  access = [#axi4.window<base = 0, size = 32, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// A subordinate that doesn't support a burst kind a manager issues.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%m = axi4.manager @mgr_module {
  access = [#axi4.window<base = 0, size = 16, burst_specs = [<incr, len = 4>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
// expected-error @below {{does not support the 'incr' burst issued by @mgr_module to addresses [0, 16)}}
axi4.subordinate %m @sub_module {
  access = [#axi4.window<base = 0, size = 16, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// A manager issuing to addresses no subordinate handles.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
// expected-error @below {{issues to addresses [0, 32) that are not handled by any subordinate}}
%m = axi4.manager @mgr_module {
  access = [#axi4.window<base = 0, size = 32, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate %m @sub_module {
  access = [#axi4.window<base = 0, size = 16, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// A manager issuing to an address two subordinates both claim.
hw.module.extern @mgr_module()
hw.module.extern @sub_module()
// expected-error @below {{issues to addresses [8, 12) that are handled by multiple subordinates}}
%m = axi4.manager @mgr_module {
  access = [#axi4.window<base = 0, size = 16, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
%x = axi4.xbar %m : !axi4.port<32, 64, 4>
axi4.subordinate %x @sub_module {
  access = [#axi4.window<base = 0, size = 12, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate %x @sub_module {
  access = [#axi4.window<base = 8, size = 8, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// Two managers whose windows overlap where they merge at an xbar.
hw.module.extern @m0()
hw.module.extern @m1()
%a = axi4.manager @m0 {
  access = [#axi4.window<base = 0, size = 16, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
%b = axi4.manager @m1 {
  access = [#axi4.window<base = 8, size = 16, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
// expected-error @below {{address range [8, 24) is reachable from both @m1 and @m0}}
%x = axi4.xbar %a, %b : !axi4.port<32, 64, 4>

// -----

// A cyclic network: two xbars feeding each other.
hw.module @cyclic() {
  // expected-error @below {{is part of a cyclic AXI4 network}}
  %x0 = axi4.xbar %x1 : !axi4.port<32, 64, 4>
  // expected-error @below {{is part of a cyclic AXI4 network}}
  %x1 = axi4.xbar %x0 : !axi4.port<32, 64, 4>
  hw.output
}

// -----

// An op the pass doesn't know how to route addresses through.
// expected-error @below {{unsupported AXI4 network op; cannot verify how it routes addresses}}
%p = unrealized_conversion_cast to !axi4.port<32, 64, 4>
