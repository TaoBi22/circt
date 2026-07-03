// RUN: circt-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-error @below {{'fixed' burst kind cannot have a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 4>} : () -> ()

// -----

// expected-error @below {{'incr' burst kind requires a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<incr>} : () -> ()

// -----

// expected-error @below {{'wrap' burst kind requires a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<wrap>} : () -> ()

// -----

hw.module.extern @mgr_module()
%clk = unrealized_conversion_cast to !axi4.clock
// expected-error @below {{access windows overlap}}
%mgr = axi4.manager @mgr_module %clk {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>,
            #axi4.window<base = 2048, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

hw.module.extern @mgr_module()
%clk = unrealized_conversion_cast to !axi4.clock
// expected-error @below {{outstanding_reads (32) exceeds the maximum of 2^4 (16)}}
%mgr = axi4.manager @mgr_module %clk {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 32 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
%mgr = axi4.manager @mgr_module %clk {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
// expected-error @below {{outstanding_requests (32) exceeds the maximum of 2^4 (16)}}
axi4.subordinate %mgr @sub_module %clk {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 32 : ui32
} : !axi4.port<32, 64, 4>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
// expected-error @below {{references unknown symbol @does_not_exist}}
%mgr = axi4.manager @does_not_exist %clk {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

func.func private @not_a_module()
%clk = unrealized_conversion_cast to !axi4.clock
// expected-error @below {{symbol @not_a_module must refer to an 'hw.module'}}
%mgr = axi4.manager @not_a_module %clk {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

%a = unrealized_conversion_cast to !axi4.port<32, 64, 4>
%b = unrealized_conversion_cast to !axi4.port<32, 64, 4>
%clk = unrealized_conversion_cast to !axi4.clock
// expected-error @below {{Xbar return type's id width must be at least the input id width + ceil(log2(number of managers)) (i.e., 5)}}
%xbar = "axi4.xbar"(%clk, %a, %b) : (!axi4.clock, !axi4.port<32, 64, 4>, !axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 4>

// -----

%a = unrealized_conversion_cast to !axi4.port<32, 64, 4>
%b = unrealized_conversion_cast to !axi4.port<32, 64, 5>
%clk = unrealized_conversion_cast to !axi4.clock
// expected-error @below {{all upstream ports must have the same type}}
%xbar = "axi4.xbar"(%clk, %a, %b) : (!axi4.clock, !axi4.port<32, 64, 4>, !axi4.port<32, 64, 5>) -> !axi4.port<32, 64, 5>

// -----

hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%clk = unrealized_conversion_cast to !axi4.clock
// expected-error @below {{result must have at most one use}}
%mgr = axi4.manager @mgr_module %clk {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate %mgr @sub_module %clk {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
axi4.subordinate %mgr @sub_module %clk {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4>
