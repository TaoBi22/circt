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

// expected-error @below {{references unknown symbol @does_not_exist}}
%node = axi4.node @does_not_exist : !axi4.node

// -----

func.func private @not_a_module()
// expected-error @below {{symbol @not_a_module must refer to an 'hw.module'}}
%node = axi4.node @not_a_module : !axi4.node

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{access windows overlap}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>,
            #axi4.window<base = 2048, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0, 0, 0>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// 'outstanding_reads' of 32 exceeds 2^4 = 16 addressable by the read ID width.
// expected-error @below {{outstanding_reads (32) exceeds the maximum of 2^4 (16)}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 32 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0, 0, 0>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// 'outstanding_writes' of 32 exceeds 2^4 = 16 addressable by the write ID width.
// expected-error @below {{outstanding_writes (32) exceeds the maximum of 2^4 (16)}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 32 : ui32
} : !axi4.port<32, 64, 4, 4, 0, 0, 0>

// -----

%a = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0, 0, 0>
%b = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0, 0, 0>
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{xbar return type's write id width must be at least the input write id width + ceil(log2(number of managers)) (i.e., 5)}}
%xbar = "axi4.xbar"(%clk, %rst, %a, %b) : (!axi4.clock, !axi4.reset, !axi4.port<32, 64, 4, 4, 0, 0, 0>, !axi4.port<32, 64, 4, 4, 0, 0, 0>) -> !axi4.port<32, 64, 4, 4, 0, 0, 0>

// -----

%a = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0, 0, 0>
%b = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0, 0, 0>
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{xbar return type's read id width must be at least the input read id width + ceil(log2(number of managers)) (i.e., 5)}}
%xbar = "axi4.xbar"(%clk, %rst, %a, %b) : (!axi4.clock, !axi4.reset, !axi4.port<32, 64, 4, 4, 0, 0, 0>, !axi4.port<32, 64, 4, 4, 0, 0, 0>) -> !axi4.port<32, 64, 5, 4, 0, 0, 0>

// -----

%a = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0, 0, 0>
%b = unrealized_conversion_cast to !axi4.port<32, 64, 5, 5, 0, 0, 0>
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{all upstream ports must have the same type}}
%xbar = "axi4.xbar"(%clk, %rst, %a, %b) : (!axi4.clock, !axi4.reset, !axi4.port<32, 64, 4, 4, 0, 0, 0>, !axi4.port<32, 64, 5, 5, 0, 0, 0>) -> !axi4.port<32, 64, 5, 5, 0, 0, 0>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{result must have at most one use}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0, 0, 0>
axi4.subordinate_port %mgr, %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0, 0, 0>
axi4.subordinate_port %mgr, %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0, 0, 0>
