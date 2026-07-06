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
} : !axi4.port<32, 64, 4, 4, 0>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// 'outstanding_reads' of 32 exceeds 2^4 = 16 addressable by the read ID width.
// expected-error @below {{outstanding_reads (32) exceeds the maximum of 2^4 (16)}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 32 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// 'outstanding_writes' of 32 exceeds 2^4 = 16 addressable by the write ID width.
// expected-error @below {{outstanding_writes (32) exceeds the maximum of 2^4 (16)}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 32 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
