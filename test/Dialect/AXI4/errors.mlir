// RUN: circt-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-error @below {{'fixed' burst kind cannot have a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 4>} : () -> ()

// -----

hw.module.extern @mgr_module()
// expected-error @below {{access windows overlap}}
%mgr = axi4.manager @mgr_module {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>,
            #axi4.window<base = 2048, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

hw.module.extern @mgr_module()
// expected-error @below {{outstanding_reads (32) exceeds the maximum of 2^4 (16)}}
%mgr = axi4.manager @mgr_module {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 32 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

hw.module.extern @mgr_module()
hw.module.extern @sub_module()
%mgr = axi4.manager @mgr_module {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
// expected-error @below {{outstanding_requests (32) exceeds the maximum of 2^4 (16)}}
axi4.subordinate %mgr @sub_module {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 32 : ui32
} : !axi4.port<32, 64, 4>

// -----

// expected-error @below {{references unknown symbol @does_not_exist}}
%mgr = axi4.manager @does_not_exist {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

func.func private @not_a_module()
// expected-error @below {{symbol @not_a_module must refer to an 'hw.module'}}
%mgr = axi4.manager @not_a_module {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

%a = unrealized_conversion_cast to !axi4.port<32, 64, 4>
%b = unrealized_conversion_cast to !axi4.port<32, 64, 8>
// expected-error @below {{requires the same type for all operands and results}}
%xbar = "axi4.xbar"(%a, %b) : (!axi4.port<32, 64, 4>, !axi4.port<32, 64, 8>) -> !axi4.port<32, 64, 4>
