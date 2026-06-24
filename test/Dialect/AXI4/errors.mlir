// RUN: circt-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-error @below {{'fixed' burst kind cannot have a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 4>} : () -> ()

// -----

// expected-error @below {{access windows overlap}}
%mgr = axi4.manager @manager_core {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>,
            #axi4.window<base = 2048, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

// 'outstanding_reads' of 32 exceeds 2^4 = 16 addressable by the ID width.
// expected-error @below {{outstanding_reads (32) exceeds the maximum of 2^4 (16)}}
%mgr = axi4.manager @manager_core {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 32 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>

// -----

%mgr = axi4.manager @manager_core {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4>
// expected-error @below {{outstanding_requests (32) exceeds the maximum of 2^4 (16)}}
axi4.subordinate %mgr @subordinate_peripheral {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 32 : ui32
} : !axi4.port<32, 64, 4>
