// RUN: circt-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-error @below {{'fixed' burst kind cannot have a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 4>} : () -> ()

// -----

// expected-error @below {{'incr' burst kind requires a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<incr>} : () -> ()

// -----

// expected-error @below {{'wrap' burst kind requires a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<wrap>} : () -> ()
