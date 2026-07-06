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
