// RUN: circt-opt -split-input-file -convert-fsm-to-core -verify-diagnostics %s

fsm.machine @foo(%arg0: i1) -> (i1) attributes {initialState = "A"} {
  // expected-error @below {{'arith.constant' op is unsupported (op from the arith dialect).}}
  %true = arith.constant true
  fsm.state @A output  {
    fsm.output %true : i1
  } transitions {
  }
}

// -----

// expected-error @below {{stateType attribute does not name a type}}
fsm.machine @foo(%arg0: i1) -> (i1) attributes {initialState = "A", stateType = "I am not a type"} {
  %true = hw.constant true
  fsm.state @A output  {
    fsm.output %true : i1
  } transitions {
  }
}

// -----

// expected-error @below {{stateType attribute must name an integer type}}
fsm.machine @foo(%arg0: i1) -> (i1) attributes {initialState = "A", stateType = !seq.clock} {
  %true = hw.constant true
  fsm.state @A output  {
    fsm.output %true : i1
  } transitions {
  }
}
