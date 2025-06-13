#!/usr/bin/bash

../../build/bin/circt-opt --lower-smt-to-z3-llvm tv.mlir --reconcile-unrealized-casts > exec.mlir
../../llvm/build/bin/mlir-cpu-runner exec.mlir -e fsm10 -shared-libs=/usr/lib/x86_64-linux-gnu/libz3.so -entry-point-result=void