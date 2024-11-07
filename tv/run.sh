#!/usr/bin/bash

circt-opt --lower-smt-to-z3-llvm tv.mlir --reconcile-unrealized-casts > exec.mlir
mlir-cpu-runner exec.mlir -e fsm10 -shared-libs=/usr/lib/libz3.so -entry-point-result=void