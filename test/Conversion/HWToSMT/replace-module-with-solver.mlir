// RUN: circt-opt %s --convert-hw-to-smt='replace-module-with-solver' | FileCheck %s

// CHECK: aa
hw.module @modA(in %in: i32, out out: i32) {
  %add = comb.add %in, %in : i32
  hw.output %add : i32
}
