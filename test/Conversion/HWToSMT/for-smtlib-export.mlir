// RUN: circt-opt %s --convert-hw-to-smt='for-smtlib-export' --split-input-file | FileCheck %s

// CHECK-LABEL: smt.solver() : () -> ()
// CHECK-NEXT: [[DECL1:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: [[DECL2:%.+]] = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: [[EQ:%.+]] = smt.eq [[DECL1]], [[DECL2]] : !smt.bv<32>
// CHECK-NEXT: smt.assert [[EQ]]

hw.module @modA(in %in: i32, out out: i32) {
  hw.output %in : i32
}

// -----

// Check that output assertions are generated correctly with multiple outputs
// CHECK-LABEL: smt.solver() : () -> ()
// CHECK-NEXT: %[[ARG0:.*]] = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: %[[ARG1:.*]] = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: %[[DECL1:.*]] = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: %[[EQ1:.*]] = smt.eq %[[ARG0]], %[[DECL1]] : !smt.bv<32>
// CHECK-NEXT: smt.assert %[[EQ1]]
// CHECK-NEXT: %[[DECL2:.*]] = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: %[[EQ2:.*]] = smt.eq %[[ARG1]], %[[DECL2]] : !smt.bv<32>
// CHECK-NEXT: smt.assert %[[EQ2]]

hw.module @modC(in %in1: i32, in %in2: i32, out out1: i32, out out2: i32) {
  hw.output %in1, %in2 : i32, i32
}

