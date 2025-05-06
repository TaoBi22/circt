// RUN: circt-opt %s --perform-essent-merges --verify-diagnostics | FileCheck %s

// CHECK-LABEL: arc.define
arc.define @EmptyArc1(%a: i8) -> i8 {
  arc.output %a : i8
}

arc.define @EmptyArc2(%a : i7, %c : i7, %b : i8) -> (i7) {
  %1 = comb.xor %a, %c : i7
  arc.output %1 : i7
}

hw.module @callInModuleTest() {
  %0 = hw.constant 0 : i8
  %1 = hw.constant 0 : i7
  arc.call @EmptyArc1(%0) : (i8) -> (i8)
  arc.call @EmptyArc2(%1, %1, %0) : (i7, i7, i8) -> (i7)
}
