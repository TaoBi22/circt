// RUN: circt-opt %s --arc-perform-essent-merges --verify-diagnostics | FileCheck %s

arc.define @EmptyArc1() {
  arc.output
}

arc.define @EmptyArc2() {
  arc.output
}

hw.module @callInModuleTest() {
  arc.call @EmptyArc1() : () -> ()
  arc.call @EmptyArc2() : () -> ()
}
