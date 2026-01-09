// RUN: circt-opt --lower-to-bmc="top-module=noProp bound=10" %s | FileCheck %s

// CHECK: %[[ADDR:.+]] = llvm.mlir.addressof @[[RES_STR:resultString_0]] : !llvm.ptr
// CHECK: llvm.call @printf(%[[ADDR]]) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr) -> ()
// CHECK: return
// CHECK: llvm.mlir.global private constant @[[RES_STR]]("No properties provided - trivially no violations.\0A\00") {addr_space = 0 : i32}

hw.module @noProp(in %in0: i32, in %in1: i32, out out: i32) attributes {num_regs = 0 : i32, initial_values = []} {
  %0 = comb.add %in0, %in1 : i32
  hw.output %0 : i32
}