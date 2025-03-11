module {
  llvm.func @printf(!llvm.ptr, ...)
  func.func @fsm10() {
    %0 = smt.solver() : () -> i1 {
      %4 = func.call @bmc_init() : () -> !smt.bv<1>
      smt.push 1
      %5 = smt.declare_fun : !smt.bv<1>
      %c0_bv4 = smt.bv.constant #smt.bv<0> : !smt.bv<4>
      %c0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c40_i32 = arith.constant 40 : i32
      %false = arith.constant false
      %true = arith.constant true
      %6:6 = scf.for %arg0 = %c0_i32 to %c40_i32 step %c1_i32 iter_args(%arg1 = %4, %arg2 = %5, %arg3 = %c0_bv4, %arg4 = %c0_bv16, %arg5 = %c0_bv32, %arg6 = %false) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<4>, !smt.bv<16>, !smt.bv<32>, i1)  : i32 {
        smt.pop 1
        smt.push 1
        %8:3 = func.call @bmc_circuit(%arg1, %arg2, %arg3, %arg4, %arg5) : (!smt.bv<1>, !smt.bv<1>, !smt.bv<4>, !smt.bv<16>, !smt.bv<32>) -> (!smt.bv<4>, !smt.bv<16>, !smt.bv<32>)
        %9 = smt.check sat {
          smt.yield %true : i1
        } unknown {
          smt.yield %true : i1
        } unsat {
          smt.yield %false : i1
        } -> i1
        %ss = llvm.mlir.addressof @satString : !llvm.ptr
        %us = llvm.mlir.addressof @unsatString : !llvm.ptr
        %string = llvm.select %9, %ss, %us : i1, !llvm.ptr
        llvm.call @printf(%string) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr) -> ()
        %10 = arith.andi %9, %arg6 : i1
        %11 = func.call @bmc_loop(%arg1) : (!smt.bv<1>) -> !smt.bv<1>
        %12 = smt.declare_fun : !smt.bv<1>
        %13 = smt.bv.not %arg1 : !smt.bv<1>
        %14 = smt.bv.and %13, %11 : !smt.bv<1>
        %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
        %15 = smt.eq %14, %c-1_bv1 : !smt.bv<1>
        %16 = smt.ite %15, %8#0, %arg3 : !smt.bv<4>
        %17 = smt.ite %15, %8#1, %arg4 : !smt.bv<16>
        %18 = smt.ite %15, %8#2, %arg5 : !smt.bv<32>
        scf.yield %11, %12, %16, %17, %18, %10 : !smt.bv<1>, !smt.bv<1>, !smt.bv<4>, !smt.bv<16>, !smt.bv<32>, i1
      }
      %7 = arith.xori %6#5, %true : i1
      smt.yield %7 : i1
    }
    %1 = llvm.mlir.addressof @resultString_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @resultString_1 : !llvm.ptr
    %3 = llvm.select %0, %1, %2 : i1, !llvm.ptr
    llvm.call @printf(%3) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr) -> ()
    return
  }
  llvm.mlir.global private constant @resultString_0("TV didn't hold\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @resultString_1("TV held\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @satString("sat\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @unsatString("unsat\0A\00") {addr_space = 0 : i32}
  func.func @bmc_init() -> !smt.bv<1> {
    %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %0 = builtin.unrealized_conversion_cast %c0_bv1 : !smt.bv<1> to !seq.clock
    %1 = builtin.unrealized_conversion_cast %0 : !seq.clock to !smt.bv<1>
    return %1 : !smt.bv<1>
  }
  func.func @bmc_loop(%arg0: !smt.bv<1>) -> !smt.bv<1> {
    %0 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to !seq.clock
    %1 = builtin.unrealized_conversion_cast %0 : !seq.clock to !smt.bv<1>
    %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %2 = smt.bv.xor %1, %c-1_bv1 : !smt.bv<1>
    %3 = builtin.unrealized_conversion_cast %2 : !smt.bv<1> to i1
    %4 = builtin.unrealized_conversion_cast %3 : i1 to !smt.bv<1>
    %5 = builtin.unrealized_conversion_cast %4 : !smt.bv<1> to !seq.clock
    %6 = builtin.unrealized_conversion_cast %5 : !seq.clock to !smt.bv<1>
    return %6 : !smt.bv<1>
  }
  func.func @bmc_circuit(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<4>, %arg3: !smt.bv<16>, %arg4: !smt.bv<32>) -> (!smt.bv<4>, !smt.bv<16>, !smt.bv<32>) {
    %0 = builtin.unrealized_conversion_cast %arg4 : !smt.bv<32> to i32
    %1 = builtin.unrealized_conversion_cast %arg3 : !smt.bv<16> to i16
    %2 = builtin.unrealized_conversion_cast %arg2 : !smt.bv<4> to i4
    %3 = builtin.unrealized_conversion_cast %0 : i32 to !smt.bv<32>
    %4 = builtin.unrealized_conversion_cast %1 : i16 to !smt.bv<16>
    %5 = builtin.unrealized_conversion_cast %2 : i4 to !smt.bv<4>
    %c0_bv4 = smt.bv.constant #smt.bv<0> : !smt.bv<4>
    %c1_bv4 = smt.bv.constant #smt.bv<1> : !smt.bv<4>
    %c2_bv4 = smt.bv.constant #smt.bv<2> : !smt.bv<4>
    %c3_bv4 = smt.bv.constant #smt.bv<3> : !smt.bv<4>
    %c4_bv4 = smt.bv.constant #smt.bv<4> : !smt.bv<4>
    %c5_bv4 = smt.bv.constant #smt.bv<5> : !smt.bv<4>
    %c6_bv4 = smt.bv.constant #smt.bv<6> : !smt.bv<4>
    %c7_bv4 = smt.bv.constant #smt.bv<7> : !smt.bv<4>
    %c-8_bv4 = smt.bv.constant #smt.bv<-8> : !smt.bv<4>
    %c-7_bv4 = smt.bv.constant #smt.bv<-7> : !smt.bv<4>
    %c-6_bv4 = smt.bv.constant #smt.bv<-6> : !smt.bv<4>
    %c0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
    %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
    %6 = smt.eq %5, %c0_bv4 : !smt.bv<4>
    %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %7 = smt.ite %6, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %8 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %9 = smt.eq %7, %c-1_bv1_0 : !smt.bv<1>
    %10 = smt.ite %9, %8, %4 : !smt.bv<16>
    %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %11 = smt.eq %7, %c-1_bv1_1 : !smt.bv<1>
    %12 = smt.ite %11, %c1_bv4, %5 : !smt.bv<4>
    %13 = smt.eq %5, %c1_bv4 : !smt.bv<4>
    %c0_bv1_2 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_3 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %14 = smt.ite %13, %c-1_bv1_3, %c0_bv1_2 : !smt.bv<1>
    %15 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %16 = smt.eq %14, %c-1_bv1_4 : !smt.bv<1>
    %17 = smt.ite %16, %15, %10 : !smt.bv<16>
    %c-1_bv1_5 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %18 = smt.eq %14, %c-1_bv1_5 : !smt.bv<1>
    %19 = smt.ite %18, %c2_bv4, %12 : !smt.bv<4>
    %20 = smt.eq %5, %c2_bv4 : !smt.bv<4>
    %c0_bv1_6 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_7 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %21 = smt.ite %20, %c-1_bv1_7, %c0_bv1_6 : !smt.bv<1>
    %22 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_8 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %23 = smt.eq %21, %c-1_bv1_8 : !smt.bv<1>
    %24 = smt.ite %23, %22, %17 : !smt.bv<16>
    %c-1_bv1_9 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %25 = smt.eq %21, %c-1_bv1_9 : !smt.bv<1>
    %26 = smt.ite %25, %c3_bv4, %19 : !smt.bv<4>
    %27 = smt.eq %5, %c3_bv4 : !smt.bv<4>
    %c0_bv1_10 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_11 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %28 = smt.ite %27, %c-1_bv1_11, %c0_bv1_10 : !smt.bv<1>
    %29 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_12 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %30 = smt.eq %28, %c-1_bv1_12 : !smt.bv<1>
    %31 = smt.ite %30, %29, %24 : !smt.bv<16>
    %c-1_bv1_13 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %32 = smt.eq %28, %c-1_bv1_13 : !smt.bv<1>
    %33 = smt.ite %32, %c4_bv4, %26 : !smt.bv<4>
    %34 = smt.eq %5, %c4_bv4 : !smt.bv<4>
    %c0_bv1_14 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_15 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %35 = smt.ite %34, %c-1_bv1_15, %c0_bv1_14 : !smt.bv<1>
    %36 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_16 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %37 = smt.eq %35, %c-1_bv1_16 : !smt.bv<1>
    %38 = smt.ite %37, %36, %31 : !smt.bv<16>
    %c-1_bv1_17 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %39 = smt.eq %35, %c-1_bv1_17 : !smt.bv<1>
    %40 = smt.ite %39, %c5_bv4, %33 : !smt.bv<4>
    %41 = smt.eq %5, %c5_bv4 : !smt.bv<4>
    %c0_bv1_18 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_19 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %42 = smt.ite %41, %c-1_bv1_19, %c0_bv1_18 : !smt.bv<1>
    %43 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_20 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %44 = smt.eq %42, %c-1_bv1_20 : !smt.bv<1>
    %45 = smt.ite %44, %43, %38 : !smt.bv<16>
    %c-1_bv1_21 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %46 = smt.eq %42, %c-1_bv1_21 : !smt.bv<1>
    %47 = smt.ite %46, %c6_bv4, %40 : !smt.bv<4>
    %48 = smt.eq %5, %c6_bv4 : !smt.bv<4>
    %c0_bv1_22 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_23 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %49 = smt.ite %48, %c-1_bv1_23, %c0_bv1_22 : !smt.bv<1>
    %50 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_24 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %51 = smt.eq %49, %c-1_bv1_24 : !smt.bv<1>
    %52 = smt.ite %51, %50, %45 : !smt.bv<16>
    %c-1_bv1_25 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %53 = smt.eq %49, %c-1_bv1_25 : !smt.bv<1>
    %54 = smt.ite %53, %c7_bv4, %47 : !smt.bv<4>
    %55 = smt.eq %5, %c7_bv4 : !smt.bv<4>
    %c0_bv1_26 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_27 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %56 = smt.ite %55, %c-1_bv1_27, %c0_bv1_26 : !smt.bv<1>
    %57 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_28 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %58 = smt.eq %56, %c-1_bv1_28 : !smt.bv<1>
    %59 = smt.ite %58, %57, %52 : !smt.bv<16>
    %c-1_bv1_29 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %60 = smt.eq %56, %c-1_bv1_29 : !smt.bv<1>
    %61 = smt.ite %60, %c-8_bv4, %54 : !smt.bv<4>
    %62 = smt.eq %5, %c-8_bv4 : !smt.bv<4>
    %c0_bv1_30 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_31 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %63 = smt.ite %62, %c-1_bv1_31, %c0_bv1_30 : !smt.bv<1>
    %64 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_32 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %65 = smt.eq %63, %c-1_bv1_32 : !smt.bv<1>
    %66 = smt.ite %65, %64, %59 : !smt.bv<16>
    %c-1_bv1_33 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %67 = smt.eq %63, %c-1_bv1_33 : !smt.bv<1>
    %68 = smt.ite %67, %c-7_bv4, %61 : !smt.bv<4>
    %69 = smt.eq %5, %c-7_bv4 : !smt.bv<4>
    %c0_bv1_34 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_35 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %70 = smt.ite %69, %c-1_bv1_35, %c0_bv1_34 : !smt.bv<1>
    %71 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_36 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %72 = smt.eq %70, %c-1_bv1_36 : !smt.bv<1>
    %73 = smt.ite %72, %71, %66 : !smt.bv<16>
    %74 = builtin.unrealized_conversion_cast %73 : !smt.bv<16> to i16
    %c-1_bv1_37 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %75 = smt.eq %70, %c-1_bv1_37 : !smt.bv<1>
    %76 = smt.ite %75, %c-6_bv4, %68 : !smt.bv<4>
    %77 = smt.eq %5, %c-6_bv4 : !smt.bv<4>
    %c0_bv1_38 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_39 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %78 = smt.ite %77, %c-1_bv1_39, %c0_bv1_38 : !smt.bv<1>
    %c-1_bv1_40 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %79 = smt.eq %78, %c-1_bv1_40 : !smt.bv<1>
    %80 = smt.ite %79, %c-6_bv4, %76 : !smt.bv<4>
    %81 = builtin.unrealized_conversion_cast %80 : !smt.bv<4> to i4
    %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
    %82 = smt.bv.add %3, %c1_bv32 : !smt.bv<32>
    %83 = builtin.unrealized_conversion_cast %82 : !smt.bv<32> to i32
    %c-1_bv1_41 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %c-1_bv1_42 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %86 = builtin.unrealized_conversion_cast %81 : i4 to !smt.bv<4>
    %87 = builtin.unrealized_conversion_cast %74 : i16 to !smt.bv<16>
    %88 = builtin.unrealized_conversion_cast %83 : i32 to !smt.bv<32>

    // FSM

    %bF__0 = smt.declare_fun "F__0" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %bF__1 = smt.declare_fun "F__1" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %bF__2 = smt.declare_fun "F__2" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %bF__3 = smt.declare_fun "F__3" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %bF__4 = smt.declare_fun "F__4" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %bF__5 = smt.declare_fun "F__5" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %bF__6 = smt.declare_fun "F__6" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %bF__7 = smt.declare_fun "F__7" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %bF__8 = smt.declare_fun "F__8" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %bF__9 = smt.declare_fun "F__9" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %bF__10 = smt.declare_fun "F__10" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %b0 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %bc0 = smt.int.constant 0
      %bc0_0 = smt.int.constant 0
      %b11 = smt.eq %barg1, %bc0_0 : !smt.int
      %b12 = smt.apply_func %bF__0(%bc0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %b13 = smt.implies %b11, %b12
      smt.yield %b13 : !smt.bool
    }
    smt.assert %b0
    %b1 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %b11 = smt.apply_func %bF__0(%barg0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %bc1 = smt.int.constant 5
      %b12 = smt.int.add %barg0, %bc1
      %bc65536 = smt.int.constant 65536
      %b13 = smt.int.mod %b12, %bc65536
      %bc1_0 = smt.int.constant 1
      %b14 = smt.int.add %barg1, %bc1_0
      %b15 = smt.apply_func %bF__1(%b13, %b14) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %btrue = smt.constant true
      %b16 = smt.and %b11, %btrue
      %b17 = smt.implies %b16, %b15
      smt.yield %b17 : !smt.bool
    }
    smt.assert %b1
    %b2 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %b11 = smt.apply_func %bF__1(%barg0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %bc1 = smt.int.constant 1
      %b12 = smt.int.add %barg0, %bc1
      %bc65536 = smt.int.constant 65536
      %b13 = smt.int.mod %b12, %bc65536
      %bc1_0 = smt.int.constant 1
      %b14 = smt.int.add %barg1, %bc1_0
      %b15 = smt.apply_func %bF__2(%b13, %b14) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %btrue = smt.constant true
      %b16 = smt.and %b11, %btrue
      %b17 = smt.implies %b16, %b15
      smt.yield %b17 : !smt.bool
    }
    smt.assert %b2
    %b3 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %b11 = smt.apply_func %bF__2(%barg0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %bc1 = smt.int.constant 1
      %b12 = smt.int.add %barg0, %bc1
      %bc65536 = smt.int.constant 65536
      %b13 = smt.int.mod %b12, %bc65536
      %bc1_0 = smt.int.constant 1
      %b14 = smt.int.add %barg1, %bc1_0
      %b15 = smt.apply_func %bF__3(%b13, %b14) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %btrue = smt.constant true
      %b16 = smt.and %b11, %btrue
      %b17 = smt.implies %b16, %b15
      smt.yield %b17 : !smt.bool
    }
    smt.assert %b3
    %b4 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %b11 = smt.apply_func %bF__3(%barg0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %bc1 = smt.int.constant 1
      %b12 = smt.int.add %barg0, %bc1
      %bc65536 = smt.int.constant 65536
      %b13 = smt.int.mod %b12, %bc65536
      %bc1_0 = smt.int.constant 1
      %b14 = smt.int.add %barg1, %bc1_0
      %b15 = smt.apply_func %bF__4(%b13, %b14) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %btrue = smt.constant true
      %b16 = smt.and %b11, %btrue
      %b17 = smt.implies %b16, %b15
      smt.yield %b17 : !smt.bool
    }
    smt.assert %b4
    %b5 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %b11 = smt.apply_func %bF__4(%barg0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %bc1 = smt.int.constant 1
      %b12 = smt.int.add %barg0, %bc1
      %bc65536 = smt.int.constant 65536
      %b13 = smt.int.mod %b12, %bc65536
      %bc1_0 = smt.int.constant 1
      %b14 = smt.int.add %barg1, %bc1_0
      %b15 = smt.apply_func %bF__5(%b13, %b14) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %btrue = smt.constant true
      %b16 = smt.and %b11, %btrue
      %b17 = smt.implies %b16, %b15
      smt.yield %b17 : !smt.bool
    }
    smt.assert %b5
    %b6 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %b11 = smt.apply_func %bF__5(%barg0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %bc1 = smt.int.constant 1
      %b12 = smt.int.add %barg0, %bc1
      %bc65536 = smt.int.constant 65536
      %b13 = smt.int.mod %b12, %bc65536
      %bc1_0 = smt.int.constant 1
      %b14 = smt.int.add %barg1, %bc1_0
      %b15 = smt.apply_func %bF__6(%b13, %b14) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %btrue = smt.constant true
      %b16 = smt.and %b11, %btrue
      %b17 = smt.implies %b16, %b15
      smt.yield %b17 : !smt.bool
    }
    smt.assert %b6
    %b7 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %b11 = smt.apply_func %bF__6(%barg0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %bc1 = smt.int.constant 1
      %b12 = smt.int.add %barg0, %bc1
      %bc65536 = smt.int.constant 65536
      %b13 = smt.int.mod %b12, %bc65536
      %bc1_0 = smt.int.constant 1
      %b14 = smt.int.add %barg1, %bc1_0
      %b15 = smt.apply_func %bF__7(%b13, %b14) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %btrue = smt.constant true
      %b16 = smt.and %b11, %btrue
      %b17 = smt.implies %b16, %b15
      smt.yield %b17 : !smt.bool
    }
    smt.assert %b7
    %b8 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %b11 = smt.apply_func %bF__7(%barg0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %bc1 = smt.int.constant 1
      %b12 = smt.int.add %barg0, %bc1
      %bc65536 = smt.int.constant 65536
      %b13 = smt.int.mod %b12, %bc65536
      %bc1_0 = smt.int.constant 1
      %b14 = smt.int.add %barg1, %bc1_0
      %b15 = smt.apply_func %bF__8(%b13, %b14) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %btrue = smt.constant true
      %b16 = smt.and %b11, %btrue
      %b17 = smt.implies %b16, %b15
      smt.yield %b17 : !smt.bool
    }
    smt.assert %b8
    %b9 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %b11 = smt.apply_func %bF__8(%barg0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %bc1 = smt.int.constant 1
      %b12 = smt.int.add %barg0, %bc1
      %bc65536 = smt.int.constant 65536
      %b13 = smt.int.mod %b12, %bc65536
      %bc1_0 = smt.int.constant 1
      %b14 = smt.int.add %barg1, %bc1_0
      %b15 = smt.apply_func %bF__9(%b13, %b14) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %btrue = smt.constant true
      %b16 = smt.and %b11, %btrue
      %b17 = smt.implies %b16, %b15
      smt.yield %b17 : !smt.bool
    }
    smt.assert %b9
    %b10 = smt.forall {
    ^bb0(%barg0: !smt.int, %barg1: !smt.int):
      %b11 = smt.apply_func %bF__9(%barg0, %barg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %bc1 = smt.int.constant 1
      %b12 = smt.int.add %barg0, %bc1
      %bc65536 = smt.int.constant 65536
      %b13 = smt.int.mod %b12, %bc65536
      %bc1_0 = smt.int.constant 1
      %b14 = smt.int.add %barg1, %bc1_0
      %b15 = smt.apply_func %bF__10(%b13, %b14) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %btrue = smt.constant true
      %b16 = smt.and %b11, %btrue
      %b17 = smt.implies %b16, %b15
      smt.yield %b17 : !smt.bool
    }
    smt.assert %b10

    %tvclause_0 = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
      %intVar = smt.bv2int %var : !smt.bv<16>
      %intsTime = smt.bv2int %sTime : !smt.bv<32>
      %apply = smt.apply_func %bF__0(%intVar, %intsTime) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %teq = smt.eq %arg4, %sTime : !smt.bv<32>
      %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
      %and = smt.and %apply, %teq, %vDiff
      %myfalse = smt.constant false
      %impl = smt.implies %and, %myfalse
      smt.yield %impl : !smt.bool
    }

    smt.assert %tvclause_0

    %tvclause_1 = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
      %intVar = smt.bv2int %var : !smt.bv<16>
      %intsTime = smt.bv2int %sTime : !smt.bv<32>
      %apply = smt.apply_func %bF__1(%intVar, %intsTime) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %teq = smt.eq %arg4, %sTime : !smt.bv<32>
      %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
      %and = smt.and %apply, %teq, %vDiff
      %myfalse = smt.constant false
      %impl = smt.implies %and, %myfalse
      smt.yield %impl : !smt.bool
    }

    smt.assert %tvclause_1

    %tvclause_2 = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
      %intVar = smt.bv2int %var : !smt.bv<16>
      %intsTime = smt.bv2int %sTime : !smt.bv<32>
      %apply = smt.apply_func %bF__2(%intVar, %intsTime) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %teq = smt.eq %arg4, %sTime : !smt.bv<32>
      %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
      %and = smt.and %apply, %teq, %vDiff
      %myfalse = smt.constant false
      %impl = smt.implies %and, %myfalse
      smt.yield %impl : !smt.bool
    }

    smt.assert %tvclause_2

    %tvclause_3 = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
      %intVar = smt.bv2int %var : !smt.bv<16>
      %intsTime = smt.bv2int %sTime : !smt.bv<32>
      %apply = smt.apply_func %bF__3(%intVar, %intsTime) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %teq = smt.eq %arg4, %sTime : !smt.bv<32>
      %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
      %and = smt.and %apply, %teq, %vDiff
      %myfalse = smt.constant false
      %impl = smt.implies %and, %myfalse
      smt.yield %impl : !smt.bool
    }

    smt.assert %tvclause_3

    %tvclause_4 = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
      %intVar = smt.bv2int %var : !smt.bv<16>
      %intsTime = smt.bv2int %sTime : !smt.bv<32>
      %apply = smt.apply_func %bF__4(%intVar, %intsTime) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %teq = smt.eq %arg4, %sTime : !smt.bv<32>
      %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
      %and = smt.and %apply, %teq, %vDiff
      %myfalse = smt.constant false
      %impl = smt.implies %and, %myfalse
      smt.yield %impl : !smt.bool
    }

    smt.assert %tvclause_4

    %tvclause_5 = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
      %intVar = smt.bv2int %var : !smt.bv<16>
      %intsTime = smt.bv2int %sTime : !smt.bv<32>
      %apply = smt.apply_func %bF__5(%intVar, %intsTime) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %teq = smt.eq %arg4, %sTime : !smt.bv<32>
      %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
      %and = smt.and %apply, %teq, %vDiff
      %myfalse = smt.constant false
      %impl = smt.implies %and, %myfalse
      smt.yield %impl : !smt.bool
    }

    smt.assert %tvclause_5

    %tvclause_6 = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
      %intVar = smt.bv2int %var : !smt.bv<16>
      %intsTime = smt.bv2int %sTime : !smt.bv<32>
      %apply = smt.apply_func %bF__6(%intVar, %intsTime) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %teq = smt.eq %arg4, %sTime : !smt.bv<32>
      %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
      %and = smt.and %apply, %teq, %vDiff
      %myfalse = smt.constant false
      %impl = smt.implies %and, %myfalse
      smt.yield %impl : !smt.bool
    }

    smt.assert %tvclause_6

    %tvclause_7 = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
      %intVar = smt.bv2int %var : !smt.bv<16>
      %intsTime = smt.bv2int %sTime : !smt.bv<32>
      %apply = smt.apply_func %bF__7(%intVar, %intsTime) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %teq = smt.eq %arg4, %sTime : !smt.bv<32>
      %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
      %and = smt.and %apply, %teq, %vDiff
      %myfalse = smt.constant false
      %impl = smt.implies %and, %myfalse
      smt.yield %impl : !smt.bool
    }

    smt.assert %tvclause_7

    %tvclause_8 = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
      %intVar = smt.bv2int %var : !smt.bv<16>
      %intsTime = smt.bv2int %sTime : !smt.bv<32>
      %apply = smt.apply_func %bF__8(%intVar, %intsTime) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %teq = smt.eq %arg4, %sTime : !smt.bv<32>
      %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
      %and = smt.and %apply, %teq, %vDiff
      %myfalse = smt.constant false
      %impl = smt.implies %and, %myfalse
      smt.yield %impl : !smt.bool
    }

    smt.assert %tvclause_8

    %tvclause_9 = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
      %intVar = smt.bv2int %var : !smt.bv<16>
      %intsTime = smt.bv2int %sTime : !smt.bv<32>
      %apply = smt.apply_func %bF__9(%intVar, %intsTime) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %teq = smt.eq %arg4, %sTime : !smt.bv<32>
      %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
      %and = smt.and %apply, %teq, %vDiff
      %myfalse = smt.constant false
      %impl = smt.implies %and, %myfalse
      smt.yield %impl : !smt.bool
    }

    smt.assert %tvclause_9

    return %86, %87, %88 : !smt.bv<4>, !smt.bv<16>, !smt.bv<32>
  }
}

