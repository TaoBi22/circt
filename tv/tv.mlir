module {
  llvm.func @printf(!llvm.ptr, ...)
  func.func @fsm10() {
    %0 = llvm.mlir.addressof @resultString_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @resultString_0 : !llvm.ptr
    %2 = smt.solver() : () -> i1 {
      %true = arith.constant true
      %false = arith.constant false
      %c20_i32 = arith.constant 20 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
      %c0_bv4 = smt.bv.constant #smt.bv<0> : !smt.bv<4>
      %4 = func.call @bmc_init() : () -> !smt.bv<1>
      %5 = smt.declare_fun : !smt.bv<1>
      %6:6 = scf.for %arg0 = %c0_i32 to %c20_i32 step %c1_i32 iter_args(%arg1 = %4, %arg2 = %5, %arg3 = %c0_bv4, %arg4 = %c0_bv16, %arg5 = %c0_bv32, %arg6 = %false) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<4>, !smt.bv<16>, !smt.bv<32>, i1)  : i32 {
        %8:3 = func.call @bmc_circuit(%arg1, %arg2, %arg3, %arg4, %arg5) : (!smt.bv<1>, !smt.bv<1>, !smt.bv<4>, !smt.bv<16>, !smt.bv<32>) -> (!smt.bv<4>, !smt.bv<16>, !smt.bv<32>)
        %9 = smt.check sat {
          smt.yield %true : i1
        } unknown {
          smt.yield %false : i1
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
        scf.yield %11, %12, %8#0, %8#1, %8#2, %10 : !smt.bv<1>, !smt.bv<1>, !smt.bv<4>, !smt.bv<16>, !smt.bv<32>, i1
      }
      %7 = arith.xori %6#5, %true : i1
      smt.yield %7 : i1
    }
    %3 = llvm.select %2, %1, %0 : i1, !llvm.ptr
    llvm.call @printf(%3) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr) -> ()
    return
  }
  llvm.mlir.global private constant @resultString_0("Bound reached with no violations!\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @resultString_1("Assertion can be violated!\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @satString("sat\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @unsatString("unsat\0A\00") {addr_space = 0 : i32}
  func.func @bmc_init() -> !smt.bv<1> {
    %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    return %c0_bv1 : !smt.bv<1>
  }
  func.func @bmc_loop(%arg0: !smt.bv<1>) -> !smt.bv<1> {
    %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %0 = smt.bv.xor %arg0, %c-1_bv1 : !smt.bv<1>
    return %0 : !smt.bv<1>
  }
  func.func @bmc_circuit(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<4>, %arg3: !smt.bv<16>, %arg4: !smt.bv<32>) -> (!smt.bv<4>, !smt.bv<16>, !smt.bv<32>) {
    %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
    %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
    %c-6_bv4 = smt.bv.constant #smt.bv<-6> : !smt.bv<4>
    %c-7_bv4 = smt.bv.constant #smt.bv<-7> : !smt.bv<4>
    %c-8_bv4 = smt.bv.constant #smt.bv<-8> : !smt.bv<4>
    %c7_bv4 = smt.bv.constant #smt.bv<7> : !smt.bv<4>
    %c6_bv4 = smt.bv.constant #smt.bv<6> : !smt.bv<4>
    %c5_bv4 = smt.bv.constant #smt.bv<5> : !smt.bv<4>
    %c4_bv4 = smt.bv.constant #smt.bv<4> : !smt.bv<4>
    %c3_bv4 = smt.bv.constant #smt.bv<3> : !smt.bv<4>
    %c2_bv4 = smt.bv.constant #smt.bv<2> : !smt.bv<4>
    %c1_bv4 = smt.bv.constant #smt.bv<1> : !smt.bv<4>
    %c0_bv4 = smt.bv.constant #smt.bv<0> : !smt.bv<4>
    %0 = smt.eq %arg2, %c0_bv4 : !smt.bv<4>
    %1 = smt.ite %0, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %2 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %3 = smt.eq %1, %c-1_bv1 : !smt.bv<1>
    %4 = smt.ite %3, %2, %arg3 : !smt.bv<16>
    %5 = smt.eq %1, %c-1_bv1 : !smt.bv<1>
    %6 = smt.ite %5, %c1_bv4, %arg2 : !smt.bv<4>
    %7 = smt.eq %arg2, %c1_bv4 : !smt.bv<4>
    %8 = smt.ite %7, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %9 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %10 = smt.eq %8, %c-1_bv1 : !smt.bv<1>
    %11 = smt.ite %10, %9, %4 : !smt.bv<16>
    %12 = smt.eq %8, %c-1_bv1 : !smt.bv<1>
    %13 = smt.ite %12, %c2_bv4, %6 : !smt.bv<4>
    %14 = smt.eq %arg2, %c2_bv4 : !smt.bv<4>
    %15 = smt.ite %14, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %16 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %17 = smt.eq %15, %c-1_bv1 : !smt.bv<1>
    %18 = smt.ite %17, %16, %11 : !smt.bv<16>
    %19 = smt.eq %15, %c-1_bv1 : !smt.bv<1>
    %20 = smt.ite %19, %c3_bv4, %13 : !smt.bv<4>
    %21 = smt.eq %arg2, %c3_bv4 : !smt.bv<4>
    %22 = smt.ite %21, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %23 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %24 = smt.eq %22, %c-1_bv1 : !smt.bv<1>
    %25 = smt.ite %24, %23, %18 : !smt.bv<16>
    %26 = smt.eq %22, %c-1_bv1 : !smt.bv<1>
    %27 = smt.ite %26, %c4_bv4, %20 : !smt.bv<4>
    %28 = smt.eq %arg2, %c4_bv4 : !smt.bv<4>
    %29 = smt.ite %28, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %30 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %31 = smt.eq %29, %c-1_bv1 : !smt.bv<1>
    %32 = smt.ite %31, %30, %25 : !smt.bv<16>
    %33 = smt.eq %29, %c-1_bv1 : !smt.bv<1>
    %34 = smt.ite %33, %c5_bv4, %27 : !smt.bv<4>
    %35 = smt.eq %arg2, %c5_bv4 : !smt.bv<4>
    %36 = smt.ite %35, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %37 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %38 = smt.eq %36, %c-1_bv1 : !smt.bv<1>
    %39 = smt.ite %38, %37, %32 : !smt.bv<16>
    %40 = smt.eq %36, %c-1_bv1 : !smt.bv<1>
    %41 = smt.ite %40, %c6_bv4, %34 : !smt.bv<4>
    %42 = smt.eq %arg2, %c6_bv4 : !smt.bv<4>
    %43 = smt.ite %42, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %44 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %45 = smt.eq %43, %c-1_bv1 : !smt.bv<1>
    %46 = smt.ite %45, %44, %39 : !smt.bv<16>
    %47 = smt.eq %43, %c-1_bv1 : !smt.bv<1>
    %48 = smt.ite %47, %c7_bv4, %41 : !smt.bv<4>
    %49 = smt.eq %arg2, %c7_bv4 : !smt.bv<4>
    %50 = smt.ite %49, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %51 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %52 = smt.eq %50, %c-1_bv1 : !smt.bv<1>
    %53 = smt.ite %52, %51, %46 : !smt.bv<16>
    %54 = smt.eq %50, %c-1_bv1 : !smt.bv<1>
    %55 = smt.ite %54, %c-8_bv4, %48 : !smt.bv<4>
    %56 = smt.eq %arg2, %c-8_bv4 : !smt.bv<4>
    %57 = smt.ite %56, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %58 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %59 = smt.eq %57, %c-1_bv1 : !smt.bv<1>
    %60 = smt.ite %59, %58, %53 : !smt.bv<16>
    %61 = smt.eq %57, %c-1_bv1 : !smt.bv<1>
    %62 = smt.ite %61, %c-7_bv4, %55 : !smt.bv<4>
    %63 = smt.eq %arg2, %c-7_bv4 : !smt.bv<4>
    %64 = smt.ite %63, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %65 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %66 = smt.eq %64, %c-1_bv1 : !smt.bv<1>
    %67 = smt.ite %66, %65, %60 : !smt.bv<16>
    %68 = smt.eq %64, %c-1_bv1 : !smt.bv<1>
    %69 = smt.ite %68, %c-6_bv4, %62 : !smt.bv<4>
    %70 = smt.eq %arg2, %c-6_bv4 : !smt.bv<4>
    %71 = smt.ite %70, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %72 = smt.eq %71, %c-1_bv1 : !smt.bv<1>
    %73 = smt.ite %72, %c-6_bv4, %69 : !smt.bv<4>
    %74 = smt.bv.add %arg4, %c1_bv32 : !smt.bv<32>

    // FSM

    %bF__0 = smt.declare_fun "F__0" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %bF__1 = smt.declare_fun "F__1" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %bF__2 = smt.declare_fun "F__2" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %bF__3 = smt.declare_fun "F__3" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %bF__4 = smt.declare_fun "F__4" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %bF__5 = smt.declare_fun "F__5" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %bF__6 = smt.declare_fun "F__6" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %bF__7 = smt.declare_fun "F__7" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %bF__8 = smt.declare_fun "F__8" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %bF__9 = smt.declare_fun "F__9" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %bF__10 = smt.declare_fun "F__10" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %b0 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %bc0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
      %bc0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %b11 = smt.eq %barg1, %bc0_bv32 : !smt.bv<32>
      %b12 = smt.apply_func %bF__0(%bc0_bv16, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %b13 = smt.implies %b11, %b12
      smt.yield %b13 : !smt.bool
    }
    smt.assert %b0
    %b1 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %b11 = smt.apply_func %bF__0(%barg0, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %bc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %bc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b12 = smt.eq %barg0, %bc1_bv16_0 : !smt.bv<16>
      %bc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b13 = smt.eq %bc1_bv16, %bc1_bv16_1 : !smt.bv<16>
      %b14 = smt.bv.add %barg0, %bc1_bv16 : !smt.bv<16>
      %bc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %b15 = smt.bv.add %barg1, %bc1_bv32 : !smt.bv<32>
      %b16 = smt.apply_func %bF__1(%b14, %b15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %btrue = smt.constant true
      %b17 = smt.and %b11, %btrue
      %b18 = smt.implies %b17, %b16
      smt.yield %b18 : !smt.bool
    }
    smt.assert %b1
    %b2 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %b11 = smt.apply_func %bF__1(%barg0, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %bc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %bc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b12 = smt.eq %barg0, %bc1_bv16_0 : !smt.bv<16>
      %bc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b13 = smt.eq %bc1_bv16, %bc1_bv16_1 : !smt.bv<16>
      %b14 = smt.bv.add %barg0, %bc1_bv16 : !smt.bv<16>
      %bc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %b15 = smt.bv.add %barg1, %bc1_bv32 : !smt.bv<32>
      %b16 = smt.apply_func %bF__2(%b14, %b15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %btrue = smt.constant true
      %b17 = smt.and %b11, %btrue
      %b18 = smt.implies %b17, %b16
      smt.yield %b18 : !smt.bool
    }
    smt.assert %b2
    %b3 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %b11 = smt.apply_func %bF__2(%barg0, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %bc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %bc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b12 = smt.eq %barg0, %bc1_bv16_0 : !smt.bv<16>
      %bc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b13 = smt.eq %bc1_bv16, %bc1_bv16_1 : !smt.bv<16>
      %b14 = smt.bv.add %barg0, %bc1_bv16 : !smt.bv<16>
      %bc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %b15 = smt.bv.add %barg1, %bc1_bv32 : !smt.bv<32>
      %b16 = smt.apply_func %bF__3(%b14, %b15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %btrue = smt.constant true
      %b17 = smt.and %b11, %btrue
      %b18 = smt.implies %b17, %b16
      smt.yield %b18 : !smt.bool
    }
    smt.assert %b3
    %b4 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %b11 = smt.apply_func %bF__3(%barg0, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %bc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %bc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b12 = smt.eq %barg0, %bc1_bv16_0 : !smt.bv<16>
      %bc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b13 = smt.eq %bc1_bv16, %bc1_bv16_1 : !smt.bv<16>
      %b14 = smt.bv.add %barg0, %bc1_bv16 : !smt.bv<16>
      %bc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %b15 = smt.bv.add %barg1, %bc1_bv32 : !smt.bv<32>
      %b16 = smt.apply_func %bF__4(%b14, %b15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %btrue = smt.constant true
      %b17 = smt.and %b11, %btrue
      %b18 = smt.implies %b17, %b16
      smt.yield %b18 : !smt.bool
    }
    smt.assert %b4
    %b5 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %b11 = smt.apply_func %bF__4(%barg0, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %bc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %bc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b12 = smt.eq %barg0, %bc1_bv16_0 : !smt.bv<16>
      %bc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b13 = smt.eq %bc1_bv16, %bc1_bv16_1 : !smt.bv<16>
      %b14 = smt.bv.add %barg0, %bc1_bv16 : !smt.bv<16>
      %bc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %b15 = smt.bv.add %barg1, %bc1_bv32 : !smt.bv<32>
      %b16 = smt.apply_func %bF__5(%b14, %b15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %btrue = smt.constant true
      %b17 = smt.and %b11, %btrue
      %b18 = smt.implies %b17, %b16
      smt.yield %b18 : !smt.bool
    }
    smt.assert %b5
    %b6 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %b11 = smt.apply_func %bF__5(%barg0, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %bc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %bc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b12 = smt.eq %barg0, %bc1_bv16_0 : !smt.bv<16>
      %bc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b13 = smt.eq %bc1_bv16, %bc1_bv16_1 : !smt.bv<16>
      %b14 = smt.bv.add %barg0, %bc1_bv16 : !smt.bv<16>
      %bc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %b15 = smt.bv.add %barg1, %bc1_bv32 : !smt.bv<32>
      %b16 = smt.apply_func %bF__6(%b14, %b15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %btrue = smt.constant true
      %b17 = smt.and %b11, %btrue
      %b18 = smt.implies %b17, %b16
      smt.yield %b18 : !smt.bool
    }
    smt.assert %b6
    %b7 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %b11 = smt.apply_func %bF__6(%barg0, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %bc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %bc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b12 = smt.eq %barg0, %bc1_bv16_0 : !smt.bv<16>
      %bc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b13 = smt.eq %bc1_bv16, %bc1_bv16_1 : !smt.bv<16>
      %b14 = smt.bv.add %barg0, %bc1_bv16 : !smt.bv<16>
      %bc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %b15 = smt.bv.add %barg1, %bc1_bv32 : !smt.bv<32>
      %b16 = smt.apply_func %bF__7(%b14, %b15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %btrue = smt.constant true
      %b17 = smt.and %b11, %btrue
      %b18 = smt.implies %b17, %b16
      smt.yield %b18 : !smt.bool
    }
    smt.assert %b7
    %b8 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %b11 = smt.apply_func %bF__7(%barg0, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %bc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %bc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b12 = smt.eq %barg0, %bc1_bv16_0 : !smt.bv<16>
      %bc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b13 = smt.eq %bc1_bv16, %bc1_bv16_1 : !smt.bv<16>
      %b14 = smt.bv.add %barg0, %bc1_bv16 : !smt.bv<16>
      %bc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %b15 = smt.bv.add %barg1, %bc1_bv32 : !smt.bv<32>
      %b16 = smt.apply_func %bF__8(%b14, %b15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %btrue = smt.constant true
      %b17 = smt.and %b11, %btrue
      %b18 = smt.implies %b17, %b16
      smt.yield %b18 : !smt.bool
    }
    smt.assert %b8
    %b9 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %b11 = smt.apply_func %bF__8(%barg0, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %bc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %bc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b12 = smt.eq %barg0, %bc1_bv16_0 : !smt.bv<16>
      %bc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b13 = smt.eq %bc1_bv16, %bc1_bv16_1 : !smt.bv<16>
      %b14 = smt.bv.add %barg0, %bc1_bv16 : !smt.bv<16>
      %bc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %b15 = smt.bv.add %barg1, %bc1_bv32 : !smt.bv<32>
      %b16 = smt.apply_func %bF__9(%b14, %b15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %btrue = smt.constant true
      %b17 = smt.and %b11, %btrue
      %b18 = smt.implies %b17, %b16
      smt.yield %b18 : !smt.bool
    }
    smt.assert %b9
    %b10 = smt.forall {
    ^bb0(%barg0: !smt.bv<16>, %barg1: !smt.bv<32>):
      %b11 = smt.apply_func %bF__9(%barg0, %barg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %bc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %bc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b12 = smt.eq %barg0, %bc1_bv16_0 : !smt.bv<16>
      %bc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %b13 = smt.eq %bc1_bv16, %bc1_bv16_1 : !smt.bv<16>
      %b14 = smt.bv.add %barg0, %bc1_bv16 : !smt.bv<16>
      %bc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %b15 = smt.bv.add %barg1, %bc1_bv32 : !smt.bv<32>
      %b16 = smt.apply_func %bF__10(%b14, %b15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %btrue = smt.constant true
      %b17 = smt.and %b11, %btrue
      %b18 = smt.implies %b17, %b16
      smt.yield %b18 : !smt.bool
    }
    smt.assert %b10

    %tvclause = smt.forall {
      ^bb0(%var: !smt.bv<16>, %sTime: !smt.bv<32>):
        %apply = smt.apply_func %bF__0(%var, %sTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
        %teq = smt.eq %arg4, %sTime : !smt.bv<32>
        %vDiff = smt.distinct %var, %arg3 : !smt.bv<16>
        %and = smt.and %apply, %teq, %vDiff
        %myfalse = smt.constant false
        %impl = smt.implies %and, %myfalse
        smt.yield %impl : !smt.bool
    }
    smt.assert %tvclause

    return %73, %67, %74 : !smt.bv<4>, !smt.bv<16>, !smt.bv<32>
  }
}

