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
        %10 = arith.ori %9, %arg6 : i1
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
  llvm.mlir.global private constant @resultString_0("Bound reached with no violations!\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @resultString_1("Assertion can be violated!\0A\00") {addr_space = 0 : i32}
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
    %6 = seq.initial() {
      %c0_bv4_43 = smt.bv.constant #smt.bv<0> : !smt.bv<4>
      %92 = builtin.unrealized_conversion_cast %c0_bv4_43 : !smt.bv<4> to i4
      seq.yield %92 : i4
    } : () -> !seq.immutable<i4>
    %c0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
    %7 = seq.initial() {
      %c0_bv16_43 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
      %92 = builtin.unrealized_conversion_cast %c0_bv16_43 : !smt.bv<16> to i16
      seq.yield %92 : i16
    } : () -> !seq.immutable<i16>
    %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
    %8 = smt.eq %5, %c0_bv4 : !smt.bv<4>
    %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %9 = smt.ite %8, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %10 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %11 = smt.eq %9, %c-1_bv1_0 : !smt.bv<1>
    %12 = smt.ite %11, %10, %4 : !smt.bv<16>
    %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %13 = smt.eq %9, %c-1_bv1_1 : !smt.bv<1>
    %14 = smt.ite %13, %c1_bv4, %5 : !smt.bv<4>
    %15 = smt.eq %5, %c1_bv4 : !smt.bv<4>
    %c0_bv1_2 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_3 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %16 = smt.ite %15, %c-1_bv1_3, %c0_bv1_2 : !smt.bv<1>
    %17 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %18 = smt.eq %16, %c-1_bv1_4 : !smt.bv<1>
    %19 = smt.ite %18, %17, %12 : !smt.bv<16>
    %c-1_bv1_5 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %20 = smt.eq %16, %c-1_bv1_5 : !smt.bv<1>
    %21 = smt.ite %20, %c2_bv4, %14 : !smt.bv<4>
    %22 = smt.eq %5, %c2_bv4 : !smt.bv<4>
    %c0_bv1_6 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_7 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %23 = smt.ite %22, %c-1_bv1_7, %c0_bv1_6 : !smt.bv<1>
    %24 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_8 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %25 = smt.eq %23, %c-1_bv1_8 : !smt.bv<1>
    %26 = smt.ite %25, %24, %19 : !smt.bv<16>
    %c-1_bv1_9 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %27 = smt.eq %23, %c-1_bv1_9 : !smt.bv<1>
    %28 = smt.ite %27, %c3_bv4, %21 : !smt.bv<4>
    %29 = smt.eq %5, %c3_bv4 : !smt.bv<4>
    %c0_bv1_10 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_11 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %30 = smt.ite %29, %c-1_bv1_11, %c0_bv1_10 : !smt.bv<1>
    %31 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_12 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %32 = smt.eq %30, %c-1_bv1_12 : !smt.bv<1>
    %33 = smt.ite %32, %31, %26 : !smt.bv<16>
    %c-1_bv1_13 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %34 = smt.eq %30, %c-1_bv1_13 : !smt.bv<1>
    %35 = smt.ite %34, %c4_bv4, %28 : !smt.bv<4>
    %36 = smt.eq %5, %c4_bv4 : !smt.bv<4>
    %c0_bv1_14 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_15 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %37 = smt.ite %36, %c-1_bv1_15, %c0_bv1_14 : !smt.bv<1>
    %38 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_16 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %39 = smt.eq %37, %c-1_bv1_16 : !smt.bv<1>
    %40 = smt.ite %39, %38, %33 : !smt.bv<16>
    %c-1_bv1_17 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %41 = smt.eq %37, %c-1_bv1_17 : !smt.bv<1>
    %42 = smt.ite %41, %c5_bv4, %35 : !smt.bv<4>
    %43 = smt.eq %5, %c5_bv4 : !smt.bv<4>
    %c0_bv1_18 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_19 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %44 = smt.ite %43, %c-1_bv1_19, %c0_bv1_18 : !smt.bv<1>
    %45 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_20 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %46 = smt.eq %44, %c-1_bv1_20 : !smt.bv<1>
    %47 = smt.ite %46, %45, %40 : !smt.bv<16>
    %c-1_bv1_21 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %48 = smt.eq %44, %c-1_bv1_21 : !smt.bv<1>
    %49 = smt.ite %48, %c6_bv4, %42 : !smt.bv<4>
    %50 = smt.eq %5, %c6_bv4 : !smt.bv<4>
    %c0_bv1_22 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_23 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %51 = smt.ite %50, %c-1_bv1_23, %c0_bv1_22 : !smt.bv<1>
    %52 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_24 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %53 = smt.eq %51, %c-1_bv1_24 : !smt.bv<1>
    %54 = smt.ite %53, %52, %47 : !smt.bv<16>
    %c-1_bv1_25 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %55 = smt.eq %51, %c-1_bv1_25 : !smt.bv<1>
    %56 = smt.ite %55, %c7_bv4, %49 : !smt.bv<4>
    %57 = smt.eq %5, %c7_bv4 : !smt.bv<4>
    %c0_bv1_26 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_27 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %58 = smt.ite %57, %c-1_bv1_27, %c0_bv1_26 : !smt.bv<1>
    %59 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_28 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %60 = smt.eq %58, %c-1_bv1_28 : !smt.bv<1>
    %61 = smt.ite %60, %59, %54 : !smt.bv<16>
    %c-1_bv1_29 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %62 = smt.eq %58, %c-1_bv1_29 : !smt.bv<1>
    %63 = smt.ite %62, %c-8_bv4, %56 : !smt.bv<4>
    %64 = smt.eq %5, %c-8_bv4 : !smt.bv<4>
    %c0_bv1_30 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_31 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %65 = smt.ite %64, %c-1_bv1_31, %c0_bv1_30 : !smt.bv<1>
    %66 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_32 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %67 = smt.eq %65, %c-1_bv1_32 : !smt.bv<1>
    %68 = smt.ite %67, %66, %61 : !smt.bv<16>
    %c-1_bv1_33 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %69 = smt.eq %65, %c-1_bv1_33 : !smt.bv<1>
    %70 = smt.ite %69, %c-7_bv4, %63 : !smt.bv<4>
    %71 = smt.eq %5, %c-7_bv4 : !smt.bv<4>
    %c0_bv1_34 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_35 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %72 = smt.ite %71, %c-1_bv1_35, %c0_bv1_34 : !smt.bv<1>
    %73 = smt.bv.add %4, %c1_bv16 : !smt.bv<16>
    %c-1_bv1_36 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %74 = smt.eq %72, %c-1_bv1_36 : !smt.bv<1>
    %75 = smt.ite %74, %73, %68 : !smt.bv<16>
    %76 = builtin.unrealized_conversion_cast %75 : !smt.bv<16> to i16
    %c-1_bv1_37 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %77 = smt.eq %72, %c-1_bv1_37 : !smt.bv<1>
    %78 = smt.ite %77, %c-6_bv4, %70 : !smt.bv<4>
    %79 = smt.eq %5, %c-6_bv4 : !smt.bv<4>
    %c0_bv1_38 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv1_39 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %80 = smt.ite %79, %c-1_bv1_39, %c0_bv1_38 : !smt.bv<1>
    %c-1_bv1_40 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %81 = smt.eq %80, %c-1_bv1_40 : !smt.bv<1>
    %82 = smt.ite %81, %c-6_bv4, %78 : !smt.bv<4>
    %83 = builtin.unrealized_conversion_cast %82 : !smt.bv<4> to i4
    %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
    %84 = seq.initial() {
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %92 = builtin.unrealized_conversion_cast %c0_bv32 : !smt.bv<32> to i32
      seq.yield %92 : i32
    } : () -> !seq.immutable<i32>
    %85 = smt.bv.add %3, %c1_bv32 : !smt.bv<32>
    %86 = builtin.unrealized_conversion_cast %85 : !smt.bv<32> to i32
    %c-1_bv1_41 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %c-1_bv1_42 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %87 = smt.eq %c-1_bv1_41, %c-1_bv1_42 : !smt.bv<1>
    %88 = smt.not %87
    smt.assert %88
    %89 = builtin.unrealized_conversion_cast %83 : i4 to !smt.bv<4>
    %90 = builtin.unrealized_conversion_cast %76 : i16 to !smt.bv<16>
    %91 = builtin.unrealized_conversion_cast %86 : i32 to !smt.bv<32>
    return %89, %90, %91 : !smt.bv<4>, !smt.bv<16>, !smt.bv<32>
  }
}

