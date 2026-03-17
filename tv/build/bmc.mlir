module {
  llvm.func @printf(!llvm.ptr, ...)
  func.func @fsm10() {
    %0 = llvm.mlir.addressof @resultString_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @resultString_0 : !llvm.ptr
    %2 = smt.solver() : () -> i1 {
      %true = arith.constant true
      %false = arith.constant false
      %c0_bv8 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
      %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
      %c0_bv2 = smt.bv.constant #smt.bv<0> : !smt.bv<2>
      %4 = func.call @bmc_init() : () -> !smt.bv<1>
      smt.push 1
      %input_0 = smt.declare_fun "input_0" : !smt.bv<1>
      %input_1 = smt.declare_fun "input_1" : !smt.bv<1>
      %input_2 = smt.declare_fun "input_2" : !smt.bv<1>
      %input_3 = smt.declare_fun "input_3" : !smt.bv<16>
      %input_5 = smt.declare_fun "input_5" : !smt.bv<1>
      smt.pop 1
      smt.push 1
      %5:8 = func.call @bmc_circuit(%input_0, %input_1, %input_2, %input_3, %4, %input_5, %c0_bv2, %c0_bv16, %c0_bv1, %c0_bv1, %c0_bv8) : (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<2>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<2>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>)
      %6 = smt.check sat {
        smt.yield %true : i1
      } unknown {
        smt.yield %true : i1
      } unsat {
        smt.yield %false : i1
      } -> i1
      %7 = func.call @bmc_loop(%4) : (!smt.bv<1>) -> !smt.bv<1>
      %8 = arith.xori %6, %true : i1
      smt.yield %8 : i1
    }
    %3 = llvm.select %2, %1, %0 : i1, !llvm.ptr
    llvm.call @printf(%3) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr) -> ()
    return
  }
  llvm.mlir.global private constant @resultString_0("Bound reached with no violations!\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @resultString_1("Assertion can be violated!\0A\00") {addr_space = 0 : i32}
  func.func @bmc_init() -> !smt.bv<1> {
    %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    return %c-1_bv1 : !smt.bv<1>
  }
  func.func @bmc_loop(%arg0: !smt.bv<1>) -> !smt.bv<1> {
    return %arg0 : !smt.bv<1>
  }
  func.func @bmc_circuit(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<16>, %arg4: !smt.bv<1>, %arg5: !smt.bv<1>, %arg6: !smt.bv<2>, %arg7: !smt.bv<16>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<8>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<2>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) {
    %c1_bv8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
    %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
    %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %c-1_bv16 = smt.bv.constant #smt.bv<-1> : !smt.bv<16>
    %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
    %c-2_bv2 = smt.bv.constant #smt.bv<-2> : !smt.bv<2>
    %c1_bv2 = smt.bv.constant #smt.bv<1> : !smt.bv<2>
    %c0_bv2 = smt.bv.constant #smt.bv<0> : !smt.bv<2>
    %0 = smt.eq %arg6, %c0_bv2 : !smt.bv<2>
    %1 = smt.ite %0, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %2 = smt.eq %arg6, %c0_bv2 : !smt.bv<2>
    %3 = smt.ite %2, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %4 = smt.bv.or %arg1, %arg2 : !smt.bv<1>
    %5 = smt.eq %arg6, %c0_bv2 : !smt.bv<2>
    %6 = smt.ite %5, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %7 = smt.eq %arg0, %c0_bv1 : !smt.bv<1>
    %8 = smt.ite %7, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %9 = smt.eq %arg1, %c0_bv1 : !smt.bv<1>
    %10 = smt.ite %9, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %11 = smt.eq %arg2, %c0_bv1 : !smt.bv<1>
    %12 = smt.ite %11, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %13 = smt.bv.and %8, %10 : !smt.bv<1>
    %14 = smt.bv.and %13, %12 : !smt.bv<1>
    %15 = smt.eq %arg6, %c0_bv2 : !smt.bv<2>
    %16 = smt.ite %15, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %17 = smt.eq %16, %c-1_bv1 : !smt.bv<1>
    %18 = smt.ite %17, %c0_bv2, %arg6 : !smt.bv<2>
    %19 = smt.bv.and %14, %6 : !smt.bv<1>
    %20 = smt.eq %19, %c-1_bv1 : !smt.bv<1>
    %21 = smt.ite %20, %c0_bv1, %arg9 : !smt.bv<1>
    %22 = smt.eq %6, %c-1_bv1 : !smt.bv<1>
    %23 = smt.ite %22, %c0_bv2, %18 : !smt.bv<2>
    %24 = smt.eq %4, %c-1_bv1 : !smt.bv<1>
    %25 = smt.ite %24, %c-2_bv2, %c0_bv2 : !smt.bv<2>
    %26 = smt.bv.and %4, %3 : !smt.bv<1>
    %27 = smt.eq %26, %c-1_bv1 : !smt.bv<1>
    %28 = smt.ite %27, %c-1_bv1, %21 : !smt.bv<1>
    %29 = smt.eq %3, %c-1_bv1 : !smt.bv<1>
    %30 = smt.ite %29, %25, %23 : !smt.bv<2>
    %31 = smt.eq %arg0, %c-1_bv1 : !smt.bv<1>
    %32 = smt.ite %31, %c1_bv2, %25 : !smt.bv<2>
    %33 = smt.bv.and %arg0, %1 : !smt.bv<1>
    %34 = smt.eq %33, %c-1_bv1 : !smt.bv<1>
    %35 = smt.ite %34, %c0_bv16, %arg7 : !smt.bv<16>
    %36 = smt.eq %33, %c-1_bv1 : !smt.bv<1>
    %37 = smt.ite %36, %c-1_bv1, %arg8 : !smt.bv<1>
    %38 = smt.eq %33, %c-1_bv1 : !smt.bv<1>
    %39 = smt.ite %38, %c0_bv1, %28 : !smt.bv<1>
    %40 = smt.eq %1, %c-1_bv1 : !smt.bv<1>
    %41 = smt.ite %40, %32, %30 : !smt.bv<2>
    %42 = smt.eq %arg6, %c1_bv2 : !smt.bv<2>
    %43 = smt.ite %42, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %44 = smt.eq %43, %c-1_bv1 : !smt.bv<1>
    %45 = smt.ite %44, %c0_bv1, %c-1_bv1 : !smt.bv<1>
    %46 = smt.eq %arg6, %c1_bv2 : !smt.bv<2>
    %47 = smt.ite %46, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %48 = smt.bv.cmp slt %arg3, %c0_bv16 : !smt.bv<16>
    %49 = smt.ite %48, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %50 = smt.bv.add %arg7, %c1_bv16 : !smt.bv<16>
    %51 = smt.eq %c-1_bv16, %arg7 : !smt.bv<16>
    %52 = smt.ite %51, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %53 = smt.eq %arg6, %c1_bv2 : !smt.bv<2>
    %54 = smt.ite %53, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %55 = smt.bv.cmp slt %arg3, %c0_bv16 : !smt.bv<16>
    %56 = smt.ite %55, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %57 = smt.bv.add %arg7, %c1_bv16 : !smt.bv<16>
    %58 = smt.distinct %c-1_bv16, %arg7 : !smt.bv<16>
    %59 = smt.ite %58, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %60 = smt.bv.or %arg1, %arg2 : !smt.bv<1>
    %61 = smt.bv.xor %60, %c-1_bv1 : !smt.bv<1>
    %62 = smt.bv.and %59, %61 : !smt.bv<1>
    %63 = smt.eq %arg6, %c1_bv2 : !smt.bv<2>
    %64 = smt.ite %63, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %65 = smt.bv.or %arg1, %arg2 : !smt.bv<1>
    %66 = smt.eq %arg6, %c1_bv2 : !smt.bv<2>
    %67 = smt.ite %66, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %68 = smt.eq %67, %c-1_bv1 : !smt.bv<1>
    %69 = smt.ite %68, %c1_bv2, %41 : !smt.bv<2>
    %70 = smt.eq %65, %c-1_bv1 : !smt.bv<1>
    %71 = smt.ite %70, %c-2_bv2, %c1_bv2 : !smt.bv<2>
    %72 = smt.bv.and %65, %64 : !smt.bv<1>
    %73 = smt.eq %72, %c-1_bv1 : !smt.bv<1>
    %74 = smt.ite %73, %c-1_bv1, %39 : !smt.bv<1>
    %75 = smt.eq %64, %c-1_bv1 : !smt.bv<1>
    %76 = smt.ite %75, %71, %69 : !smt.bv<2>
    %77 = smt.eq %62, %c-1_bv1 : !smt.bv<1>
    %78 = smt.ite %77, %c1_bv2, %71 : !smt.bv<2>
    %79 = smt.bv.and %62, %54 : !smt.bv<1>
    %80 = smt.eq %79, %c-1_bv1 : !smt.bv<1>
    %81 = smt.ite %80, %56, %37 : !smt.bv<1>
    %82 = smt.eq %79, %c-1_bv1 : !smt.bv<1>
    %83 = smt.ite %82, %57, %35 : !smt.bv<16>
    %84 = smt.eq %79, %c-1_bv1 : !smt.bv<1>
    %85 = smt.ite %84, %c0_bv1, %74 : !smt.bv<1>
    %86 = smt.eq %54, %c-1_bv1 : !smt.bv<1>
    %87 = smt.ite %86, %78, %76 : !smt.bv<2>
    %88 = smt.eq %52, %c-1_bv1 : !smt.bv<1>
    %89 = smt.ite %88, %c0_bv2, %78 : !smt.bv<2>
    %90 = smt.bv.and %52, %47 : !smt.bv<1>
    %91 = smt.eq %90, %c-1_bv1 : !smt.bv<1>
    %92 = smt.ite %91, %49, %81 : !smt.bv<1>
    %93 = smt.eq %90, %c-1_bv1 : !smt.bv<1>
    %94 = smt.ite %93, %50, %83 : !smt.bv<16>
    %95 = smt.eq %90, %c-1_bv1 : !smt.bv<1>
    %96 = smt.ite %95, %c0_bv1, %85 : !smt.bv<1>
    %97 = smt.eq %47, %c-1_bv1 : !smt.bv<1>
    %98 = smt.ite %97, %89, %87 : !smt.bv<2>
    %99 = smt.eq %arg6, %c-2_bv2 : !smt.bv<2>
    %100 = smt.ite %99, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %101 = smt.eq %100, %c-1_bv1 : !smt.bv<1>
    %102 = smt.ite %101, %arg9, %45 : !smt.bv<1>
    %103 = smt.eq %100, %c-1_bv1 : !smt.bv<1>
    %104 = smt.ite %103, %c-1_bv1, %arg9 : !smt.bv<1>
    %105 = smt.eq %100, %c-1_bv1 : !smt.bv<1>
    %106 = smt.ite %105, %c0_bv1, %43 : !smt.bv<1>
    %107 = smt.eq %arg6, %c-2_bv2 : !smt.bv<2>
    %108 = smt.ite %107, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %109 = smt.eq %108, %c-1_bv1 : !smt.bv<1>
    %110 = smt.ite %109, %c-2_bv2, %98 : !smt.bv<2>
    %111 = smt.bv.add %arg10, %c1_bv8 : !smt.bv<8>
    %112 = smt.eq %arg5, %c-1_bv1 : !smt.bv<1>
    %113 = smt.ite %112, %c0_bv2, %110 : !smt.bv<2>
    %114 = smt.eq %arg5, %c-1_bv1 : !smt.bv<1>
    %115 = smt.ite %114, %c0_bv16, %94 : !smt.bv<16>
    %116 = smt.eq %arg5, %c-1_bv1 : !smt.bv<1>
    %117 = smt.ite %116, %c0_bv1, %92 : !smt.bv<1>
    %118 = smt.eq %arg5, %c-1_bv1 : !smt.bv<1>
    %119 = smt.ite %118, %c0_bv1, %96 : !smt.bv<1>
    return %102, %104, %106, %113, %115, %117, %119, %111 : !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<2>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>
  }
}

