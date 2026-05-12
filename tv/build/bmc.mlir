module {
  llvm.func @printf(!llvm.ptr, ...)
  func.func @mbx_fsm() {
    %0 = llvm.mlir.addressof @resultString_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @resultString_0 : !llvm.ptr
    %2 = smt.solver() : () -> i1 {
      %true = arith.constant true
      %false = arith.constant false
      %c6_i32 = arith.constant 6 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_bv8 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
      %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c0_bv2 = smt.bv.constant #smt.bv<0> : !smt.bv<2>
      %4 = func.call @bmc_init() : () -> !smt.bv<1>
      smt.push 1
      %input_0 = smt.declare_fun "input_0" : !smt.bv<1>
      %input_1 = smt.declare_fun "input_1" : !smt.bv<1>
      %input_2 = smt.declare_fun "input_2" : !smt.bv<1>
      %input_3 = smt.declare_fun "input_3" : !smt.bv<1>
      %input_4 = smt.declare_fun "input_4" : !smt.bv<1>
      %input_5 = smt.declare_fun "input_5" : !smt.bv<1>
      %input_6 = smt.declare_fun "input_6" : !smt.bv<1>
      %input_7 = smt.declare_fun "input_7" : !smt.bv<1>
      %input_8 = smt.declare_fun "input_8" : !smt.bv<1>
      %input_10 = smt.declare_fun "input_10" : !smt.bv<1>
      %5:15 = scf.for %arg0 = %c0_i32 to %c6_i32 step %c1_i32 iter_args(%arg1 = %input_0, %arg2 = %input_1, %arg3 = %input_2, %arg4 = %input_3, %arg5 = %input_4, %arg6 = %input_5, %arg7 = %input_6, %arg8 = %input_7, %arg9 = %input_8, %arg10 = %4, %arg11 = %input_10, %arg12 = %c0_bv2, %arg13 = %c0_bv1, %arg14 = %c0_bv8, %arg15 = %false) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<2>, !smt.bv<1>, !smt.bv<8>, i1)  : i32 {
        smt.pop 1
        smt.push 1
        %7:12 = func.call @bmc_circuit(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<2>, !smt.bv<1>, !smt.bv<8>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<2>, !smt.bv<1>, !smt.bv<8>)
        %8 = smt.check sat {
          smt.yield %true : i1
        } unknown {
          smt.yield %true : i1
        } unsat {
          smt.yield %false : i1
        } -> i1
        %9 = arith.ori %8, %arg15 : i1
        %10 = func.call @bmc_loop(%arg10) : (!smt.bv<1>) -> !smt.bv<1>
        %input_0_0 = smt.declare_fun "input_0" : !smt.bv<1>
        %input_1_1 = smt.declare_fun "input_1" : !smt.bv<1>
        %input_2_2 = smt.declare_fun "input_2" : !smt.bv<1>
        %input_3_3 = smt.declare_fun "input_3" : !smt.bv<1>
        %input_4_4 = smt.declare_fun "input_4" : !smt.bv<1>
        %input_5_5 = smt.declare_fun "input_5" : !smt.bv<1>
        %input_6_6 = smt.declare_fun "input_6" : !smt.bv<1>
        %input_7_7 = smt.declare_fun "input_7" : !smt.bv<1>
        %input_8_8 = smt.declare_fun "input_8" : !smt.bv<1>
        %input_10_9 = smt.declare_fun "input_10" : !smt.bv<1>
        scf.yield %input_0_0, %input_1_1, %input_2_2, %input_3_3, %input_4_4, %input_5_5, %input_6_6, %input_7_7, %input_8_8, %10, %input_10_9, %7#9, %7#10, %7#11, %9 : !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<2>, !smt.bv<1>, !smt.bv<8>, i1
      }
      %6 = arith.xori %5#14, %true : i1
      smt.yield %6 : i1
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
  func.func @bmc_circuit(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<1>, %arg6: !smt.bv<1>, %arg7: !smt.bv<1>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<1>, %arg11: !smt.bv<2>, %arg12: !smt.bv<1>, %arg13: !smt.bv<8>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<2>, !smt.bv<1>, !smt.bv<8>) {
    %c1_bv8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
    %c-4_bv3 = smt.bv.constant #smt.bv<-4> : !smt.bv<3>
    %c0_bv3 = smt.bv.constant #smt.bv<0> : !smt.bv<3>
    %c-3_bv3 = smt.bv.constant #smt.bv<-3> : !smt.bv<3>
    %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c-1_bv2 = smt.bv.constant #smt.bv<-1> : !smt.bv<2>
    %c-2_bv2 = smt.bv.constant #smt.bv<-2> : !smt.bv<2>
    %c1_bv2 = smt.bv.constant #smt.bv<1> : !smt.bv<2>
    %c0_bv2 = smt.bv.constant #smt.bv<0> : !smt.bv<2>
    %0 = builtin.unrealized_conversion_cast %arg13 : !smt.bv<8> to i8
    %1 = builtin.unrealized_conversion_cast %arg12 : !smt.bv<1> to i1
    %2 = builtin.unrealized_conversion_cast %arg11 : !smt.bv<2> to i2
    %3 = builtin.unrealized_conversion_cast %arg10 : !smt.bv<1> to i1
    %4 = builtin.unrealized_conversion_cast %arg9 : !smt.bv<1> to !seq.clock
    %5 = builtin.unrealized_conversion_cast %arg8 : !smt.bv<1> to i1
    %6 = builtin.unrealized_conversion_cast %arg7 : !smt.bv<1> to i1
    %7 = builtin.unrealized_conversion_cast %arg6 : !smt.bv<1> to i1
    %8 = builtin.unrealized_conversion_cast %arg5 : !smt.bv<1> to i1
    %9 = builtin.unrealized_conversion_cast %arg4 : !smt.bv<1> to i1
    %10 = builtin.unrealized_conversion_cast %arg3 : !smt.bv<1> to i1
    %11 = builtin.unrealized_conversion_cast %arg2 : !smt.bv<1> to i1
    %12 = builtin.unrealized_conversion_cast %arg1 : !smt.bv<1> to i1
    %13 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to i1
    %14 = smt.bv.and %arg1, %arg6 : !smt.bv<1>
    %15 = smt.bv.or %arg3, %arg4 : !smt.bv<1>
    %16 = smt.bv.or %15, %arg2 : !smt.bv<1>
    %17 = smt.bv.or %14, %16 : !smt.bv<1>
    %18 = smt.bv.xor %16, %c-1_bv1 : !smt.bv<1>
    %19 = smt.bv.and %arg1, %arg6 : !smt.bv<1>
    %20 = smt.bv.xor %arg3, %c-1_bv1 : !smt.bv<1>
    %21 = smt.eq %20, %c-1_bv1 : !smt.bv<1>
    %22 = smt.ite %21, %arg4, %arg5 : !smt.bv<1>
    %23 = smt.eq %20, %c-1_bv1 : !smt.bv<1>
    %24 = smt.ite %23, %c-3_bv3, %c0_bv3 : !smt.bv<3>
    %25 = smt.bv.xor %19, %c-1_bv1 : !smt.bv<1>
    %26 = smt.bv.or %arg3, %25 : !smt.bv<1>
    %27 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %28 = smt.bv.and %20, %27 : !smt.bv<1>
    %29 = smt.bv.xor %28, %c-1_bv1 : !smt.bv<1>
    %30 = smt.bv.and %27, %arg3 : !smt.bv<1>
    %31 = smt.bv.xor %30, %c-1_bv1 : !smt.bv<1>
    %32 = smt.bv.and %27, %31 : !smt.bv<1>
    %33 = smt.bv.and %32, %29 : !smt.bv<1>
    %34 = smt.bv.extract %24 from 2 : (!smt.bv<3>) -> !smt.bv<1>
    %35 = smt.bv.extract %24 from 0 : (!smt.bv<3>) -> !smt.bv<1>
    %36 = smt.bv.concat %34, %35 : !smt.bv<1>, !smt.bv<1>
    %37 = smt.eq %36, %c-1_bv2 : !smt.bv<2>
    %38 = smt.ite %37, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %39 = smt.bv.xor %30, %c-1_bv1 : !smt.bv<1>
    %40 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %41 = smt.bv.and %40, %39 : !smt.bv<1>
    %42 = smt.bv.and %41, %28 : !smt.bv<1>
    %43 = smt.bv.and %42, %22 : !smt.bv<1>
    %44 = smt.bv.and %43, %38 : !smt.bv<1>
    %45 = smt.bv.xor %26, %c-1_bv1 : !smt.bv<1>
    %46 = smt.bv.xor %22, %c-1_bv1 : !smt.bv<1>
    %47 = smt.bv.xor %30, %c-1_bv1 : !smt.bv<1>
    %48 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %49 = smt.bv.and %48, %47 : !smt.bv<1>
    %50 = smt.bv.and %49, %28 : !smt.bv<1>
    %51 = smt.bv.and %50, %46 : !smt.bv<1>
    %52 = smt.bv.and %51, %45 : !smt.bv<1>
    %53 = smt.eq %arg11, %c0_bv2 : !smt.bv<2>
    %54 = smt.ite %53, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %55 = smt.bv.xor %arg3, %c-1_bv1 : !smt.bv<1>
    %56 = smt.eq %55, %c-1_bv1 : !smt.bv<1>
    %57 = smt.ite %56, %arg4, %arg5 : !smt.bv<1>
    %58 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %59 = smt.bv.and %58, %arg3 : !smt.bv<1>
    %60 = smt.bv.xor %59, %c-1_bv1 : !smt.bv<1>
    %61 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %62 = smt.bv.and %61, %60 : !smt.bv<1>
    %63 = smt.bv.and %62, %58 : !smt.bv<1>
    %64 = smt.bv.and %63, %57 : !smt.bv<1>
    %65 = smt.bv.and %64, %55 : !smt.bv<1>
    %66 = smt.eq %arg11, %c0_bv2 : !smt.bv<2>
    %67 = smt.ite %66, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %68 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %69 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %70 = smt.bv.and %69, %68 : !smt.bv<1>
    %71 = smt.bv.and %70, %arg3 : !smt.bv<1>
    %72 = smt.eq %arg11, %c0_bv2 : !smt.bv<2>
    %73 = smt.ite %72, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %74 = smt.bv.and %arg1, %arg6 : !smt.bv<1>
    %75 = smt.bv.xor %arg3, %c-1_bv1 : !smt.bv<1>
    %76 = smt.eq %75, %c-1_bv1 : !smt.bv<1>
    %77 = smt.ite %76, %arg4, %arg5 : !smt.bv<1>
    %78 = smt.bv.xor %74, %c-1_bv1 : !smt.bv<1>
    %79 = smt.bv.or %arg3, %78 : !smt.bv<1>
    %80 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %81 = smt.bv.and %80, %arg3 : !smt.bv<1>
    %82 = smt.bv.xor %79, %c-1_bv1 : !smt.bv<1>
    %83 = smt.bv.xor %77, %c-1_bv1 : !smt.bv<1>
    %84 = smt.bv.xor %81, %c-1_bv1 : !smt.bv<1>
    %85 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %86 = smt.bv.and %85, %84 : !smt.bv<1>
    %87 = smt.bv.and %86, %75 : !smt.bv<1>
    %88 = smt.bv.and %87, %80 : !smt.bv<1>
    %89 = smt.bv.and %88, %83 : !smt.bv<1>
    %90 = smt.bv.and %89, %82 : !smt.bv<1>
    %91 = smt.eq %arg11, %c0_bv2 : !smt.bv<2>
    %92 = smt.ite %91, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %93 = smt.bv.and %arg1, %arg6 : !smt.bv<1>
    %94 = smt.bv.xor %arg3, %c-1_bv1 : !smt.bv<1>
    %95 = smt.eq %94, %c-1_bv1 : !smt.bv<1>
    %96 = smt.ite %95, %arg4, %arg5 : !smt.bv<1>
    %97 = smt.bv.xor %93, %c-1_bv1 : !smt.bv<1>
    %98 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %99 = smt.bv.and %94, %98 : !smt.bv<1>
    %100 = smt.bv.and %98, %arg3 : !smt.bv<1>
    %101 = smt.bv.xor %96, %c-1_bv1 : !smt.bv<1>
    %102 = smt.bv.and %101, %97 : !smt.bv<1>
    %103 = smt.bv.xor %99, %c-1_bv1 : !smt.bv<1>
    %104 = smt.bv.or %103, %102 : !smt.bv<1>
    %105 = smt.bv.or %104, %arg3 : !smt.bv<1>
    %106 = smt.bv.xor %100, %c-1_bv1 : !smt.bv<1>
    %107 = smt.bv.and %106, %105 : !smt.bv<1>
    %108 = smt.bv.or %arg2, %107 : !smt.bv<1>
    %109 = smt.eq %arg11, %c0_bv2 : !smt.bv<2>
    %110 = smt.ite %109, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %111 = smt.eq %110, %c-1_bv1 : !smt.bv<1>
    %112 = smt.ite %111, %c0_bv2, %arg11 : !smt.bv<2>
    %113 = smt.bv.and %108, %92 : !smt.bv<1>
    %114 = smt.eq %113, %c-1_bv1 : !smt.bv<1>
    %115 = smt.ite %114, %c0_bv1, %arg12 : !smt.bv<1>
    %116 = smt.eq %92, %c-1_bv1 : !smt.bv<1>
    %117 = smt.ite %116, %c0_bv2, %112 : !smt.bv<2>
    %118 = smt.eq %90, %c-1_bv1 : !smt.bv<1>
    %119 = smt.ite %118, %c1_bv2, %c0_bv2 : !smt.bv<2>
    %120 = smt.bv.and %90, %73 : !smt.bv<1>
    %121 = smt.eq %120, %c-1_bv1 : !smt.bv<1>
    %122 = smt.ite %121, %c0_bv1, %115 : !smt.bv<1>
    %123 = smt.eq %73, %c-1_bv1 : !smt.bv<1>
    %124 = smt.ite %123, %119, %117 : !smt.bv<2>
    %125 = smt.eq %71, %c-1_bv1 : !smt.bv<1>
    %126 = smt.ite %125, %c-2_bv2, %119 : !smt.bv<2>
    %127 = smt.bv.and %71, %67 : !smt.bv<1>
    %128 = smt.eq %127, %c-1_bv1 : !smt.bv<1>
    %129 = smt.ite %128, %c0_bv1, %122 : !smt.bv<1>
    %130 = smt.eq %67, %c-1_bv1 : !smt.bv<1>
    %131 = smt.ite %130, %126, %124 : !smt.bv<2>
    %132 = smt.eq %65, %c-1_bv1 : !smt.bv<1>
    %133 = smt.ite %132, %c-1_bv2, %126 : !smt.bv<2>
    %134 = smt.bv.and %65, %54 : !smt.bv<1>
    %135 = smt.eq %134, %c-1_bv1 : !smt.bv<1>
    %136 = smt.ite %135, %c0_bv1, %129 : !smt.bv<1>
    %137 = smt.eq %54, %c-1_bv1 : !smt.bv<1>
    %138 = smt.ite %137, %133, %131 : !smt.bv<2>
    %139 = smt.bv.or %arg3, %arg4 : !smt.bv<1>
    %140 = smt.bv.or %139, %arg2 : !smt.bv<1>
    %141 = smt.bv.or %140, %arg5 : !smt.bv<1>
    %142 = smt.bv.xor %141, %c-1_bv1 : !smt.bv<1>
    %143 = smt.bv.xor %arg3, %c-1_bv1 : !smt.bv<1>
    %144 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %145 = smt.bv.and %143, %144 : !smt.bv<1>
    %146 = smt.bv.and %145, %arg4 : !smt.bv<1>
    %147 = smt.eq %146, %c-1_bv1 : !smt.bv<1>
    %148 = smt.ite %147, %c-3_bv3, %c0_bv3 : !smt.bv<3>
    %149 = smt.bv.xor %146, %c-1_bv1 : !smt.bv<1>
    %150 = smt.bv.and %144, %arg3 : !smt.bv<1>
    %151 = smt.eq %150, %c-1_bv1 : !smt.bv<1>
    %152 = smt.ite %151, %c-4_bv3, %148 : !smt.bv<3>
    %153 = smt.bv.xor %150, %c-1_bv1 : !smt.bv<1>
    %154 = smt.bv.xor %arg4, %c-1_bv1 : !smt.bv<1>
    %155 = smt.bv.and %154, %145 : !smt.bv<1>
    %156 = smt.bv.xor %155, %c-1_bv1 : !smt.bv<1>
    %157 = smt.bv.and %144, %156 : !smt.bv<1>
    %158 = smt.bv.and %157, %153 : !smt.bv<1>
    %159 = smt.bv.and %158, %149 : !smt.bv<1>
    %160 = smt.bv.extract %152 from 2 : (!smt.bv<3>) -> !smt.bv<1>
    %161 = smt.bv.extract %152 from 0 : (!smt.bv<3>) -> !smt.bv<1>
    %162 = smt.bv.concat %160, %161 : !smt.bv<1>, !smt.bv<1>
    %163 = smt.eq %162, %c-1_bv2 : !smt.bv<2>
    %164 = smt.ite %163, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %165 = smt.bv.xor %155, %c-1_bv1 : !smt.bv<1>
    %166 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %167 = smt.bv.and %166, %165 : !smt.bv<1>
    %168 = smt.bv.and %167, %164 : !smt.bv<1>
    %169 = smt.eq %arg11, %c1_bv2 : !smt.bv<2>
    %170 = smt.ite %169, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %171 = smt.eq %170, %c-1_bv1 : !smt.bv<1>
    %172 = smt.ite %171, %c0_bv1, %arg1 : !smt.bv<1>
    %173 = smt.eq %170, %c-1_bv1 : !smt.bv<1>
    %174 = smt.ite %173, %141, %17 : !smt.bv<1>
    %175 = smt.eq %170, %c-1_bv1 : !smt.bv<1>
    %176 = smt.ite %175, %142, %18 : !smt.bv<1>
    %177 = smt.eq %170, %c-1_bv1 : !smt.bv<1>
    %178 = smt.ite %177, %c0_bv1, %52 : !smt.bv<1>
    %179 = smt.eq %170, %c-1_bv1 : !smt.bv<1>
    %180 = smt.ite %179, %168, %44 : !smt.bv<1>
    %181 = smt.eq %170, %c-1_bv1 : !smt.bv<1>
    %182 = smt.ite %181, %159, %33 : !smt.bv<1>
    %183 = smt.eq %arg11, %c1_bv2 : !smt.bv<2>
    %184 = smt.ite %183, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %185 = smt.bv.xor %arg3, %c-1_bv1 : !smt.bv<1>
    %186 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %187 = smt.bv.and %185, %186 : !smt.bv<1>
    %188 = smt.bv.and %186, %arg3 : !smt.bv<1>
    %189 = smt.bv.xor %arg4, %c-1_bv1 : !smt.bv<1>
    %190 = smt.bv.and %189, %187 : !smt.bv<1>
    %191 = smt.bv.xor %188, %c-1_bv1 : !smt.bv<1>
    %192 = smt.bv.xor %190, %c-1_bv1 : !smt.bv<1>
    %193 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %194 = smt.bv.and %193, %192 : !smt.bv<1>
    %195 = smt.bv.and %194, %191 : !smt.bv<1>
    %196 = smt.bv.and %195, %187 : !smt.bv<1>
    %197 = smt.bv.and %196, %arg4 : !smt.bv<1>
    %198 = smt.eq %arg11, %c1_bv2 : !smt.bv<2>
    %199 = smt.ite %198, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %200 = smt.bv.xor %arg3, %c-1_bv1 : !smt.bv<1>
    %201 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %202 = smt.bv.xor %arg4, %c-1_bv1 : !smt.bv<1>
    %203 = smt.bv.and %202, %200 : !smt.bv<1>
    %204 = smt.bv.and %203, %201 : !smt.bv<1>
    %205 = smt.bv.xor %204, %c-1_bv1 : !smt.bv<1>
    %206 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %207 = smt.bv.and %206, %205 : !smt.bv<1>
    %208 = smt.bv.and %207, %201 : !smt.bv<1>
    %209 = smt.bv.and %208, %arg3 : !smt.bv<1>
    %210 = smt.eq %arg11, %c1_bv2 : !smt.bv<2>
    %211 = smt.ite %210, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %212 = smt.bv.xor %arg3, %c-1_bv1 : !smt.bv<1>
    %213 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %214 = smt.bv.xor %arg4, %c-1_bv1 : !smt.bv<1>
    %215 = smt.bv.xor %arg5, %c-1_bv1 : !smt.bv<1>
    %216 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %217 = smt.bv.and %216, %214 : !smt.bv<1>
    %218 = smt.bv.and %217, %212 : !smt.bv<1>
    %219 = smt.bv.and %218, %213 : !smt.bv<1>
    %220 = smt.bv.and %219, %215 : !smt.bv<1>
    %221 = smt.eq %arg11, %c1_bv2 : !smt.bv<2>
    %222 = smt.ite %221, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %223 = smt.bv.xor %arg3, %c-1_bv1 : !smt.bv<1>
    %224 = smt.bv.xor %arg4, %c-1_bv1 : !smt.bv<1>
    %225 = smt.bv.and %223, %224 : !smt.bv<1>
    %226 = smt.bv.and %225, %arg5 : !smt.bv<1>
    %227 = smt.bv.xor %arg3, %c-1_bv1 : !smt.bv<1>
    %228 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %229 = smt.bv.and %227, %228 : !smt.bv<1>
    %230 = smt.bv.and %229, %arg4 : !smt.bv<1>
    %231 = smt.bv.and %228, %arg3 : !smt.bv<1>
    %232 = smt.bv.xor %arg4, %c-1_bv1 : !smt.bv<1>
    %233 = smt.bv.and %232, %229 : !smt.bv<1>
    %234 = smt.bv.xor %230, %c-1_bv1 : !smt.bv<1>
    %235 = smt.bv.xor %231, %c-1_bv1 : !smt.bv<1>
    %236 = smt.bv.and %235, %234 : !smt.bv<1>
    %237 = smt.eq %233, %c-1_bv1 : !smt.bv<1>
    %238 = smt.ite %237, %arg5, %236 : !smt.bv<1>
    %239 = smt.bv.or %arg2, %238 : !smt.bv<1>
    %240 = smt.eq %arg11, %c1_bv2 : !smt.bv<2>
    %241 = smt.ite %240, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %242 = smt.eq %241, %c-1_bv1 : !smt.bv<1>
    %243 = smt.ite %242, %c1_bv2, %138 : !smt.bv<2>
    %244 = smt.eq %239, %c-1_bv1 : !smt.bv<1>
    %245 = smt.ite %244, %c0_bv2, %c1_bv2 : !smt.bv<2>
    %246 = smt.bv.and %239, %222 : !smt.bv<1>
    %247 = smt.eq %246, %c-1_bv1 : !smt.bv<1>
    %248 = smt.ite %247, %226, %136 : !smt.bv<1>
    %249 = smt.eq %222, %c-1_bv1 : !smt.bv<1>
    %250 = smt.ite %249, %245, %243 : !smt.bv<2>
    %251 = smt.eq %220, %c-1_bv1 : !smt.bv<1>
    %252 = smt.ite %251, %c1_bv2, %245 : !smt.bv<2>
    %253 = smt.bv.and %220, %211 : !smt.bv<1>
    %254 = smt.eq %253, %c-1_bv1 : !smt.bv<1>
    %255 = smt.ite %254, %c0_bv1, %248 : !smt.bv<1>
    %256 = smt.eq %211, %c-1_bv1 : !smt.bv<1>
    %257 = smt.ite %256, %252, %250 : !smt.bv<2>
    %258 = smt.eq %209, %c-1_bv1 : !smt.bv<1>
    %259 = smt.ite %258, %c-2_bv2, %252 : !smt.bv<2>
    %260 = smt.bv.and %209, %199 : !smt.bv<1>
    %261 = smt.eq %260, %c-1_bv1 : !smt.bv<1>
    %262 = smt.ite %261, %c0_bv1, %255 : !smt.bv<1>
    %263 = smt.eq %199, %c-1_bv1 : !smt.bv<1>
    %264 = smt.ite %263, %259, %257 : !smt.bv<2>
    %265 = smt.eq %197, %c-1_bv1 : !smt.bv<1>
    %266 = smt.ite %265, %c-1_bv2, %259 : !smt.bv<2>
    %267 = smt.bv.and %197, %184 : !smt.bv<1>
    %268 = smt.eq %267, %c-1_bv1 : !smt.bv<1>
    %269 = smt.ite %268, %c0_bv1, %262 : !smt.bv<1>
    %270 = smt.eq %184, %c-1_bv1 : !smt.bv<1>
    %271 = smt.ite %270, %266, %264 : !smt.bv<2>
    %272 = smt.bv.or %arg3, %arg4 : !smt.bv<1>
    %273 = smt.bv.or %272, %arg2 : !smt.bv<1>
    %274 = smt.bv.xor %273, %c-1_bv1 : !smt.bv<1>
    %275 = smt.bv.concat %c-2_bv2, %arg4 : !smt.bv<2>, !smt.bv<1>
    %276 = smt.eq %arg2, %c-1_bv1 : !smt.bv<1>
    %277 = smt.ite %276, %c0_bv3, %275 : !smt.bv<3>
    %278 = smt.bv.extract %277 from 2 : (!smt.bv<3>) -> !smt.bv<1>
    %279 = smt.bv.extract %277 from 0 : (!smt.bv<3>) -> !smt.bv<1>
    %280 = smt.bv.concat %278, %279 : !smt.bv<1>, !smt.bv<1>
    %281 = smt.eq %280, %c-1_bv2 : !smt.bv<2>
    %282 = smt.ite %281, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %283 = smt.eq %arg11, %c-2_bv2 : !smt.bv<2>
    %284 = smt.ite %283, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %285 = smt.eq %284, %c-1_bv1 : !smt.bv<1>
    %286 = smt.ite %285, %c0_bv1, %172 : !smt.bv<1>
    %287 = smt.eq %284, %c-1_bv1 : !smt.bv<1>
    %288 = smt.ite %287, %c0_bv1, %c0_bv1 : !smt.bv<1>
    %289 = smt.eq %284, %c-1_bv1 : !smt.bv<1>
    %290 = smt.ite %289, %c0_bv1, %170 : !smt.bv<1>
    %291 = smt.eq %284, %c-1_bv1 : !smt.bv<1>
    %292 = smt.ite %291, %c0_bv1, %c0_bv1 : !smt.bv<1>
    %293 = smt.eq %284, %c-1_bv1 : !smt.bv<1>
    %294 = smt.ite %293, %273, %174 : !smt.bv<1>
    %295 = smt.eq %284, %c-1_bv1 : !smt.bv<1>
    %296 = smt.ite %295, %274, %176 : !smt.bv<1>
    %297 = smt.eq %284, %c-1_bv1 : !smt.bv<1>
    %298 = smt.ite %297, %c0_bv1, %178 : !smt.bv<1>
    %299 = smt.eq %284, %c-1_bv1 : !smt.bv<1>
    %300 = smt.ite %299, %282, %180 : !smt.bv<1>
    %301 = smt.eq %284, %c-1_bv1 : !smt.bv<1>
    %302 = smt.ite %301, %c0_bv1, %182 : !smt.bv<1>
    %303 = smt.eq %arg11, %c-2_bv2 : !smt.bv<2>
    %304 = smt.ite %303, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %305 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %306 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %307 = smt.bv.and %306, %305 : !smt.bv<1>
    %308 = smt.bv.and %307, %arg4 : !smt.bv<1>
    %309 = smt.eq %arg11, %c-2_bv2 : !smt.bv<2>
    %310 = smt.ite %309, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %311 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %312 = smt.bv.xor %arg4, %c-1_bv1 : !smt.bv<1>
    %313 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %314 = smt.bv.and %313, %311 : !smt.bv<1>
    %315 = smt.bv.and %314, %312 : !smt.bv<1>
    %316 = smt.eq %arg11, %c-2_bv2 : !smt.bv<2>
    %317 = smt.ite %316, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %318 = smt.eq %arg11, %c-2_bv2 : !smt.bv<2>
    %319 = smt.ite %318, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %320 = smt.eq %319, %c-1_bv1 : !smt.bv<1>
    %321 = smt.ite %320, %c-2_bv2, %271 : !smt.bv<2>
    %322 = smt.eq %arg2, %c-1_bv1 : !smt.bv<1>
    %323 = smt.ite %322, %c0_bv2, %c-2_bv2 : !smt.bv<2>
    %324 = smt.bv.and %arg2, %317 : !smt.bv<1>
    %325 = smt.eq %324, %c-1_bv1 : !smt.bv<1>
    %326 = smt.ite %325, %c0_bv1, %269 : !smt.bv<1>
    %327 = smt.eq %317, %c-1_bv1 : !smt.bv<1>
    %328 = smt.ite %327, %323, %321 : !smt.bv<2>
    %329 = smt.eq %315, %c-1_bv1 : !smt.bv<1>
    %330 = smt.ite %329, %c-2_bv2, %323 : !smt.bv<2>
    %331 = smt.bv.and %315, %310 : !smt.bv<1>
    %332 = smt.eq %331, %c-1_bv1 : !smt.bv<1>
    %333 = smt.ite %332, %c0_bv1, %326 : !smt.bv<1>
    %334 = smt.eq %310, %c-1_bv1 : !smt.bv<1>
    %335 = smt.ite %334, %330, %328 : !smt.bv<2>
    %336 = smt.eq %308, %c-1_bv1 : !smt.bv<1>
    %337 = smt.ite %336, %c-1_bv2, %330 : !smt.bv<2>
    %338 = smt.bv.and %308, %304 : !smt.bv<1>
    %339 = smt.eq %338, %c-1_bv1 : !smt.bv<1>
    %340 = smt.ite %339, %c0_bv1, %333 : !smt.bv<1>
    %341 = smt.eq %304, %c-1_bv1 : !smt.bv<1>
    %342 = smt.ite %341, %337, %335 : !smt.bv<2>
    %343 = smt.bv.or %arg3, %arg4 : !smt.bv<1>
    %344 = smt.bv.or %343, %arg2 : !smt.bv<1>
    %345 = smt.bv.xor %344, %c-1_bv1 : !smt.bv<1>
    %346 = smt.eq %arg11, %c-1_bv2 : !smt.bv<2>
    %347 = smt.ite %346, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %348 = smt.eq %347, %c-1_bv1 : !smt.bv<1>
    %349 = smt.ite %348, %c0_bv1, %286 : !smt.bv<1>
    %350 = builtin.unrealized_conversion_cast %349 : !smt.bv<1> to i1
    %351 = smt.eq %347, %c-1_bv1 : !smt.bv<1>
    %352 = smt.ite %351, %c0_bv1, %288 : !smt.bv<1>
    %353 = builtin.unrealized_conversion_cast %352 : !smt.bv<1> to i1
    %354 = smt.eq %347, %c-1_bv1 : !smt.bv<1>
    %355 = smt.ite %354, %c0_bv1, %290 : !smt.bv<1>
    %356 = builtin.unrealized_conversion_cast %355 : !smt.bv<1> to i1
    %357 = smt.eq %347, %c-1_bv1 : !smt.bv<1>
    %358 = smt.ite %357, %c-1_bv1, %292 : !smt.bv<1>
    %359 = builtin.unrealized_conversion_cast %358 : !smt.bv<1> to i1
    %360 = smt.eq %347, %c-1_bv1 : !smt.bv<1>
    %361 = smt.ite %360, %344, %294 : !smt.bv<1>
    %362 = builtin.unrealized_conversion_cast %361 : !smt.bv<1> to i1
    %363 = smt.eq %347, %c-1_bv1 : !smt.bv<1>
    %364 = smt.ite %363, %345, %296 : !smt.bv<1>
    %365 = builtin.unrealized_conversion_cast %364 : !smt.bv<1> to i1
    %366 = smt.eq %347, %c-1_bv1 : !smt.bv<1>
    %367 = smt.ite %366, %c0_bv1, %298 : !smt.bv<1>
    %368 = builtin.unrealized_conversion_cast %367 : !smt.bv<1> to i1
    %369 = smt.eq %347, %c-1_bv1 : !smt.bv<1>
    %370 = smt.ite %369, %c0_bv1, %300 : !smt.bv<1>
    %371 = builtin.unrealized_conversion_cast %370 : !smt.bv<1> to i1
    %372 = smt.eq %347, %c-1_bv1 : !smt.bv<1>
    %373 = smt.ite %372, %c0_bv1, %302 : !smt.bv<1>
    %374 = builtin.unrealized_conversion_cast %373 : !smt.bv<1> to i1
    %375 = smt.eq %arg11, %c-1_bv2 : !smt.bv<2>
    %376 = smt.ite %375, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %377 = smt.bv.xor %arg2, %c-1_bv1 : !smt.bv<1>
    %378 = smt.eq %arg11, %c-1_bv2 : !smt.bv<2>
    %379 = smt.ite %378, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %380 = smt.eq %arg11, %c-1_bv2 : !smt.bv<2>
    %381 = smt.ite %380, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %382 = smt.eq %381, %c-1_bv1 : !smt.bv<1>
    %383 = smt.ite %382, %c-1_bv2, %342 : !smt.bv<2>
    %384 = smt.eq %arg2, %c-1_bv1 : !smt.bv<1>
    %385 = smt.ite %384, %c0_bv2, %c-1_bv2 : !smt.bv<2>
    %386 = smt.bv.and %arg2, %379 : !smt.bv<1>
    %387 = smt.eq %386, %c-1_bv1 : !smt.bv<1>
    %388 = smt.ite %387, %c0_bv1, %340 : !smt.bv<1>
    %389 = smt.eq %379, %c-1_bv1 : !smt.bv<1>
    %390 = smt.ite %389, %385, %383 : !smt.bv<2>
    %391 = smt.eq %377, %c-1_bv1 : !smt.bv<1>
    %392 = smt.ite %391, %c-1_bv2, %385 : !smt.bv<2>
    %393 = smt.bv.and %377, %376 : !smt.bv<1>
    %394 = smt.eq %393, %c-1_bv1 : !smt.bv<1>
    %395 = smt.ite %394, %c0_bv1, %388 : !smt.bv<1>
    %396 = smt.eq %376, %c-1_bv1 : !smt.bv<1>
    %397 = smt.ite %396, %392, %390 : !smt.bv<2>
    %398 = smt.bv.add %arg13, %c1_bv8 : !smt.bv<8>
    %399 = builtin.unrealized_conversion_cast %398 : !smt.bv<8> to i8
    %400 = smt.eq %arg10, %c-1_bv1 : !smt.bv<1>
    %401 = smt.ite %400, %c0_bv2, %397 : !smt.bv<2>
    %402 = builtin.unrealized_conversion_cast %401 : !smt.bv<2> to i2
    %403 = smt.eq %arg10, %c-1_bv1 : !smt.bv<1>
    %404 = smt.ite %403, %c0_bv1, %395 : !smt.bv<1>
    %405 = builtin.unrealized_conversion_cast %404 : !smt.bv<1> to i1
    %406 = dbg.scope "mbx_fsm", "mbx_fsm"
    dbg.variable "in0", %13 scope %406 : i1
    dbg.variable "in1", %12 scope %406 : i1
    dbg.variable "in2", %11 scope %406 : i1
    dbg.variable "in3", %10 scope %406 : i1
    dbg.variable "in4", %9 scope %406 : i1
    dbg.variable "in5", %8 scope %406 : i1
    dbg.variable "in6", %7 scope %406 : i1
    dbg.variable "in7", %6 scope %406 : i1
    dbg.variable "in8", %5 scope %406 : i1
    dbg.variable "clk", %4 scope %406 : !seq.clock
    dbg.variable "rst", %3 scope %406 : i1
    dbg.variable "state_reg_state", %2 scope %406 : i2
    dbg.variable "_sh1_state", %1 scope %406 : i1
    dbg.variable "time_reg_state", %0 scope %406 : i8
    dbg.variable "out0", %350 scope %406 : i1
    dbg.variable "out1", %353 scope %406 : i1
    dbg.variable "out2", %356 scope %406 : i1
    dbg.variable "out3", %359 scope %406 : i1
    dbg.variable "out4", %362 scope %406 : i1
    dbg.variable "out5", %365 scope %406 : i1
    dbg.variable "out6", %368 scope %406 : i1
    dbg.variable "out7", %371 scope %406 : i1
    dbg.variable "out8", %374 scope %406 : i1
    dbg.variable "state_reg_next", %402 scope %406 : i2
    dbg.variable "_sh1_next", %405 scope %406 : i1
    dbg.variable "time_reg_next", %399 scope %406 : i8
    return %349, %352, %355, %358, %361, %364, %367, %370, %373, %401, %404, %398 : !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<2>, !smt.bv<1>, !smt.bv<8>
  }
}

