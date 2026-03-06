module {
  llvm.func @printf(!llvm.ptr, ...)
  func.func @fsm30() {
    %0 = llvm.mlir.addressof @resultString_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @resultString_0 : !llvm.ptr
    %2 = smt.solver() : () -> i1 {
      %true = arith.constant true
      %false = arith.constant false
      %c10_i32 = arith.constant 10 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %c0_bv8 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
      %c0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
      %c0_bv5 = smt.bv.constant #smt.bv<0> : !smt.bv<5>
      %4 = func.call @bmc_init() : () -> !smt.bv<1>
      smt.push 1
      %input_1 = smt.declare_fun "input_1" : !smt.bv<1>
      %5:6 = scf.for %arg0 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg1 = %4, %arg2 = %input_1, %arg3 = %c0_bv5, %arg4 = %c0_bv16, %arg5 = %c0_bv8, %arg6 = %false) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<5>, !smt.bv<16>, !smt.bv<8>, i1)  : i32 {
        smt.pop 1
        smt.push 1
        %7:3 = func.call @bmc_circuit(%arg1, %arg2, %arg3, %arg4, %arg5) : (!smt.bv<1>, !smt.bv<1>, !smt.bv<5>, !smt.bv<16>, !smt.bv<8>) -> (!smt.bv<5>, !smt.bv<16>, !smt.bv<8>)
        %8 = smt.check sat {
          smt.yield %true : i1
        } unknown {
          smt.yield %true : i1
        } unsat {
          smt.yield %false : i1
        } -> i1
        %9 = arith.ori %8, %arg6 : i1
        %10 = func.call @bmc_loop(%arg1) : (!smt.bv<1>) -> !smt.bv<1>
        %input_1_0 = smt.declare_fun "input_1" : !smt.bv<1>
        scf.yield %10, %input_1_0, %7#0, %7#1, %7#2, %9 : !smt.bv<1>, !smt.bv<1>, !smt.bv<5>, !smt.bv<16>, !smt.bv<8>, i1
      }
      %6 = arith.xori %5#5, %true : i1
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
  func.func @bmc_circuit(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<5>, %arg3: !smt.bv<16>, %arg4: !smt.bv<8>) -> (!smt.bv<5>, !smt.bv<16>, !smt.bv<8>) {
    %c1_bv8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
    %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
    %c0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
    %c-2_bv5 = smt.bv.constant #smt.bv<-2> : !smt.bv<5>
    %c-3_bv5 = smt.bv.constant #smt.bv<-3> : !smt.bv<5>
    %c-4_bv5 = smt.bv.constant #smt.bv<-4> : !smt.bv<5>
    %c-5_bv5 = smt.bv.constant #smt.bv<-5> : !smt.bv<5>
    %c-6_bv5 = smt.bv.constant #smt.bv<-6> : !smt.bv<5>
    %c-7_bv5 = smt.bv.constant #smt.bv<-7> : !smt.bv<5>
    %c-8_bv5 = smt.bv.constant #smt.bv<-8> : !smt.bv<5>
    %c-9_bv5 = smt.bv.constant #smt.bv<-9> : !smt.bv<5>
    %c-10_bv5 = smt.bv.constant #smt.bv<-10> : !smt.bv<5>
    %c-11_bv5 = smt.bv.constant #smt.bv<-11> : !smt.bv<5>
    %c-12_bv5 = smt.bv.constant #smt.bv<-12> : !smt.bv<5>
    %c-13_bv5 = smt.bv.constant #smt.bv<-13> : !smt.bv<5>
    %c-14_bv5 = smt.bv.constant #smt.bv<-14> : !smt.bv<5>
    %c-15_bv5 = smt.bv.constant #smt.bv<-15> : !smt.bv<5>
    %c-16_bv5 = smt.bv.constant #smt.bv<-16> : !smt.bv<5>
    %c15_bv5 = smt.bv.constant #smt.bv<15> : !smt.bv<5>
    %c14_bv5 = smt.bv.constant #smt.bv<14> : !smt.bv<5>
    %c13_bv5 = smt.bv.constant #smt.bv<13> : !smt.bv<5>
    %c12_bv5 = smt.bv.constant #smt.bv<12> : !smt.bv<5>
    %c11_bv5 = smt.bv.constant #smt.bv<11> : !smt.bv<5>
    %c10_bv5 = smt.bv.constant #smt.bv<10> : !smt.bv<5>
    %c9_bv5 = smt.bv.constant #smt.bv<9> : !smt.bv<5>
    %c8_bv5 = smt.bv.constant #smt.bv<8> : !smt.bv<5>
    %c7_bv5 = smt.bv.constant #smt.bv<7> : !smt.bv<5>
    %c6_bv5 = smt.bv.constant #smt.bv<6> : !smt.bv<5>
    %c5_bv5 = smt.bv.constant #smt.bv<5> : !smt.bv<5>
    %c4_bv5 = smt.bv.constant #smt.bv<4> : !smt.bv<5>
    %c3_bv5 = smt.bv.constant #smt.bv<3> : !smt.bv<5>
    %c2_bv5 = smt.bv.constant #smt.bv<2> : !smt.bv<5>
    %c1_bv5 = smt.bv.constant #smt.bv<1> : !smt.bv<5>
    %c0_bv5 = smt.bv.constant #smt.bv<0> : !smt.bv<5>
    %0 = smt.eq %arg2, %c0_bv5 : !smt.bv<5>
    %1 = smt.ite %0, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %2 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %3 = smt.eq %1, %c-1_bv1 : !smt.bv<1>
    %4 = smt.ite %3, %2, %arg3 : !smt.bv<16>
    %5 = smt.eq %1, %c-1_bv1 : !smt.bv<1>
    %6 = smt.ite %5, %c1_bv5, %arg2 : !smt.bv<5>
    %7 = smt.eq %arg2, %c1_bv5 : !smt.bv<5>
    %8 = smt.ite %7, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %9 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %10 = smt.eq %8, %c-1_bv1 : !smt.bv<1>
    %11 = smt.ite %10, %9, %4 : !smt.bv<16>
    %12 = smt.eq %8, %c-1_bv1 : !smt.bv<1>
    %13 = smt.ite %12, %c2_bv5, %6 : !smt.bv<5>
    %14 = smt.eq %arg2, %c2_bv5 : !smt.bv<5>
    %15 = smt.ite %14, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %16 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %17 = smt.eq %15, %c-1_bv1 : !smt.bv<1>
    %18 = smt.ite %17, %16, %11 : !smt.bv<16>
    %19 = smt.eq %15, %c-1_bv1 : !smt.bv<1>
    %20 = smt.ite %19, %c3_bv5, %13 : !smt.bv<5>
    %21 = smt.eq %arg2, %c3_bv5 : !smt.bv<5>
    %22 = smt.ite %21, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %23 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %24 = smt.eq %22, %c-1_bv1 : !smt.bv<1>
    %25 = smt.ite %24, %23, %18 : !smt.bv<16>
    %26 = smt.eq %22, %c-1_bv1 : !smt.bv<1>
    %27 = smt.ite %26, %c4_bv5, %20 : !smt.bv<5>
    %28 = smt.eq %arg2, %c4_bv5 : !smt.bv<5>
    %29 = smt.ite %28, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %30 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %31 = smt.eq %29, %c-1_bv1 : !smt.bv<1>
    %32 = smt.ite %31, %30, %25 : !smt.bv<16>
    %33 = smt.eq %29, %c-1_bv1 : !smt.bv<1>
    %34 = smt.ite %33, %c5_bv5, %27 : !smt.bv<5>
    %35 = smt.eq %arg2, %c5_bv5 : !smt.bv<5>
    %36 = smt.ite %35, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %37 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %38 = smt.eq %36, %c-1_bv1 : !smt.bv<1>
    %39 = smt.ite %38, %37, %32 : !smt.bv<16>
    %40 = smt.eq %36, %c-1_bv1 : !smt.bv<1>
    %41 = smt.ite %40, %c6_bv5, %34 : !smt.bv<5>
    %42 = smt.eq %arg2, %c6_bv5 : !smt.bv<5>
    %43 = smt.ite %42, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %44 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %45 = smt.eq %43, %c-1_bv1 : !smt.bv<1>
    %46 = smt.ite %45, %44, %39 : !smt.bv<16>
    %47 = smt.eq %43, %c-1_bv1 : !smt.bv<1>
    %48 = smt.ite %47, %c7_bv5, %41 : !smt.bv<5>
    %49 = smt.eq %arg2, %c7_bv5 : !smt.bv<5>
    %50 = smt.ite %49, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %51 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %52 = smt.eq %50, %c-1_bv1 : !smt.bv<1>
    %53 = smt.ite %52, %51, %46 : !smt.bv<16>
    %54 = smt.eq %50, %c-1_bv1 : !smt.bv<1>
    %55 = smt.ite %54, %c8_bv5, %48 : !smt.bv<5>
    %56 = smt.eq %arg2, %c8_bv5 : !smt.bv<5>
    %57 = smt.ite %56, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %58 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %59 = smt.eq %57, %c-1_bv1 : !smt.bv<1>
    %60 = smt.ite %59, %58, %53 : !smt.bv<16>
    %61 = smt.eq %57, %c-1_bv1 : !smt.bv<1>
    %62 = smt.ite %61, %c9_bv5, %55 : !smt.bv<5>
    %63 = smt.eq %arg2, %c9_bv5 : !smt.bv<5>
    %64 = smt.ite %63, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %65 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %66 = smt.eq %64, %c-1_bv1 : !smt.bv<1>
    %67 = smt.ite %66, %65, %60 : !smt.bv<16>
    %68 = smt.eq %64, %c-1_bv1 : !smt.bv<1>
    %69 = smt.ite %68, %c10_bv5, %62 : !smt.bv<5>
    %70 = smt.eq %arg2, %c10_bv5 : !smt.bv<5>
    %71 = smt.ite %70, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %72 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %73 = smt.eq %71, %c-1_bv1 : !smt.bv<1>
    %74 = smt.ite %73, %72, %67 : !smt.bv<16>
    %75 = smt.eq %71, %c-1_bv1 : !smt.bv<1>
    %76 = smt.ite %75, %c11_bv5, %69 : !smt.bv<5>
    %77 = smt.eq %arg2, %c11_bv5 : !smt.bv<5>
    %78 = smt.ite %77, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %79 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %80 = smt.eq %78, %c-1_bv1 : !smt.bv<1>
    %81 = smt.ite %80, %79, %74 : !smt.bv<16>
    %82 = smt.eq %78, %c-1_bv1 : !smt.bv<1>
    %83 = smt.ite %82, %c12_bv5, %76 : !smt.bv<5>
    %84 = smt.eq %arg2, %c12_bv5 : !smt.bv<5>
    %85 = smt.ite %84, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %86 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %87 = smt.eq %85, %c-1_bv1 : !smt.bv<1>
    %88 = smt.ite %87, %86, %81 : !smt.bv<16>
    %89 = smt.eq %85, %c-1_bv1 : !smt.bv<1>
    %90 = smt.ite %89, %c13_bv5, %83 : !smt.bv<5>
    %91 = smt.eq %arg2, %c13_bv5 : !smt.bv<5>
    %92 = smt.ite %91, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %93 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %94 = smt.eq %92, %c-1_bv1 : !smt.bv<1>
    %95 = smt.ite %94, %93, %88 : !smt.bv<16>
    %96 = smt.eq %92, %c-1_bv1 : !smt.bv<1>
    %97 = smt.ite %96, %c14_bv5, %90 : !smt.bv<5>
    %98 = smt.eq %arg2, %c14_bv5 : !smt.bv<5>
    %99 = smt.ite %98, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %100 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %101 = smt.eq %99, %c-1_bv1 : !smt.bv<1>
    %102 = smt.ite %101, %100, %95 : !smt.bv<16>
    %103 = smt.eq %99, %c-1_bv1 : !smt.bv<1>
    %104 = smt.ite %103, %c15_bv5, %97 : !smt.bv<5>
    %105 = smt.eq %arg2, %c15_bv5 : !smt.bv<5>
    %106 = smt.ite %105, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %107 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %108 = smt.eq %106, %c-1_bv1 : !smt.bv<1>
    %109 = smt.ite %108, %107, %102 : !smt.bv<16>
    %110 = smt.eq %106, %c-1_bv1 : !smt.bv<1>
    %111 = smt.ite %110, %c-16_bv5, %104 : !smt.bv<5>
    %112 = smt.eq %arg2, %c-16_bv5 : !smt.bv<5>
    %113 = smt.ite %112, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %114 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %115 = smt.eq %113, %c-1_bv1 : !smt.bv<1>
    %116 = smt.ite %115, %114, %109 : !smt.bv<16>
    %117 = smt.eq %113, %c-1_bv1 : !smt.bv<1>
    %118 = smt.ite %117, %c-15_bv5, %111 : !smt.bv<5>
    %119 = smt.eq %arg2, %c-15_bv5 : !smt.bv<5>
    %120 = smt.ite %119, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %121 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %122 = smt.eq %120, %c-1_bv1 : !smt.bv<1>
    %123 = smt.ite %122, %121, %116 : !smt.bv<16>
    %124 = smt.eq %120, %c-1_bv1 : !smt.bv<1>
    %125 = smt.ite %124, %c-14_bv5, %118 : !smt.bv<5>
    %126 = smt.eq %arg2, %c-14_bv5 : !smt.bv<5>
    %127 = smt.ite %126, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %128 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %129 = smt.eq %127, %c-1_bv1 : !smt.bv<1>
    %130 = smt.ite %129, %128, %123 : !smt.bv<16>
    %131 = smt.eq %127, %c-1_bv1 : !smt.bv<1>
    %132 = smt.ite %131, %c-13_bv5, %125 : !smt.bv<5>
    %133 = smt.eq %arg2, %c-13_bv5 : !smt.bv<5>
    %134 = smt.ite %133, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %135 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %136 = smt.eq %134, %c-1_bv1 : !smt.bv<1>
    %137 = smt.ite %136, %135, %130 : !smt.bv<16>
    %138 = smt.eq %134, %c-1_bv1 : !smt.bv<1>
    %139 = smt.ite %138, %c-12_bv5, %132 : !smt.bv<5>
    %140 = smt.eq %arg2, %c-12_bv5 : !smt.bv<5>
    %141 = smt.ite %140, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %142 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %143 = smt.eq %141, %c-1_bv1 : !smt.bv<1>
    %144 = smt.ite %143, %142, %137 : !smt.bv<16>
    %145 = smt.eq %141, %c-1_bv1 : !smt.bv<1>
    %146 = smt.ite %145, %c-11_bv5, %139 : !smt.bv<5>
    %147 = smt.eq %arg2, %c-11_bv5 : !smt.bv<5>
    %148 = smt.ite %147, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %149 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %150 = smt.eq %148, %c-1_bv1 : !smt.bv<1>
    %151 = smt.ite %150, %149, %144 : !smt.bv<16>
    %152 = smt.eq %148, %c-1_bv1 : !smt.bv<1>
    %153 = smt.ite %152, %c-10_bv5, %146 : !smt.bv<5>
    %154 = smt.eq %arg2, %c-10_bv5 : !smt.bv<5>
    %155 = smt.ite %154, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %156 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %157 = smt.eq %155, %c-1_bv1 : !smt.bv<1>
    %158 = smt.ite %157, %156, %151 : !smt.bv<16>
    %159 = smt.eq %155, %c-1_bv1 : !smt.bv<1>
    %160 = smt.ite %159, %c-9_bv5, %153 : !smt.bv<5>
    %161 = smt.eq %arg2, %c-9_bv5 : !smt.bv<5>
    %162 = smt.ite %161, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %163 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %164 = smt.eq %162, %c-1_bv1 : !smt.bv<1>
    %165 = smt.ite %164, %163, %158 : !smt.bv<16>
    %166 = smt.eq %162, %c-1_bv1 : !smt.bv<1>
    %167 = smt.ite %166, %c-8_bv5, %160 : !smt.bv<5>
    %168 = smt.eq %arg2, %c-8_bv5 : !smt.bv<5>
    %169 = smt.ite %168, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %170 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %171 = smt.eq %169, %c-1_bv1 : !smt.bv<1>
    %172 = smt.ite %171, %170, %165 : !smt.bv<16>
    %173 = smt.eq %169, %c-1_bv1 : !smt.bv<1>
    %174 = smt.ite %173, %c-7_bv5, %167 : !smt.bv<5>
    %175 = smt.eq %arg2, %c-7_bv5 : !smt.bv<5>
    %176 = smt.ite %175, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %177 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %178 = smt.eq %176, %c-1_bv1 : !smt.bv<1>
    %179 = smt.ite %178, %177, %172 : !smt.bv<16>
    %180 = smt.eq %176, %c-1_bv1 : !smt.bv<1>
    %181 = smt.ite %180, %c-6_bv5, %174 : !smt.bv<5>
    %182 = smt.eq %arg2, %c-6_bv5 : !smt.bv<5>
    %183 = smt.ite %182, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %184 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %185 = smt.eq %183, %c-1_bv1 : !smt.bv<1>
    %186 = smt.ite %185, %184, %179 : !smt.bv<16>
    %187 = smt.eq %183, %c-1_bv1 : !smt.bv<1>
    %188 = smt.ite %187, %c-5_bv5, %181 : !smt.bv<5>
    %189 = smt.eq %arg2, %c-5_bv5 : !smt.bv<5>
    %190 = smt.ite %189, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %191 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %192 = smt.eq %190, %c-1_bv1 : !smt.bv<1>
    %193 = smt.ite %192, %191, %186 : !smt.bv<16>
    %194 = smt.eq %190, %c-1_bv1 : !smt.bv<1>
    %195 = smt.ite %194, %c-4_bv5, %188 : !smt.bv<5>
    %196 = smt.eq %arg2, %c-4_bv5 : !smt.bv<5>
    %197 = smt.ite %196, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %198 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %199 = smt.eq %197, %c-1_bv1 : !smt.bv<1>
    %200 = smt.ite %199, %198, %193 : !smt.bv<16>
    %201 = smt.eq %197, %c-1_bv1 : !smt.bv<1>
    %202 = smt.ite %201, %c-3_bv5, %195 : !smt.bv<5>
    %203 = smt.eq %arg2, %c-3_bv5 : !smt.bv<5>
    %204 = smt.ite %203, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %205 = smt.bv.add %arg3, %c1_bv16 : !smt.bv<16>
    %206 = smt.eq %204, %c-1_bv1 : !smt.bv<1>
    %207 = smt.ite %206, %205, %200 : !smt.bv<16>
    %208 = smt.eq %204, %c-1_bv1 : !smt.bv<1>
    %209 = smt.ite %208, %c-2_bv5, %202 : !smt.bv<5>
    %210 = smt.eq %arg2, %c-2_bv5 : !smt.bv<5>
    %211 = smt.ite %210, %c-1_bv1, %c0_bv1 : !smt.bv<1>
    %212 = smt.eq %211, %c-1_bv1 : !smt.bv<1>
    %213 = smt.ite %212, %c-2_bv5, %209 : !smt.bv<5>
    %214 = smt.bv.add %arg4, %c1_bv8 : !smt.bv<8>
    %215 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
    %216 = smt.ite %215, %c0_bv5, %213 : !smt.bv<5>
    %217 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
    %218 = smt.ite %217, %c0_bv16, %207 : !smt.bv<16>
    return %216, %218, %214 : !smt.bv<5>, !smt.bv<16>, !smt.bv<8>
  }
}

