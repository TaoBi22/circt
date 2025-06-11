module {
  llvm.func @printf(!llvm.ptr, ...)
  func.func @fsm10() {
    %0 = llvm.mlir.addressof @resultString_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @resultString_0 : !llvm.ptr
    %2 = smt.solver() : () -> i1 {
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %true = arith.constant true
      %false = arith.constant false
      %c20_i32 = arith.constant 20 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %4 = func.call @bmc_init() : () -> !smt.bv<1>
      smt.push 1
      %5 = smt.declare_fun : !smt.bv<1>
      %6 = smt.declare_fun : !smt.bv<4>
      %7 = smt.declare_fun : !smt.bv<16>
      %8 = smt.declare_fun : !smt.bv<32>
      %9:6 = scf.for %arg0 = %c0_i32 to %c20_i32 step %c1_i32 iter_args(%arg1 = %4, %arg2 = %5, %arg3 = %6, %arg4 = %7, %arg5 = %8, %arg6 = %false) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<4>, !smt.bv<16>, !smt.bv<32>, i1)  : i32 {
        smt.pop 1
        smt.push 1
        %11:3 = func.call @bmc_circuit(%arg1, %arg2, %arg3, %arg4, %arg5) : (!smt.bv<1>, !smt.bv<1>, !smt.bv<4>, !smt.bv<16>, !smt.bv<32>) -> (!smt.bv<4>, !smt.bv<16>, !smt.bv<32>)
        %12 = smt.check sat {
          smt.yield %true : i1
        } unknown {
          smt.yield %true : i1
        } unsat {
          smt.yield %false : i1
        } -> i1
%ss = llvm.mlir.addressof @satString : !llvm.ptr
%us = llvm.mlir.addressof @unsatString : !llvm.ptr
%string = llvm.select %12, %ss, %us : i1, !llvm.ptr
%printf = llvm.mlir.addressof @printf : !llvm.func<void (ptr, ...)>
        %13 = arith.ori %12, %arg6 : i1
        %14 = func.call @bmc_loop(%arg1) : (!smt.bv<1>) -> !smt.bv<1>
        %15 = smt.declare_fun : !smt.bv<1>
        %16 = smt.bv.not %arg1 : !smt.bv<1>
        %17 = smt.bv.and %16, %14 : !smt.bv<1>
        %18 = smt.eq %17, %c-1_bv1 : !smt.bv<1>
        %19 = smt.ite %18, %11#0, %arg3 : !smt.bv<4>
        %20 = smt.ite %18, %11#1, %arg4 : !smt.bv<16>
        %21 = smt.ite %18, %11#2, %arg5 : !smt.bv<32>
        scf.yield %14, %15, %19, %20, %21, %13 : !smt.bv<1>, !smt.bv<1>, !smt.bv<4>, !smt.bv<16>, !smt.bv<32>, i1
      }
      %10 = arith.xori %9#5, %true : i1
      smt.yield %10 : i1
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
%tvclause_0 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__0(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_0
%tvclause_1 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__1(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_1
%tvclause_2 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__2(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_2
%tvclause_3 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__3(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_3
%tvclause_4 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__4(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_4
%tvclause_5 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__5(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_5
%tvclause_6 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__6(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_6
%tvclause_7 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__7(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_7
%tvclause_8 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__8(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_8
%tvclause_9 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__9(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_9
%tvclause_10 = smt.forall{
^bb0(%var_0: !smt.bv<16>, %rtlTime: !smt.bv<32>):
%apply = smt.apply_func %obsF__10(%var_0, %rtlTime) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>
%antecedent = smt.and %apply, %rightTime : !smt.bool
%var_0_eq = smt.eq %var_0, %x0 : !smt.bv<16>
%consequent = smt.and %var_0_eq : !smt.bool
%impl = smt.implies %antecedent, %consequent : !smt.bool
smt.yield %impl : !smt.bool
}
smt.assert %tvclause_10
    return %73, %67, %74 : !smt.bv<4>, !smt.bv<16>, !smt.bv<32>
  }
}

