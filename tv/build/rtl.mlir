module {
  hw.module @fsm10(in %in0 : i1, in %in1 : i1, in %in2 : i1, in %in3 : i16, out out0 : i1, out out1 : i1, out out2 : i1, in %clk : !seq.clock, in %rst : i1) {
    %c0_i2 = hw.constant 0 : i2
    %c1_i2 = hw.constant 1 : i2
    %c-2_i2 = hw.constant -2 : i2
    %0 = seq.initial() {
      %c0_i2_5 = hw.constant 0 : i2
      seq.yield %c0_i2_5 : i2
    } : () -> !seq.immutable<i2>
    %state_reg = seq.compreg sym @state_reg %72, %clk reset %rst, %c0_i2 initial %0 : i2  
    %c0_i16 = hw.constant 0 : i16
    %1 = seq.initial() {
      %c0_i16_5 = hw.constant 0 : i16
      seq.yield %c0_i16_5 : i16
    } : () -> !seq.immutable<i16>
    %ctr_slice_idx = seq.compreg sym @ctr_slice_idx %64, %clk reset %rst, %c0_i16 initial %1 : i16  
    %false = hw.constant false
    %2 = seq.initial() {
      %false_5 = hw.constant false
      seq.yield %false_5 : i1
    } : () -> !seq.immutable<i1>
    %ctr_carry = seq.compreg sym @ctr_carry %63, %clk reset %rst, %false initial %2 : i1  
    %false_0 = hw.constant false
    %3 = seq.initial() {
      %false_5 = hw.constant false
      seq.yield %false_5 : i1
    } : () -> !seq.immutable<i1>
    %error_o = seq.compreg sym @error_o %65, %clk reset %rst, %false_0 initial %3 : i1  
    %c0_i16_1 = hw.constant 0 : i16
    %false_2 = hw.constant false
    %c-1_i16 = hw.constant -1 : i16
    %true = hw.constant true
    %false_3 = hw.constant false
    %4 = comb.icmp eq %state_reg, %c0_i2 : i2
    %5 = comb.icmp eq %state_reg, %c0_i2 : i2
    %6 = comb.icmp eq %state_reg, %c0_i2 : i2
    %7 = comb.or %in1, %in2 : i1
    %8 = comb.icmp eq %in0, %false_2 : i1
    %9 = comb.and %7, %8 : i1
    %10 = comb.icmp eq %state_reg, %c0_i2 : i2
    %11 = comb.icmp eq %in0, %false_2 : i1
    %12 = comb.icmp eq %in1, %false_2 : i1
    %13 = comb.icmp eq %in2, %false_2 : i1
    %14 = comb.and %11, %12 : i1
    %15 = comb.and %14, %13 : i1
    %16 = comb.icmp eq %state_reg, %c0_i2 : i2
    %17 = comb.mux %16, %c0_i2, %state_reg : i2
    %18 = comb.mux %15, %c0_i2, %c0_i2 : i2
    %19 = comb.and %15, %10 : i1
    %20 = comb.mux %19, %false_3, %error_o : i1
    %21 = comb.mux %10, %18, %17 : i2
    %22 = comb.mux %7, %c-2_i2, %18 : i2
    %23 = comb.and %7, %6 : i1
    %24 = comb.mux %23, %true, %20 : i1
    %25 = comb.mux %6, %22, %21 : i2
    %26 = comb.mux %in0, %c1_i2, %22 : i2
    %27 = comb.and %in0, %5 : i1
    %28 = comb.mux %27, %c0_i16_1, %ctr_slice_idx : i16
    %29 = comb.mux %27, %true, %ctr_carry : i1
    %30 = comb.mux %27, %false_3, %24 : i1
    %31 = comb.mux %5, %26, %25 : i2
    %32 = comb.icmp eq %state_reg, %c1_i2 : i2
    %33 = comb.mux %32, %false_3, %true : i1
    %34 = comb.mux %32, %error_o, %error_o : i1
    %35 = comb.mux %32, %true, %false_3 : i1
    %36 = comb.icmp eq %state_reg, %c1_i2 : i2
    %37 = comb.icmp slt %in3, %c0_i16_1 : i16
    %c1_i16 = hw.constant 1 : i16
    %38 = comb.add %ctr_slice_idx, %c1_i16 : i16
    %39 = comb.icmp eq %c-1_i16, %ctr_slice_idx : i16
    %40 = comb.icmp eq %state_reg, %c1_i2 : i2
    %41 = comb.icmp slt %in3, %c0_i16_1 : i16
    %c1_i16_4 = hw.constant 1 : i16
    %42 = comb.add %ctr_slice_idx, %c1_i16_4 : i16
    %43 = comb.icmp ne %c-1_i16, %ctr_slice_idx : i16
    %44 = comb.or %in1, %in2 : i1
    %45 = comb.xor %44, %true : i1
    %46 = comb.and %43, %45 : i1
    %47 = comb.icmp eq %state_reg, %c1_i2 : i2
    %48 = comb.or %in1, %in2 : i1
    %49 = comb.icmp eq %state_reg, %c1_i2 : i2
    %50 = comb.mux %49, %c1_i2, %31 : i2
    %51 = comb.mux %48, %c-2_i2, %c1_i2 : i2
    %52 = comb.and %48, %47 : i1
    %53 = comb.mux %52, %true, %30 : i1
    %54 = comb.mux %47, %51, %50 : i2
    %55 = comb.mux %46, %c1_i2, %51 : i2
    %56 = comb.and %46, %40 : i1
    %57 = comb.mux %56, %41, %29 : i1
    %58 = comb.mux %56, %42, %28 : i16
    %59 = comb.mux %56, %false_3, %53 : i1
    %60 = comb.mux %40, %55, %54 : i2
    %61 = comb.mux %39, %c0_i2, %55 : i2
    %62 = comb.and %39, %36 : i1
    %63 = comb.mux %62, %37, %57 : i1
    %64 = comb.mux %62, %38, %58 : i16
    %65 = comb.mux %62, %false_3, %59 : i1
    %66 = comb.mux %36, %61, %60 : i2
    %67 = comb.icmp eq %state_reg, %c-2_i2 : i2
    %68 = comb.mux %67, %error_o, %33 : i1
    %69 = comb.mux %67, %true, %34 : i1
    %70 = comb.mux %67, %false_3, %35 : i1
    %71 = comb.icmp eq %state_reg, %c-2_i2 : i2
    %72 = comb.mux %71, %c-2_i2, %66 : i2
    %timer_init = seq.initial() {
%c0_i8_0_in = hw.constant 0 : i8
seq.yield %c0_i8_0_in : i8
} : () -> !seq.immutable<i8>
%mySpecialConstant = hw.constant 1 : i8
%time_reg = seq.compreg sym @time_reg %added, %clk initial %timer_init : i8
%added = comb.add %time_reg, %mySpecialConstant : i8
    hw.output %68, %69, %70 : i1, i1, i1
}}