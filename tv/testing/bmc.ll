; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@ctx = internal global ptr null, align 8
@solver = internal global ptr null, align 8
@resultString_0 = private constant [35 x i8] c"Bound reached with no violations!\0A\00"
@resultString_1 = private constant [28 x i8] c"Assertion can be violated!\0A\00"
@str = internal constant [5 x i8] c"F__0\00"
@str_0 = internal constant [5 x i8] c"F__1\00"
@str_1 = internal constant [5 x i8] c"F__2\00"
@str_2 = internal constant [5 x i8] c"F__3\00"
@str_3 = internal constant [5 x i8] c"F__4\00"
@str_4 = internal constant [5 x i8] c"F__5\00"
@str_5 = internal constant [5 x i8] c"F__6\00"
@str_6 = internal constant [5 x i8] c"F__7\00"
@str_7 = internal constant [5 x i8] c"F__8\00"
@str_8 = internal constant [5 x i8] c"F__9\00"
@str_9 = internal constant [6 x i8] c"F__10\00"

declare ptr @Z3_mk_false(ptr)

declare ptr @Z3_mk_distinct(ptr, i32, ptr)

declare ptr @Z3_mk_and(ptr, i32, ptr)

declare ptr @Z3_mk_true(ptr)

declare void @Z3_solver_assert(ptr, ptr, ptr)

declare ptr @Z3_mk_implies(ptr, ptr, ptr)

declare ptr @Z3_mk_app(ptr, ptr, i32, ptr)

declare ptr @Z3_mk_forall_const(ptr, i32, i32, ptr, i32, ptr, ptr)

declare ptr @Z3_mk_fresh_func_decl(ptr, ptr, i32, ptr, ptr)

declare ptr @Z3_mk_bool_sort(ptr)

declare ptr @Z3_mk_bvadd(ptr, ptr, ptr)

declare ptr @Z3_mk_ite(ptr, ptr, ptr, ptr)

declare ptr @Z3_mk_eq(ptr, ptr, ptr)

declare ptr @Z3_mk_bvxor(ptr, ptr, ptr)

declare i32 @Z3_solver_check(ptr, ptr)

declare ptr @Z3_mk_fresh_const(ptr, ptr, ptr)

declare ptr @Z3_mk_unsigned_int64(ptr, i64, ptr)

declare ptr @Z3_mk_bv_sort(ptr, i32)

declare void @Z3_del_context(ptr)

declare void @Z3_solver_dec_ref(ptr, ptr)

declare void @Z3_solver_inc_ref(ptr, ptr)

declare ptr @Z3_mk_solver(ptr)

declare void @Z3_del_config(ptr)

declare ptr @Z3_mk_context(ptr)

declare ptr @Z3_mk_config()

declare void @printf(ptr, ...)

define void @fsm10() {
  %1 = call ptr @Z3_mk_config()
  %2 = call ptr @Z3_mk_context(ptr %1)
  store ptr %2, ptr @ctx, align 8
  call void @Z3_del_config(ptr %1)
  %3 = call ptr @Z3_mk_solver(ptr %2)
  call void @Z3_solver_inc_ref(ptr %2, ptr %3)
  store ptr %3, ptr @solver, align 8
  %4 = call i1 @solver_0()
  call void @Z3_solver_dec_ref(ptr %2, ptr %3)
  call void @Z3_del_context(ptr %2)
  %5 = select i1 %4, ptr @resultString_0, ptr @resultString_1
  call void (ptr, ...) @printf(ptr %5)
  ret void
}

define ptr @bmc_init() {
  %1 = load ptr, ptr @ctx, align 8
  %2 = call ptr @Z3_mk_bv_sort(ptr %1, i32 1)
  %3 = call ptr @Z3_mk_unsigned_int64(ptr %1, i64 0, ptr %2)
  ret ptr %3
}

define ptr @bmc_loop(ptr %0) {
  %2 = load ptr, ptr @ctx, align 8
  %3 = call ptr @Z3_mk_bv_sort(ptr %2, i32 1)
  %4 = call ptr @Z3_mk_unsigned_int64(ptr %2, i64 1, ptr %3)
  %5 = call ptr @Z3_mk_bvxor(ptr %2, ptr %0, ptr %4)
  ret ptr %5
}

define { ptr, ptr, ptr } @bmc_circuit(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) {
  %6 = load ptr, ptr @solver, align 8
  %7 = load ptr, ptr @ctx, align 8
  %8 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %9 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %8)
  %10 = call ptr @Z3_mk_bv_sort(ptr %7, i32 1)
  %11 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %10)
  %12 = call ptr @Z3_mk_bv_sort(ptr %7, i32 1)
  %13 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 0, ptr %12)
  %14 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %15 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %14)
  %16 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %17 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 10, ptr %16)
  %18 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %19 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 9, ptr %18)
  %20 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %21 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 8, ptr %20)
  %22 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %23 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 7, ptr %22)
  %24 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %25 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 6, ptr %24)
  %26 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %27 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 5, ptr %26)
  %28 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %29 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 4, ptr %28)
  %30 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %31 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 3, ptr %30)
  %32 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %33 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 2, ptr %32)
  %34 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %35 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %34)
  %36 = call ptr @Z3_mk_bv_sort(ptr %7, i32 4)
  %37 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 0, ptr %36)
  %38 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %37)
  %39 = call ptr @Z3_mk_ite(ptr %7, ptr %38, ptr %11, ptr %13)
  %40 = call ptr @Z3_mk_bvadd(ptr %7, ptr %3, ptr %15)
  %41 = call ptr @Z3_mk_eq(ptr %7, ptr %39, ptr %11)
  %42 = call ptr @Z3_mk_ite(ptr %7, ptr %41, ptr %40, ptr %3)
  %43 = call ptr @Z3_mk_eq(ptr %7, ptr %39, ptr %11)
  %44 = call ptr @Z3_mk_ite(ptr %7, ptr %43, ptr %35, ptr %2)
  %45 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %35)
  %46 = call ptr @Z3_mk_ite(ptr %7, ptr %45, ptr %11, ptr %13)
  %47 = call ptr @Z3_mk_bvadd(ptr %7, ptr %3, ptr %15)
  %48 = call ptr @Z3_mk_eq(ptr %7, ptr %46, ptr %11)
  %49 = call ptr @Z3_mk_ite(ptr %7, ptr %48, ptr %47, ptr %42)
  %50 = call ptr @Z3_mk_eq(ptr %7, ptr %46, ptr %11)
  %51 = call ptr @Z3_mk_ite(ptr %7, ptr %50, ptr %33, ptr %44)
  %52 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %33)
  %53 = call ptr @Z3_mk_ite(ptr %7, ptr %52, ptr %11, ptr %13)
  %54 = call ptr @Z3_mk_bvadd(ptr %7, ptr %3, ptr %15)
  %55 = call ptr @Z3_mk_eq(ptr %7, ptr %53, ptr %11)
  %56 = call ptr @Z3_mk_ite(ptr %7, ptr %55, ptr %54, ptr %49)
  %57 = call ptr @Z3_mk_eq(ptr %7, ptr %53, ptr %11)
  %58 = call ptr @Z3_mk_ite(ptr %7, ptr %57, ptr %31, ptr %51)
  %59 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %31)
  %60 = call ptr @Z3_mk_ite(ptr %7, ptr %59, ptr %11, ptr %13)
  %61 = call ptr @Z3_mk_bvadd(ptr %7, ptr %3, ptr %15)
  %62 = call ptr @Z3_mk_eq(ptr %7, ptr %60, ptr %11)
  %63 = call ptr @Z3_mk_ite(ptr %7, ptr %62, ptr %61, ptr %56)
  %64 = call ptr @Z3_mk_eq(ptr %7, ptr %60, ptr %11)
  %65 = call ptr @Z3_mk_ite(ptr %7, ptr %64, ptr %29, ptr %58)
  %66 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %29)
  %67 = call ptr @Z3_mk_ite(ptr %7, ptr %66, ptr %11, ptr %13)
  %68 = call ptr @Z3_mk_bvadd(ptr %7, ptr %3, ptr %15)
  %69 = call ptr @Z3_mk_eq(ptr %7, ptr %67, ptr %11)
  %70 = call ptr @Z3_mk_ite(ptr %7, ptr %69, ptr %68, ptr %63)
  %71 = call ptr @Z3_mk_eq(ptr %7, ptr %67, ptr %11)
  %72 = call ptr @Z3_mk_ite(ptr %7, ptr %71, ptr %27, ptr %65)
  %73 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %27)
  %74 = call ptr @Z3_mk_ite(ptr %7, ptr %73, ptr %11, ptr %13)
  %75 = call ptr @Z3_mk_bvadd(ptr %7, ptr %3, ptr %15)
  %76 = call ptr @Z3_mk_eq(ptr %7, ptr %74, ptr %11)
  %77 = call ptr @Z3_mk_ite(ptr %7, ptr %76, ptr %75, ptr %70)
  %78 = call ptr @Z3_mk_eq(ptr %7, ptr %74, ptr %11)
  %79 = call ptr @Z3_mk_ite(ptr %7, ptr %78, ptr %25, ptr %72)
  %80 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %25)
  %81 = call ptr @Z3_mk_ite(ptr %7, ptr %80, ptr %11, ptr %13)
  %82 = call ptr @Z3_mk_bvadd(ptr %7, ptr %3, ptr %15)
  %83 = call ptr @Z3_mk_eq(ptr %7, ptr %81, ptr %11)
  %84 = call ptr @Z3_mk_ite(ptr %7, ptr %83, ptr %82, ptr %77)
  %85 = call ptr @Z3_mk_eq(ptr %7, ptr %81, ptr %11)
  %86 = call ptr @Z3_mk_ite(ptr %7, ptr %85, ptr %23, ptr %79)
  %87 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %23)
  %88 = call ptr @Z3_mk_ite(ptr %7, ptr %87, ptr %11, ptr %13)
  %89 = call ptr @Z3_mk_bvadd(ptr %7, ptr %3, ptr %15)
  %90 = call ptr @Z3_mk_eq(ptr %7, ptr %88, ptr %11)
  %91 = call ptr @Z3_mk_ite(ptr %7, ptr %90, ptr %89, ptr %84)
  %92 = call ptr @Z3_mk_eq(ptr %7, ptr %88, ptr %11)
  %93 = call ptr @Z3_mk_ite(ptr %7, ptr %92, ptr %21, ptr %86)
  %94 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %21)
  %95 = call ptr @Z3_mk_ite(ptr %7, ptr %94, ptr %11, ptr %13)
  %96 = call ptr @Z3_mk_bvadd(ptr %7, ptr %3, ptr %15)
  %97 = call ptr @Z3_mk_eq(ptr %7, ptr %95, ptr %11)
  %98 = call ptr @Z3_mk_ite(ptr %7, ptr %97, ptr %96, ptr %91)
  %99 = call ptr @Z3_mk_eq(ptr %7, ptr %95, ptr %11)
  %100 = call ptr @Z3_mk_ite(ptr %7, ptr %99, ptr %19, ptr %93)
  %101 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %19)
  %102 = call ptr @Z3_mk_ite(ptr %7, ptr %101, ptr %11, ptr %13)
  %103 = call ptr @Z3_mk_bvadd(ptr %7, ptr %3, ptr %15)
  %104 = call ptr @Z3_mk_eq(ptr %7, ptr %102, ptr %11)
  %105 = call ptr @Z3_mk_ite(ptr %7, ptr %104, ptr %103, ptr %98)
  %106 = call ptr @Z3_mk_eq(ptr %7, ptr %102, ptr %11)
  %107 = call ptr @Z3_mk_ite(ptr %7, ptr %106, ptr %17, ptr %100)
  %108 = call ptr @Z3_mk_eq(ptr %7, ptr %2, ptr %17)
  %109 = call ptr @Z3_mk_ite(ptr %7, ptr %108, ptr %11, ptr %13)
  %110 = call ptr @Z3_mk_eq(ptr %7, ptr %109, ptr %11)
  %111 = call ptr @Z3_mk_ite(ptr %7, ptr %110, ptr %17, ptr %107)
  %112 = call ptr @Z3_mk_bvadd(ptr %7, ptr %4, ptr %9)
  %113 = call ptr @Z3_mk_bool_sort(ptr %7)
  %114 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %115 = insertvalue [2 x ptr] undef, ptr %114, 0
  %116 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %117 = insertvalue [2 x ptr] %115, ptr %116, 1
  %118 = alloca [2 x ptr], align 8
  store [2 x ptr] %117, ptr %118, align 8
  %119 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str, i32 2, ptr %118, ptr %113)
  %120 = call ptr @Z3_mk_bool_sort(ptr %7)
  %121 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %122 = insertvalue [2 x ptr] undef, ptr %121, 0
  %123 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %124 = insertvalue [2 x ptr] %122, ptr %123, 1
  %125 = alloca [2 x ptr], align 8
  store [2 x ptr] %124, ptr %125, align 8
  %126 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str_0, i32 2, ptr %125, ptr %120)
  %127 = call ptr @Z3_mk_bool_sort(ptr %7)
  %128 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %129 = insertvalue [2 x ptr] undef, ptr %128, 0
  %130 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %131 = insertvalue [2 x ptr] %129, ptr %130, 1
  %132 = alloca [2 x ptr], align 8
  store [2 x ptr] %131, ptr %132, align 8
  %133 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str_1, i32 2, ptr %132, ptr %127)
  %134 = call ptr @Z3_mk_bool_sort(ptr %7)
  %135 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %136 = insertvalue [2 x ptr] undef, ptr %135, 0
  %137 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %138 = insertvalue [2 x ptr] %136, ptr %137, 1
  %139 = alloca [2 x ptr], align 8
  store [2 x ptr] %138, ptr %139, align 8
  %140 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str_2, i32 2, ptr %139, ptr %134)
  %141 = call ptr @Z3_mk_bool_sort(ptr %7)
  %142 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %143 = insertvalue [2 x ptr] undef, ptr %142, 0
  %144 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %145 = insertvalue [2 x ptr] %143, ptr %144, 1
  %146 = alloca [2 x ptr], align 8
  store [2 x ptr] %145, ptr %146, align 8
  %147 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str_3, i32 2, ptr %146, ptr %141)
  %148 = call ptr @Z3_mk_bool_sort(ptr %7)
  %149 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %150 = insertvalue [2 x ptr] undef, ptr %149, 0
  %151 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %152 = insertvalue [2 x ptr] %150, ptr %151, 1
  %153 = alloca [2 x ptr], align 8
  store [2 x ptr] %152, ptr %153, align 8
  %154 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str_4, i32 2, ptr %153, ptr %148)
  %155 = call ptr @Z3_mk_bool_sort(ptr %7)
  %156 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %157 = insertvalue [2 x ptr] undef, ptr %156, 0
  %158 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %159 = insertvalue [2 x ptr] %157, ptr %158, 1
  %160 = alloca [2 x ptr], align 8
  store [2 x ptr] %159, ptr %160, align 8
  %161 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str_5, i32 2, ptr %160, ptr %155)
  %162 = call ptr @Z3_mk_bool_sort(ptr %7)
  %163 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %164 = insertvalue [2 x ptr] undef, ptr %163, 0
  %165 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %166 = insertvalue [2 x ptr] %164, ptr %165, 1
  %167 = alloca [2 x ptr], align 8
  store [2 x ptr] %166, ptr %167, align 8
  %168 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str_6, i32 2, ptr %167, ptr %162)
  %169 = call ptr @Z3_mk_bool_sort(ptr %7)
  %170 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %171 = insertvalue [2 x ptr] undef, ptr %170, 0
  %172 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %173 = insertvalue [2 x ptr] %171, ptr %172, 1
  %174 = alloca [2 x ptr], align 8
  store [2 x ptr] %173, ptr %174, align 8
  %175 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str_7, i32 2, ptr %174, ptr %169)
  %176 = call ptr @Z3_mk_bool_sort(ptr %7)
  %177 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %178 = insertvalue [2 x ptr] undef, ptr %177, 0
  %179 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %180 = insertvalue [2 x ptr] %178, ptr %179, 1
  %181 = alloca [2 x ptr], align 8
  store [2 x ptr] %180, ptr %181, align 8
  %182 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str_8, i32 2, ptr %181, ptr %176)
  %183 = call ptr @Z3_mk_bool_sort(ptr %7)
  %184 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %185 = insertvalue [2 x ptr] undef, ptr %184, 0
  %186 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %187 = insertvalue [2 x ptr] %185, ptr %186, 1
  %188 = alloca [2 x ptr], align 8
  store [2 x ptr] %187, ptr %188, align 8
  %189 = call ptr @Z3_mk_fresh_func_decl(ptr %7, ptr @str_9, i32 2, ptr %188, ptr %183)
  %190 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %191 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %190)
  %192 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %193 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %192)
  %194 = alloca [2 x ptr], align 8
  %195 = insertvalue [2 x ptr] undef, ptr %191, 0
  %196 = insertvalue [2 x ptr] %195, ptr %193, 1
  store [2 x ptr] %196, ptr %194, align 8
  %197 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %198 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 0, ptr %197)
  %199 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %200 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 0, ptr %199)
  %201 = call ptr @Z3_mk_eq(ptr %7, ptr %193, ptr %200)
  %202 = insertvalue [2 x ptr] undef, ptr %198, 0
  %203 = insertvalue [2 x ptr] %202, ptr %193, 1
  %204 = alloca [2 x ptr], align 8
  store [2 x ptr] %203, ptr %204, align 8
  %205 = call ptr @Z3_mk_app(ptr %7, ptr %119, i32 2, ptr %204)
  %206 = call ptr @Z3_mk_implies(ptr %7, ptr %201, ptr %205)
  %207 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %194, i32 0, ptr null, ptr %206)
  call void @Z3_solver_assert(ptr %7, ptr %6, ptr %207)
  %208 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %209 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %208)
  %210 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %211 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %210)
  %212 = alloca [2 x ptr], align 8
  %213 = insertvalue [2 x ptr] undef, ptr %209, 0
  %214 = insertvalue [2 x ptr] %213, ptr %211, 1
  store [2 x ptr] %214, ptr %212, align 8
  %215 = insertvalue [2 x ptr] undef, ptr %209, 0
  %216 = insertvalue [2 x ptr] %215, ptr %211, 1
  %217 = alloca [2 x ptr], align 8
  store [2 x ptr] %216, ptr %217, align 8
  %218 = call ptr @Z3_mk_app(ptr %7, ptr %119, i32 2, ptr %217)
  %219 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %220 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %219)
  %221 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %222 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %221)
  %223 = call ptr @Z3_mk_eq(ptr %7, ptr %209, ptr %222)
  %224 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %225 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %224)
  %226 = call ptr @Z3_mk_eq(ptr %7, ptr %220, ptr %225)
  %227 = call ptr @Z3_mk_bvadd(ptr %7, ptr %209, ptr %220)
  %228 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %229 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %228)
  %230 = call ptr @Z3_mk_bvadd(ptr %7, ptr %211, ptr %229)
  %231 = insertvalue [2 x ptr] undef, ptr %227, 0
  %232 = insertvalue [2 x ptr] %231, ptr %230, 1
  %233 = alloca [2 x ptr], align 8
  store [2 x ptr] %232, ptr %233, align 8
  %234 = call ptr @Z3_mk_app(ptr %7, ptr %126, i32 2, ptr %233)
  %235 = call ptr @Z3_mk_true(ptr %7)
  %236 = alloca [2 x ptr], align 8
  %237 = insertvalue [2 x ptr] undef, ptr %218, 0
  %238 = insertvalue [2 x ptr] %237, ptr %235, 1
  store [2 x ptr] %238, ptr %236, align 8
  %239 = call ptr @Z3_mk_and(ptr %7, i32 2, ptr %236)
  %240 = call ptr @Z3_mk_implies(ptr %7, ptr %239, ptr %234)
  %241 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %212, i32 0, ptr null, ptr %240)
  call void @Z3_solver_assert(ptr %7, ptr %6, ptr %241)
  %242 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %243 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %242)
  %244 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %245 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %244)
  %246 = alloca [2 x ptr], align 8
  %247 = insertvalue [2 x ptr] undef, ptr %243, 0
  %248 = insertvalue [2 x ptr] %247, ptr %245, 1
  store [2 x ptr] %248, ptr %246, align 8
  %249 = insertvalue [2 x ptr] undef, ptr %243, 0
  %250 = insertvalue [2 x ptr] %249, ptr %245, 1
  %251 = alloca [2 x ptr], align 8
  store [2 x ptr] %250, ptr %251, align 8
  %252 = call ptr @Z3_mk_app(ptr %7, ptr %126, i32 2, ptr %251)
  %253 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %254 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %253)
  %255 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %256 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %255)
  %257 = call ptr @Z3_mk_eq(ptr %7, ptr %243, ptr %256)
  %258 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %259 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %258)
  %260 = call ptr @Z3_mk_eq(ptr %7, ptr %254, ptr %259)
  %261 = call ptr @Z3_mk_bvadd(ptr %7, ptr %243, ptr %254)
  %262 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %263 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %262)
  %264 = call ptr @Z3_mk_bvadd(ptr %7, ptr %245, ptr %263)
  %265 = insertvalue [2 x ptr] undef, ptr %261, 0
  %266 = insertvalue [2 x ptr] %265, ptr %264, 1
  %267 = alloca [2 x ptr], align 8
  store [2 x ptr] %266, ptr %267, align 8
  %268 = call ptr @Z3_mk_app(ptr %7, ptr %133, i32 2, ptr %267)
  %269 = call ptr @Z3_mk_true(ptr %7)
  %270 = alloca [2 x ptr], align 8
  %271 = insertvalue [2 x ptr] undef, ptr %252, 0
  %272 = insertvalue [2 x ptr] %271, ptr %269, 1
  store [2 x ptr] %272, ptr %270, align 8
  %273 = call ptr @Z3_mk_and(ptr %7, i32 2, ptr %270)
  %274 = call ptr @Z3_mk_implies(ptr %7, ptr %273, ptr %268)
  %275 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %246, i32 0, ptr null, ptr %274)
  call void @Z3_solver_assert(ptr %7, ptr %6, ptr %275)
  %276 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %277 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %276)
  %278 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %279 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %278)
  %280 = alloca [2 x ptr], align 8
  %281 = insertvalue [2 x ptr] undef, ptr %277, 0
  %282 = insertvalue [2 x ptr] %281, ptr %279, 1
  store [2 x ptr] %282, ptr %280, align 8
  %283 = insertvalue [2 x ptr] undef, ptr %277, 0
  %284 = insertvalue [2 x ptr] %283, ptr %279, 1
  %285 = alloca [2 x ptr], align 8
  store [2 x ptr] %284, ptr %285, align 8
  %286 = call ptr @Z3_mk_app(ptr %7, ptr %133, i32 2, ptr %285)
  %287 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %288 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %287)
  %289 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %290 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %289)
  %291 = call ptr @Z3_mk_eq(ptr %7, ptr %277, ptr %290)
  %292 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %293 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %292)
  %294 = call ptr @Z3_mk_eq(ptr %7, ptr %288, ptr %293)
  %295 = call ptr @Z3_mk_bvadd(ptr %7, ptr %277, ptr %288)
  %296 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %297 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %296)
  %298 = call ptr @Z3_mk_bvadd(ptr %7, ptr %279, ptr %297)
  %299 = insertvalue [2 x ptr] undef, ptr %295, 0
  %300 = insertvalue [2 x ptr] %299, ptr %298, 1
  %301 = alloca [2 x ptr], align 8
  store [2 x ptr] %300, ptr %301, align 8
  %302 = call ptr @Z3_mk_app(ptr %7, ptr %140, i32 2, ptr %301)
  %303 = call ptr @Z3_mk_true(ptr %7)
  %304 = alloca [2 x ptr], align 8
  %305 = insertvalue [2 x ptr] undef, ptr %286, 0
  %306 = insertvalue [2 x ptr] %305, ptr %303, 1
  store [2 x ptr] %306, ptr %304, align 8
  %307 = call ptr @Z3_mk_and(ptr %7, i32 2, ptr %304)
  %308 = call ptr @Z3_mk_implies(ptr %7, ptr %307, ptr %302)
  %309 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %280, i32 0, ptr null, ptr %308)
  call void @Z3_solver_assert(ptr %7, ptr %6, ptr %309)
  %310 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %311 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %310)
  %312 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %313 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %312)
  %314 = alloca [2 x ptr], align 8
  %315 = insertvalue [2 x ptr] undef, ptr %311, 0
  %316 = insertvalue [2 x ptr] %315, ptr %313, 1
  store [2 x ptr] %316, ptr %314, align 8
  %317 = insertvalue [2 x ptr] undef, ptr %311, 0
  %318 = insertvalue [2 x ptr] %317, ptr %313, 1
  %319 = alloca [2 x ptr], align 8
  store [2 x ptr] %318, ptr %319, align 8
  %320 = call ptr @Z3_mk_app(ptr %7, ptr %140, i32 2, ptr %319)
  %321 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %322 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %321)
  %323 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %324 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %323)
  %325 = call ptr @Z3_mk_eq(ptr %7, ptr %311, ptr %324)
  %326 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %327 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %326)
  %328 = call ptr @Z3_mk_eq(ptr %7, ptr %322, ptr %327)
  %329 = call ptr @Z3_mk_bvadd(ptr %7, ptr %311, ptr %322)
  %330 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %331 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %330)
  %332 = call ptr @Z3_mk_bvadd(ptr %7, ptr %313, ptr %331)
  %333 = insertvalue [2 x ptr] undef, ptr %329, 0
  %334 = insertvalue [2 x ptr] %333, ptr %332, 1
  %335 = alloca [2 x ptr], align 8
  store [2 x ptr] %334, ptr %335, align 8
  %336 = call ptr @Z3_mk_app(ptr %7, ptr %147, i32 2, ptr %335)
  %337 = call ptr @Z3_mk_true(ptr %7)
  %338 = alloca [2 x ptr], align 8
  %339 = insertvalue [2 x ptr] undef, ptr %320, 0
  %340 = insertvalue [2 x ptr] %339, ptr %337, 1
  store [2 x ptr] %340, ptr %338, align 8
  %341 = call ptr @Z3_mk_and(ptr %7, i32 2, ptr %338)
  %342 = call ptr @Z3_mk_implies(ptr %7, ptr %341, ptr %336)
  %343 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %314, i32 0, ptr null, ptr %342)
  call void @Z3_solver_assert(ptr %7, ptr %6, ptr %343)
  %344 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %345 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %344)
  %346 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %347 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %346)
  %348 = alloca [2 x ptr], align 8
  %349 = insertvalue [2 x ptr] undef, ptr %345, 0
  %350 = insertvalue [2 x ptr] %349, ptr %347, 1
  store [2 x ptr] %350, ptr %348, align 8
  %351 = insertvalue [2 x ptr] undef, ptr %345, 0
  %352 = insertvalue [2 x ptr] %351, ptr %347, 1
  %353 = alloca [2 x ptr], align 8
  store [2 x ptr] %352, ptr %353, align 8
  %354 = call ptr @Z3_mk_app(ptr %7, ptr %147, i32 2, ptr %353)
  %355 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %356 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %355)
  %357 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %358 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %357)
  %359 = call ptr @Z3_mk_eq(ptr %7, ptr %345, ptr %358)
  %360 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %361 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %360)
  %362 = call ptr @Z3_mk_eq(ptr %7, ptr %356, ptr %361)
  %363 = call ptr @Z3_mk_bvadd(ptr %7, ptr %345, ptr %356)
  %364 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %365 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %364)
  %366 = call ptr @Z3_mk_bvadd(ptr %7, ptr %347, ptr %365)
  %367 = insertvalue [2 x ptr] undef, ptr %363, 0
  %368 = insertvalue [2 x ptr] %367, ptr %366, 1
  %369 = alloca [2 x ptr], align 8
  store [2 x ptr] %368, ptr %369, align 8
  %370 = call ptr @Z3_mk_app(ptr %7, ptr %154, i32 2, ptr %369)
  %371 = call ptr @Z3_mk_true(ptr %7)
  %372 = alloca [2 x ptr], align 8
  %373 = insertvalue [2 x ptr] undef, ptr %354, 0
  %374 = insertvalue [2 x ptr] %373, ptr %371, 1
  store [2 x ptr] %374, ptr %372, align 8
  %375 = call ptr @Z3_mk_and(ptr %7, i32 2, ptr %372)
  %376 = call ptr @Z3_mk_implies(ptr %7, ptr %375, ptr %370)
  %377 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %348, i32 0, ptr null, ptr %376)
  call void @Z3_solver_assert(ptr %7, ptr %6, ptr %377)
  %378 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %379 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %378)
  %380 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %381 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %380)
  %382 = alloca [2 x ptr], align 8
  %383 = insertvalue [2 x ptr] undef, ptr %379, 0
  %384 = insertvalue [2 x ptr] %383, ptr %381, 1
  store [2 x ptr] %384, ptr %382, align 8
  %385 = insertvalue [2 x ptr] undef, ptr %379, 0
  %386 = insertvalue [2 x ptr] %385, ptr %381, 1
  %387 = alloca [2 x ptr], align 8
  store [2 x ptr] %386, ptr %387, align 8
  %388 = call ptr @Z3_mk_app(ptr %7, ptr %154, i32 2, ptr %387)
  %389 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %390 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %389)
  %391 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %392 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %391)
  %393 = call ptr @Z3_mk_eq(ptr %7, ptr %379, ptr %392)
  %394 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %395 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %394)
  %396 = call ptr @Z3_mk_eq(ptr %7, ptr %390, ptr %395)
  %397 = call ptr @Z3_mk_bvadd(ptr %7, ptr %379, ptr %390)
  %398 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %399 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %398)
  %400 = call ptr @Z3_mk_bvadd(ptr %7, ptr %381, ptr %399)
  %401 = insertvalue [2 x ptr] undef, ptr %397, 0
  %402 = insertvalue [2 x ptr] %401, ptr %400, 1
  %403 = alloca [2 x ptr], align 8
  store [2 x ptr] %402, ptr %403, align 8
  %404 = call ptr @Z3_mk_app(ptr %7, ptr %161, i32 2, ptr %403)
  %405 = call ptr @Z3_mk_true(ptr %7)
  %406 = alloca [2 x ptr], align 8
  %407 = insertvalue [2 x ptr] undef, ptr %388, 0
  %408 = insertvalue [2 x ptr] %407, ptr %405, 1
  store [2 x ptr] %408, ptr %406, align 8
  %409 = call ptr @Z3_mk_and(ptr %7, i32 2, ptr %406)
  %410 = call ptr @Z3_mk_implies(ptr %7, ptr %409, ptr %404)
  %411 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %382, i32 0, ptr null, ptr %410)
  call void @Z3_solver_assert(ptr %7, ptr %6, ptr %411)
  %412 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %413 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %412)
  %414 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %415 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %414)
  %416 = alloca [2 x ptr], align 8
  %417 = insertvalue [2 x ptr] undef, ptr %413, 0
  %418 = insertvalue [2 x ptr] %417, ptr %415, 1
  store [2 x ptr] %418, ptr %416, align 8
  %419 = insertvalue [2 x ptr] undef, ptr %413, 0
  %420 = insertvalue [2 x ptr] %419, ptr %415, 1
  %421 = alloca [2 x ptr], align 8
  store [2 x ptr] %420, ptr %421, align 8
  %422 = call ptr @Z3_mk_app(ptr %7, ptr %161, i32 2, ptr %421)
  %423 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %424 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %423)
  %425 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %426 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %425)
  %427 = call ptr @Z3_mk_eq(ptr %7, ptr %413, ptr %426)
  %428 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %429 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %428)
  %430 = call ptr @Z3_mk_eq(ptr %7, ptr %424, ptr %429)
  %431 = call ptr @Z3_mk_bvadd(ptr %7, ptr %413, ptr %424)
  %432 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %433 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %432)
  %434 = call ptr @Z3_mk_bvadd(ptr %7, ptr %415, ptr %433)
  %435 = insertvalue [2 x ptr] undef, ptr %431, 0
  %436 = insertvalue [2 x ptr] %435, ptr %434, 1
  %437 = alloca [2 x ptr], align 8
  store [2 x ptr] %436, ptr %437, align 8
  %438 = call ptr @Z3_mk_app(ptr %7, ptr %168, i32 2, ptr %437)
  %439 = call ptr @Z3_mk_true(ptr %7)
  %440 = alloca [2 x ptr], align 8
  %441 = insertvalue [2 x ptr] undef, ptr %422, 0
  %442 = insertvalue [2 x ptr] %441, ptr %439, 1
  store [2 x ptr] %442, ptr %440, align 8
  %443 = call ptr @Z3_mk_and(ptr %7, i32 2, ptr %440)
  %444 = call ptr @Z3_mk_implies(ptr %7, ptr %443, ptr %438)
  %445 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %416, i32 0, ptr null, ptr %444)
  call void @Z3_solver_assert(ptr %7, ptr %6, ptr %445)
  %446 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %447 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %446)
  %448 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %449 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %448)
  %450 = alloca [2 x ptr], align 8
  %451 = insertvalue [2 x ptr] undef, ptr %447, 0
  %452 = insertvalue [2 x ptr] %451, ptr %449, 1
  store [2 x ptr] %452, ptr %450, align 8
  %453 = insertvalue [2 x ptr] undef, ptr %447, 0
  %454 = insertvalue [2 x ptr] %453, ptr %449, 1
  %455 = alloca [2 x ptr], align 8
  store [2 x ptr] %454, ptr %455, align 8
  %456 = call ptr @Z3_mk_app(ptr %7, ptr %168, i32 2, ptr %455)
  %457 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %458 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %457)
  %459 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %460 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %459)
  %461 = call ptr @Z3_mk_eq(ptr %7, ptr %447, ptr %460)
  %462 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %463 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %462)
  %464 = call ptr @Z3_mk_eq(ptr %7, ptr %458, ptr %463)
  %465 = call ptr @Z3_mk_bvadd(ptr %7, ptr %447, ptr %458)
  %466 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %467 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %466)
  %468 = call ptr @Z3_mk_bvadd(ptr %7, ptr %449, ptr %467)
  %469 = insertvalue [2 x ptr] undef, ptr %465, 0
  %470 = insertvalue [2 x ptr] %469, ptr %468, 1
  %471 = alloca [2 x ptr], align 8
  store [2 x ptr] %470, ptr %471, align 8
  %472 = call ptr @Z3_mk_app(ptr %7, ptr %175, i32 2, ptr %471)
  %473 = call ptr @Z3_mk_true(ptr %7)
  %474 = alloca [2 x ptr], align 8
  %475 = insertvalue [2 x ptr] undef, ptr %456, 0
  %476 = insertvalue [2 x ptr] %475, ptr %473, 1
  store [2 x ptr] %476, ptr %474, align 8
  %477 = call ptr @Z3_mk_and(ptr %7, i32 2, ptr %474)
  %478 = call ptr @Z3_mk_implies(ptr %7, ptr %477, ptr %472)
  %479 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %450, i32 0, ptr null, ptr %478)
  call void @Z3_solver_assert(ptr %7, ptr %6, ptr %479)
  %480 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %481 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %480)
  %482 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %483 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %482)
  %484 = alloca [2 x ptr], align 8
  %485 = insertvalue [2 x ptr] undef, ptr %481, 0
  %486 = insertvalue [2 x ptr] %485, ptr %483, 1
  store [2 x ptr] %486, ptr %484, align 8
  %487 = insertvalue [2 x ptr] undef, ptr %481, 0
  %488 = insertvalue [2 x ptr] %487, ptr %483, 1
  %489 = alloca [2 x ptr], align 8
  store [2 x ptr] %488, ptr %489, align 8
  %490 = call ptr @Z3_mk_app(ptr %7, ptr %175, i32 2, ptr %489)
  %491 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %492 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %491)
  %493 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %494 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %493)
  %495 = call ptr @Z3_mk_eq(ptr %7, ptr %481, ptr %494)
  %496 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %497 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %496)
  %498 = call ptr @Z3_mk_eq(ptr %7, ptr %492, ptr %497)
  %499 = call ptr @Z3_mk_bvadd(ptr %7, ptr %481, ptr %492)
  %500 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %501 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %500)
  %502 = call ptr @Z3_mk_bvadd(ptr %7, ptr %483, ptr %501)
  %503 = insertvalue [2 x ptr] undef, ptr %499, 0
  %504 = insertvalue [2 x ptr] %503, ptr %502, 1
  %505 = alloca [2 x ptr], align 8
  store [2 x ptr] %504, ptr %505, align 8
  %506 = call ptr @Z3_mk_app(ptr %7, ptr %182, i32 2, ptr %505)
  %507 = call ptr @Z3_mk_true(ptr %7)
  %508 = alloca [2 x ptr], align 8
  %509 = insertvalue [2 x ptr] undef, ptr %490, 0
  %510 = insertvalue [2 x ptr] %509, ptr %507, 1
  store [2 x ptr] %510, ptr %508, align 8
  %511 = call ptr @Z3_mk_and(ptr %7, i32 2, ptr %508)
  %512 = call ptr @Z3_mk_implies(ptr %7, ptr %511, ptr %506)
  %513 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %484, i32 0, ptr null, ptr %512)
  call void @Z3_solver_assert(ptr %7, ptr %6, ptr %513)
  %514 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %515 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %514)
  %516 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %517 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %516)
  %518 = alloca [2 x ptr], align 8
  %519 = insertvalue [2 x ptr] undef, ptr %515, 0
  %520 = insertvalue [2 x ptr] %519, ptr %517, 1
  store [2 x ptr] %520, ptr %518, align 8
  %521 = insertvalue [2 x ptr] undef, ptr %515, 0
  %522 = insertvalue [2 x ptr] %521, ptr %517, 1
  %523 = alloca [2 x ptr], align 8
  store [2 x ptr] %522, ptr %523, align 8
  %524 = call ptr @Z3_mk_app(ptr %7, ptr %182, i32 2, ptr %523)
  %525 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %526 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %525)
  %527 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %528 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %527)
  %529 = call ptr @Z3_mk_eq(ptr %7, ptr %515, ptr %528)
  %530 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %531 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %530)
  %532 = call ptr @Z3_mk_eq(ptr %7, ptr %526, ptr %531)
  %533 = call ptr @Z3_mk_bvadd(ptr %7, ptr %515, ptr %526)
  %534 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %535 = call ptr @Z3_mk_unsigned_int64(ptr %7, i64 1, ptr %534)
  %536 = call ptr @Z3_mk_bvadd(ptr %7, ptr %517, ptr %535)
  %537 = insertvalue [2 x ptr] undef, ptr %533, 0
  %538 = insertvalue [2 x ptr] %537, ptr %536, 1
  %539 = alloca [2 x ptr], align 8
  store [2 x ptr] %538, ptr %539, align 8
  %540 = call ptr @Z3_mk_app(ptr %7, ptr %189, i32 2, ptr %539)
  %541 = call ptr @Z3_mk_true(ptr %7)
  %542 = alloca [2 x ptr], align 8
  %543 = insertvalue [2 x ptr] undef, ptr %524, 0
  %544 = insertvalue [2 x ptr] %543, ptr %541, 1
  store [2 x ptr] %544, ptr %542, align 8
  %545 = call ptr @Z3_mk_and(ptr %7, i32 2, ptr %542)
  %546 = call ptr @Z3_mk_implies(ptr %7, ptr %545, ptr %540)
  %547 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %518, i32 0, ptr null, ptr %546)
  %548 = call ptr @Z3_mk_bv_sort(ptr %7, i32 16)
  %549 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %548)
  %550 = call ptr @Z3_mk_bv_sort(ptr %7, i32 32)
  %551 = call ptr @Z3_mk_fresh_const(ptr %7, ptr null, ptr %550)
  %552 = alloca [2 x ptr], align 8
  %553 = insertvalue [2 x ptr] undef, ptr %549, 0
  %554 = insertvalue [2 x ptr] %553, ptr %551, 1
  store [2 x ptr] %554, ptr %552, align 8
  %555 = insertvalue [2 x ptr] undef, ptr %549, 0
  %556 = insertvalue [2 x ptr] %555, ptr %551, 1
  %557 = alloca [2 x ptr], align 8
  store [2 x ptr] %556, ptr %557, align 8
  %558 = call ptr @Z3_mk_app(ptr %7, ptr %119, i32 2, ptr %557)
  %559 = call ptr @Z3_mk_eq(ptr %7, ptr %4, ptr %551)
  %560 = alloca [2 x ptr], align 8
  %561 = insertvalue [2 x ptr] undef, ptr %549, 0
  %562 = insertvalue [2 x ptr] %561, ptr %105, 1
  store [2 x ptr] %562, ptr %560, align 8
  %563 = call ptr @Z3_mk_distinct(ptr %7, i32 2, ptr %560)
  %564 = alloca [3 x ptr], align 8
  %565 = insertvalue [3 x ptr] undef, ptr %558, 0
  %566 = insertvalue [3 x ptr] %565, ptr %559, 1
  %567 = insertvalue [3 x ptr] %566, ptr %563, 2
  store [3 x ptr] %567, ptr %564, align 8
  %568 = call ptr @Z3_mk_and(ptr %7, i32 3, ptr %564)
  %569 = call ptr @Z3_mk_false(ptr %7)
  %570 = call ptr @Z3_mk_implies(ptr %7, ptr %568, ptr %569)
  %571 = call ptr @Z3_mk_forall_const(ptr %7, i32 0, i32 2, ptr %552, i32 0, ptr null, ptr %570)
  %572 = insertvalue { ptr, ptr, ptr } undef, ptr %111, 0
  %573 = insertvalue { ptr, ptr, ptr } %572, ptr %105, 1
  %574 = insertvalue { ptr, ptr, ptr } %573, ptr %112, 2
  ret { ptr, ptr, ptr } %574
}

define i1 @solver_0() {
  %1 = load ptr, ptr @ctx, align 8
  %2 = call ptr @Z3_mk_bv_sort(ptr %1, i32 32)
  %3 = call ptr @Z3_mk_unsigned_int64(ptr %1, i64 0, ptr %2)
  %4 = call ptr @Z3_mk_bv_sort(ptr %1, i32 16)
  %5 = call ptr @Z3_mk_unsigned_int64(ptr %1, i64 0, ptr %4)
  %6 = call ptr @Z3_mk_bv_sort(ptr %1, i32 4)
  %7 = call ptr @Z3_mk_unsigned_int64(ptr %1, i64 0, ptr %6)
  %8 = call ptr @bmc_init()
  %9 = call ptr @Z3_mk_bv_sort(ptr %1, i32 1)
  %10 = call ptr @Z3_mk_fresh_const(ptr %1, ptr null, ptr %9)
  br label %11

11:                                               ; preds = %39, %0
  %12 = phi i32 [ %45, %39 ], [ 0, %0 ]
  %13 = phi ptr [ %42, %39 ], [ %8, %0 ]
  %14 = phi ptr [ %44, %39 ], [ %10, %0 ]
  %15 = phi ptr [ %24, %39 ], [ %7, %0 ]
  %16 = phi ptr [ %25, %39 ], [ %5, %0 ]
  %17 = phi ptr [ %26, %39 ], [ %3, %0 ]
  %18 = phi i1 [ %41, %39 ], [ false, %0 ]
  %19 = icmp slt i32 %12, 20
  br i1 %19, label %20, label %46

20:                                               ; preds = %11
  %21 = load ptr, ptr @ctx, align 8
  %22 = load ptr, ptr @solver, align 8
  %23 = call { ptr, ptr, ptr } @bmc_circuit(ptr %13, ptr %14, ptr %15, ptr %16, ptr %17)
  %24 = extractvalue { ptr, ptr, ptr } %23, 0
  %25 = extractvalue { ptr, ptr, ptr } %23, 1
  %26 = extractvalue { ptr, ptr, ptr } %23, 2
  %27 = call i32 @Z3_solver_check(ptr %21, ptr %22)
  %28 = icmp eq i32 %27, 1
  br i1 %28, label %29, label %30

29:                                               ; preds = %20
  br label %37

30:                                               ; preds = %20
  %31 = icmp eq i32 %27, -1
  br i1 %31, label %32, label %33

32:                                               ; preds = %30
  br label %34

33:                                               ; preds = %30
  br label %34

34:                                               ; preds = %32, %33
  %35 = phi i1 [ true, %33 ], [ false, %32 ]
  br label %36

36:                                               ; preds = %34
  br label %37

37:                                               ; preds = %29, %36
  %38 = phi i1 [ %35, %36 ], [ true, %29 ]
  br label %39

39:                                               ; preds = %37
  %40 = load ptr, ptr @ctx, align 8
  %41 = or i1 %38, %18
  %42 = call ptr @bmc_loop(ptr %13)
  %43 = call ptr @Z3_mk_bv_sort(ptr %40, i32 1)
  %44 = call ptr @Z3_mk_fresh_const(ptr %40, ptr null, ptr %43)
  %45 = add i32 %12, 1
  br label %11

46:                                               ; preds = %11
  %47 = xor i1 %18, true
  ret i1 %47
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
