#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func private @rtclock() -> f64
  func.func private @record_timing(!llvm.ptr, f64)
  llvm.mlir.global private constant @op_name_000_arg0_1("op_name_000_arg0_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_001_arg0_1("op_name_001_arg0_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_002_arg1_1("op_name_002_arg1_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_003_arg2_1("op_name_003_arg2_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_004_arg3_1("op_name_004_arg3_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_005_arg4_1("op_name_005_arg4_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_006_arg5_1("op_name_006_arg5_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_007_arg6_1("op_name_007_arg6_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_008_arg0_1("op_name_008_arg0_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_009_arg7_1("op_name_009_arg7_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_010_arg8_1("op_name_010_arg8_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_011_arg9_1("op_name_011_arg9_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_012_arg10_1("op_name_012_arg10_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_013_pow_1("op_name_013_pow_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_014_mean("op_name_014_mean\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_015_add("op_name_015_add\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_016_rsqrt("op_name_016_rsqrt\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_017_mul("op_name_017_mul\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_018_mul_1("op_name_018_mul_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_019_permute("op_name_019_permute\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_020_view("op_name_020_view\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_021_mm("op_name_021_mm\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_022_view_1("op_name_022_view_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_023_permute_1("op_name_023_permute_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_024_view_2("op_name_024_view_2\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_025_mm_1("op_name_025_mm_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_026_view_3("op_name_026_view_3\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_027_permute_2("op_name_027_permute_2\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_028_view_4("op_name_028_view_4\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_029_mm_2("op_name_029_mm_2\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_030_view_5("op_name_030_view_5\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_031_view_6("op_name_031_view_6\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_032_permute_3("op_name_032_permute_3\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_033_view_7("op_name_033_view_7\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_034_permute_4("op_name_034_permute_4\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_035_view_8("op_name_035_view_8\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_036_permute_5("op_name_036_permute_5\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_037_unsqueeze("op_name_037_unsqueeze\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_038_expand("op_name_038_expand\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_039_clone("op_name_039_clone\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_040_view_9("op_name_040_view_9\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_041_unsqueeze_1("op_name_041_unsqueeze_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_042_expand_1("op_name_042_expand_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_043_clone_1("op_name_043_clone_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_044_view_10("op_name_044_view_10\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_045_permute_6("op_name_045_permute_6\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_046_expand_2("op_name_046_expand_2\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_047_view_11("op_name_047_view_11\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_048_expand_3("op_name_048_expand_3\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_049_view_12("op_name_049_view_12\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_050_bmm("op_name_050_bmm\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_051_view_13("op_name_051_view_13\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_052_mul_2("op_name_052_mul_2\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_053_view_14("op_name_053_view_14\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_054_convert_element_type("op_name_054_convert_element_type\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_055_sub("op_name_055_sub\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_056_mul_3("op_name_056_mul_3\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_057_add_1("op_name_057_add_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_058_amax("op_name_058_amax\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_059_sub_1("op_name_059_sub_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_060_exp("op_name_060_exp\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_061_sum_1("op_name_061_sum_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_062_div("op_name_062_div\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_063_expand_4("op_name_063_expand_4\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_064_view_15("op_name_064_view_15\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_065_expand_5("op_name_065_expand_5\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_066_view_16("op_name_066_view_16\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_067_bmm_1("op_name_067_bmm_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_068_view_17("op_name_068_view_17\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_069_permute_7("op_name_069_permute_7\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_070_clone_2("op_name_070_clone_2\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_071_view_18("op_name_071_view_18\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_072_permute_8("op_name_072_permute_8\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_073_view_19("op_name_073_view_19\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_074_mm_3("op_name_074_mm_3\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_075_view_20("op_name_075_view_20\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_076_add_2("op_name_076_add_2\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_077_pow_2("op_name_077_pow_2\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_078_mean_1("op_name_078_mean_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_079_add_3("op_name_079_add_3\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_080_rsqrt_1("op_name_080_rsqrt_1\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_081_mul_4("op_name_081_mul_4\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_082_mul_5("op_name_082_mul_5\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_083_permute_9("op_name_083_permute_9\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_084_view_21("op_name_084_view_21\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_085_mm_4("op_name_085_mm_4\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_086_view_22("op_name_086_view_22\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_087_sigmoid("op_name_087_sigmoid\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_088_mul_6("op_name_088_mul_6\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_089_permute_10("op_name_089_permute_10\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_090_view_23("op_name_090_view_23\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_091_mm_5("op_name_091_mm_5\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_092_view_24("op_name_092_view_24\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_093_mul_7("op_name_093_mul_7\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_094_permute_11("op_name_094_permute_11\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_095_view_25("op_name_095_view_25\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_096_mm_6("op_name_096_mm_6\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_097_view_26("op_name_097_view_26\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_098_add_4("op_name_098_add_4\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_099_output("op_name_099_output\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @op_name_100_output("op_name_100_output\00") {addr_space = 0 : i32}
  func.func @subgraph0(%arg0: tensor<1x40x1536xf32>, %arg1: tensor<1x40x1536xf32>, %arg2: tensor<1536xf32>, %arg3: tensor<1536x1536xf32>, %arg4: tensor<256x1536xf32>, %arg5: tensor<256x1536xf32>, %arg6: tensor<1x40xi64>, %arg7: tensor<1536x1536xf32>, %arg8: tensor<1x40x1536xf32>, %arg9: tensor<1536xf32>, %arg10: tensor<8960x1536xf32>, %arg11: tensor<8960x1536xf32>, %arg12: tensor<1536x8960xf32>) -> tensor<1x40x1536xf32> {
    %0 = call @rtclock() : () -> f64
    %1 = call @rtclock() : () -> f64
    %2 = arith.subf %1, %0 : f64
    %3 = llvm.mlir.addressof @op_name_000_arg0_1 : !llvm.ptr
    call @record_timing(%3, %2) : (!llvm.ptr, f64) -> ()
    %4 = call @rtclock() : () -> f64
    %5 = call @rtclock() : () -> f64
    %6 = arith.subf %5, %4 : f64
    %7 = llvm.mlir.addressof @op_name_001_arg0_1 : !llvm.ptr
    call @record_timing(%7, %6) : (!llvm.ptr, f64) -> ()
    %8 = call @rtclock() : () -> f64
    %9 = call @rtclock() : () -> f64
    %10 = arith.subf %9, %8 : f64
    %11 = llvm.mlir.addressof @op_name_002_arg1_1 : !llvm.ptr
    call @record_timing(%11, %10) : (!llvm.ptr, f64) -> ()
    %12 = call @rtclock() : () -> f64
    %13 = call @rtclock() : () -> f64
    %14 = arith.subf %13, %12 : f64
    %15 = llvm.mlir.addressof @op_name_003_arg2_1 : !llvm.ptr
    call @record_timing(%15, %14) : (!llvm.ptr, f64) -> ()
    %16 = call @rtclock() : () -> f64
    %17 = call @rtclock() : () -> f64
    %18 = arith.subf %17, %16 : f64
    %19 = llvm.mlir.addressof @op_name_004_arg3_1 : !llvm.ptr
    call @record_timing(%19, %18) : (!llvm.ptr, f64) -> ()
    %20 = call @rtclock() : () -> f64
    %21 = call @rtclock() : () -> f64
    %22 = arith.subf %21, %20 : f64
    %23 = llvm.mlir.addressof @op_name_005_arg4_1 : !llvm.ptr
    call @record_timing(%23, %22) : (!llvm.ptr, f64) -> ()
    %24 = call @rtclock() : () -> f64
    %25 = call @rtclock() : () -> f64
    %26 = arith.subf %25, %24 : f64
    %27 = llvm.mlir.addressof @op_name_006_arg5_1 : !llvm.ptr
    call @record_timing(%27, %26) : (!llvm.ptr, f64) -> ()
    %28 = call @rtclock() : () -> f64
    %29 = call @rtclock() : () -> f64
    %30 = arith.subf %29, %28 : f64
    %31 = llvm.mlir.addressof @op_name_007_arg6_1 : !llvm.ptr
    call @record_timing(%31, %30) : (!llvm.ptr, f64) -> ()
    %32 = call @rtclock() : () -> f64
    %33 = call @rtclock() : () -> f64
    %34 = arith.subf %33, %32 : f64
    %35 = llvm.mlir.addressof @op_name_008_arg0_1 : !llvm.ptr
    call @record_timing(%35, %34) : (!llvm.ptr, f64) -> ()
    %36 = call @rtclock() : () -> f64
    %37 = call @rtclock() : () -> f64
    %38 = arith.subf %37, %36 : f64
    %39 = llvm.mlir.addressof @op_name_009_arg7_1 : !llvm.ptr
    call @record_timing(%39, %38) : (!llvm.ptr, f64) -> ()
    %40 = call @rtclock() : () -> f64
    %41 = call @rtclock() : () -> f64
    %42 = arith.subf %41, %40 : f64
    %43 = llvm.mlir.addressof @op_name_010_arg8_1 : !llvm.ptr
    call @record_timing(%43, %42) : (!llvm.ptr, f64) -> ()
    %44 = call @rtclock() : () -> f64
    %45 = call @rtclock() : () -> f64
    %46 = arith.subf %45, %44 : f64
    %47 = llvm.mlir.addressof @op_name_011_arg9_1 : !llvm.ptr
    call @record_timing(%47, %46) : (!llvm.ptr, f64) -> ()
    %48 = call @rtclock() : () -> f64
    %49 = call @rtclock() : () -> f64
    %50 = arith.subf %49, %48 : f64
    %51 = llvm.mlir.addressof @op_name_012_arg10_1 : !llvm.ptr
    call @record_timing(%51, %50) : (!llvm.ptr, f64) -> ()
    %52 = call @rtclock() : () -> f64
    %53 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32 = arith.constant 2 : i32
    %54 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8 : tensor<1x40x1536xf32>) outs(%53 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %553 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %553 : f32
    } -> tensor<1x40x1536xf32>
    %55 = call @rtclock() : () -> f64
    %56 = arith.subf %55, %52 : f64
    %57 = llvm.mlir.addressof @op_name_013_pow_1 : !llvm.ptr
    call @record_timing(%57, %56) : (!llvm.ptr, f64) -> ()
    %58 = call @rtclock() : () -> f64
    %59 = tosa.reduce_sum %54 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %60 = "tosa.const"() <{values = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %61 = tosa.reciprocal %60 : (tensor<1xf32>) -> tensor<1xf32>
    %62 = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
    %63 = tosa.reshape %61, %62 : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
    %64 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %65 = tosa.mul %63, %59, %64 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>, tensor<1xi8>) -> tensor<1x40x1xf32>
    %66 = call @rtclock() : () -> f64
    %67 = arith.subf %66, %58 : f64
    %68 = llvm.mlir.addressof @op_name_014_mean : !llvm.ptr
    call @record_timing(%68, %67) : (!llvm.ptr, f64) -> ()
    %69 = call @rtclock() : () -> f64
    %70 = "tosa.const"() <{values = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %71 = tosa.add %65, %70 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %72 = call @rtclock() : () -> f64
    %73 = arith.subf %72, %69 : f64
    %74 = llvm.mlir.addressof @op_name_015_add : !llvm.ptr
    call @record_timing(%74, %73) : (!llvm.ptr, f64) -> ()
    %75 = call @rtclock() : () -> f64
    %76 = tosa.rsqrt %71 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %77 = call @rtclock() : () -> f64
    %78 = arith.subf %77, %75 : f64
    %79 = llvm.mlir.addressof @op_name_016_rsqrt : !llvm.ptr
    call @record_timing(%79, %78) : (!llvm.ptr, f64) -> ()
    %80 = call @rtclock() : () -> f64
    %81 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %82 = tosa.mul %arg8, %76, %81 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>
    %83 = call @rtclock() : () -> f64
    %84 = arith.subf %83, %80 : f64
    %85 = llvm.mlir.addressof @op_name_017_mul : !llvm.ptr
    call @record_timing(%85, %84) : (!llvm.ptr, f64) -> ()
    %86 = call @rtclock() : () -> f64
    %87 = tosa.const_shape  {values = dense<[1, 1, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %88 = tosa.reshape %arg2, %87 : (tensor<1536xf32>, !tosa.shape<3>) -> tensor<1x1x1536xf32>
    %89 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %90 = tosa.mul %88, %82, %89 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>
    %91 = call @rtclock() : () -> f64
    %92 = arith.subf %91, %86 : f64
    %93 = llvm.mlir.addressof @op_name_018_mul_1 : !llvm.ptr
    call @record_timing(%93, %92) : (!llvm.ptr, f64) -> ()
    %94 = call @rtclock() : () -> f64
    %95 = tosa.transpose %arg3 {perms = array<i32: 1, 0>} : (tensor<1536x1536xf32>) -> tensor<1536x1536xf32>
    %96 = call @rtclock() : () -> f64
    %97 = arith.subf %96, %94 : f64
    %98 = llvm.mlir.addressof @op_name_019_permute : !llvm.ptr
    call @record_timing(%98, %97) : (!llvm.ptr, f64) -> ()
    %99 = call @rtclock() : () -> f64
    %100 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %101 = tosa.reshape %90, %100 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %102 = call @rtclock() : () -> f64
    %103 = arith.subf %102, %99 : f64
    %104 = llvm.mlir.addressof @op_name_020_view : !llvm.ptr
    call @record_timing(%104, %103) : (!llvm.ptr, f64) -> ()
    %105 = call @rtclock() : () -> f64
    %cst = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %106 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%101, %95 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %107 = call @rtclock() : () -> f64
    %108 = arith.subf %107, %105 : f64
    %109 = llvm.mlir.addressof @op_name_021_mm : !llvm.ptr
    call @record_timing(%109, %108) : (!llvm.ptr, f64) -> ()
    %110 = call @rtclock() : () -> f64
    %111 = tosa.const_shape  {values = dense<[1, 40, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %112 = tosa.reshape %106, %111 : (tensor<40x1536xf32>, !tosa.shape<3>) -> tensor<1x40x1536xf32>
    %113 = call @rtclock() : () -> f64
    %114 = arith.subf %113, %110 : f64
    %115 = llvm.mlir.addressof @op_name_022_view_1 : !llvm.ptr
    call @record_timing(%115, %114) : (!llvm.ptr, f64) -> ()
    %116 = call @rtclock() : () -> f64
    %117 = tosa.transpose %arg4 {perms = array<i32: 1, 0>} : (tensor<256x1536xf32>) -> tensor<1536x256xf32>
    %118 = call @rtclock() : () -> f64
    %119 = arith.subf %118, %116 : f64
    %120 = llvm.mlir.addressof @op_name_023_permute_1 : !llvm.ptr
    call @record_timing(%120, %119) : (!llvm.ptr, f64) -> ()
    %121 = call @rtclock() : () -> f64
    %122 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %123 = tosa.reshape %90, %122 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %124 = call @rtclock() : () -> f64
    %125 = arith.subf %124, %121 : f64
    %126 = llvm.mlir.addressof @op_name_024_view_2 : !llvm.ptr
    call @record_timing(%126, %125) : (!llvm.ptr, f64) -> ()
    %127 = call @rtclock() : () -> f64
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<40x256xf32>
    %128 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%123, %117 : tensor<40x1536xf32>, tensor<1536x256xf32>) outs(%cst_0 : tensor<40x256xf32>) -> tensor<40x256xf32>
    %129 = call @rtclock() : () -> f64
    %130 = arith.subf %129, %127 : f64
    %131 = llvm.mlir.addressof @op_name_025_mm_1 : !llvm.ptr
    call @record_timing(%131, %130) : (!llvm.ptr, f64) -> ()
    %132 = call @rtclock() : () -> f64
    %133 = tosa.const_shape  {values = dense<[1, 40, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %134 = tosa.reshape %128, %133 : (tensor<40x256xf32>, !tosa.shape<3>) -> tensor<1x40x256xf32>
    %135 = call @rtclock() : () -> f64
    %136 = arith.subf %135, %132 : f64
    %137 = llvm.mlir.addressof @op_name_026_view_3 : !llvm.ptr
    call @record_timing(%137, %136) : (!llvm.ptr, f64) -> ()
    %138 = call @rtclock() : () -> f64
    %139 = tosa.transpose %arg5 {perms = array<i32: 1, 0>} : (tensor<256x1536xf32>) -> tensor<1536x256xf32>
    %140 = call @rtclock() : () -> f64
    %141 = arith.subf %140, %138 : f64
    %142 = llvm.mlir.addressof @op_name_027_permute_2 : !llvm.ptr
    call @record_timing(%142, %141) : (!llvm.ptr, f64) -> ()
    %143 = call @rtclock() : () -> f64
    %144 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %145 = tosa.reshape %90, %144 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %146 = call @rtclock() : () -> f64
    %147 = arith.subf %146, %143 : f64
    %148 = llvm.mlir.addressof @op_name_028_view_4 : !llvm.ptr
    call @record_timing(%148, %147) : (!llvm.ptr, f64) -> ()
    %149 = call @rtclock() : () -> f64
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<40x256xf32>
    %150 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%145, %139 : tensor<40x1536xf32>, tensor<1536x256xf32>) outs(%cst_1 : tensor<40x256xf32>) -> tensor<40x256xf32>
    %151 = call @rtclock() : () -> f64
    %152 = arith.subf %151, %149 : f64
    %153 = llvm.mlir.addressof @op_name_029_mm_2 : !llvm.ptr
    call @record_timing(%153, %152) : (!llvm.ptr, f64) -> ()
    %154 = call @rtclock() : () -> f64
    %155 = tosa.const_shape  {values = dense<[1, 40, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %156 = tosa.reshape %150, %155 : (tensor<40x256xf32>, !tosa.shape<3>) -> tensor<1x40x256xf32>
    %157 = call @rtclock() : () -> f64
    %158 = arith.subf %157, %154 : f64
    %159 = llvm.mlir.addressof @op_name_030_view_5 : !llvm.ptr
    call @record_timing(%159, %158) : (!llvm.ptr, f64) -> ()
    %160 = call @rtclock() : () -> f64
    %161 = tosa.const_shape  {values = dense<[1, 40, 12, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %162 = tosa.reshape %112, %161 : (tensor<1x40x1536xf32>, !tosa.shape<4>) -> tensor<1x40x12x128xf32>
    %163 = call @rtclock() : () -> f64
    %164 = arith.subf %163, %160 : f64
    %165 = llvm.mlir.addressof @op_name_031_view_6 : !llvm.ptr
    call @record_timing(%165, %164) : (!llvm.ptr, f64) -> ()
    %166 = call @rtclock() : () -> f64
    %167 = tosa.transpose %162 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x40x12x128xf32>) -> tensor<1x12x40x128xf32>
    %168 = call @rtclock() : () -> f64
    %169 = arith.subf %168, %166 : f64
    %170 = llvm.mlir.addressof @op_name_032_permute_3 : !llvm.ptr
    call @record_timing(%170, %169) : (!llvm.ptr, f64) -> ()
    %171 = call @rtclock() : () -> f64
    %172 = tosa.const_shape  {values = dense<[1, 40, 2, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %173 = tosa.reshape %134, %172 : (tensor<1x40x256xf32>, !tosa.shape<4>) -> tensor<1x40x2x128xf32>
    %174 = call @rtclock() : () -> f64
    %175 = arith.subf %174, %171 : f64
    %176 = llvm.mlir.addressof @op_name_033_view_7 : !llvm.ptr
    call @record_timing(%176, %175) : (!llvm.ptr, f64) -> ()
    %177 = call @rtclock() : () -> f64
    %178 = tosa.transpose %173 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x40x2x128xf32>) -> tensor<1x2x40x128xf32>
    %179 = call @rtclock() : () -> f64
    %180 = arith.subf %179, %177 : f64
    %181 = llvm.mlir.addressof @op_name_034_permute_4 : !llvm.ptr
    call @record_timing(%181, %180) : (!llvm.ptr, f64) -> ()
    %182 = call @rtclock() : () -> f64
    %183 = tosa.const_shape  {values = dense<[1, 40, 2, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %184 = tosa.reshape %156, %183 : (tensor<1x40x256xf32>, !tosa.shape<4>) -> tensor<1x40x2x128xf32>
    %185 = call @rtclock() : () -> f64
    %186 = arith.subf %185, %182 : f64
    %187 = llvm.mlir.addressof @op_name_035_view_8 : !llvm.ptr
    call @record_timing(%187, %186) : (!llvm.ptr, f64) -> ()
    %188 = call @rtclock() : () -> f64
    %189 = tosa.transpose %184 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x40x2x128xf32>) -> tensor<1x2x40x128xf32>
    %190 = call @rtclock() : () -> f64
    %191 = arith.subf %190, %188 : f64
    %192 = llvm.mlir.addressof @op_name_036_permute_5 : !llvm.ptr
    call @record_timing(%192, %191) : (!llvm.ptr, f64) -> ()
    %193 = call @rtclock() : () -> f64
    %194 = tosa.const_shape  {values = dense<[1, 2, 1, 40, 128]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %195 = tosa.reshape %178, %194 : (tensor<1x2x40x128xf32>, !tosa.shape<5>) -> tensor<1x2x1x40x128xf32>
    %196 = call @rtclock() : () -> f64
    %197 = arith.subf %196, %193 : f64
    %198 = llvm.mlir.addressof @op_name_037_unsqueeze : !llvm.ptr
    call @record_timing(%198, %197) : (!llvm.ptr, f64) -> ()
    %199 = call @rtclock() : () -> f64
    %200 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %201 = tosa.add %195, %200 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %202 = call @rtclock() : () -> f64
    %203 = arith.subf %202, %199 : f64
    %204 = llvm.mlir.addressof @op_name_038_expand : !llvm.ptr
    call @record_timing(%204, %203) : (!llvm.ptr, f64) -> ()
    %205 = call @rtclock() : () -> f64
    %206 = call @rtclock() : () -> f64
    %207 = arith.subf %206, %205 : f64
    %208 = llvm.mlir.addressof @op_name_039_clone : !llvm.ptr
    call @record_timing(%208, %207) : (!llvm.ptr, f64) -> ()
    %209 = call @rtclock() : () -> f64
    %210 = tosa.const_shape  {values = dense<[1, 12, 40, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %211 = tosa.reshape %201, %210 : (tensor<1x2x6x40x128xf32>, !tosa.shape<4>) -> tensor<1x12x40x128xf32>
    %212 = call @rtclock() : () -> f64
    %213 = arith.subf %212, %209 : f64
    %214 = llvm.mlir.addressof @op_name_040_view_9 : !llvm.ptr
    call @record_timing(%214, %213) : (!llvm.ptr, f64) -> ()
    %215 = call @rtclock() : () -> f64
    %216 = tosa.const_shape  {values = dense<[1, 2, 1, 40, 128]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %217 = tosa.reshape %189, %216 : (tensor<1x2x40x128xf32>, !tosa.shape<5>) -> tensor<1x2x1x40x128xf32>
    %218 = call @rtclock() : () -> f64
    %219 = arith.subf %218, %215 : f64
    %220 = llvm.mlir.addressof @op_name_041_unsqueeze_1 : !llvm.ptr
    call @record_timing(%220, %219) : (!llvm.ptr, f64) -> ()
    %221 = call @rtclock() : () -> f64
    %222 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x2x6x40x128xf32>}> : () -> tensor<1x2x6x40x128xf32>
    %223 = tosa.add %217, %222 : (tensor<1x2x1x40x128xf32>, tensor<1x2x6x40x128xf32>) -> tensor<1x2x6x40x128xf32>
    %224 = call @rtclock() : () -> f64
    %225 = arith.subf %224, %221 : f64
    %226 = llvm.mlir.addressof @op_name_042_expand_1 : !llvm.ptr
    call @record_timing(%226, %225) : (!llvm.ptr, f64) -> ()
    %227 = call @rtclock() : () -> f64
    %228 = call @rtclock() : () -> f64
    %229 = arith.subf %228, %227 : f64
    %230 = llvm.mlir.addressof @op_name_043_clone_1 : !llvm.ptr
    call @record_timing(%230, %229) : (!llvm.ptr, f64) -> ()
    %231 = call @rtclock() : () -> f64
    %232 = tosa.const_shape  {values = dense<[1, 12, 40, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %233 = tosa.reshape %223, %232 : (tensor<1x2x6x40x128xf32>, !tosa.shape<4>) -> tensor<1x12x40x128xf32>
    %234 = call @rtclock() : () -> f64
    %235 = arith.subf %234, %231 : f64
    %236 = llvm.mlir.addressof @op_name_044_view_10 : !llvm.ptr
    call @record_timing(%236, %235) : (!llvm.ptr, f64) -> ()
    %237 = call @rtclock() : () -> f64
    %238 = tosa.transpose %211 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x40x128xf32>) -> tensor<1x12x128x40xf32>
    %239 = call @rtclock() : () -> f64
    %240 = arith.subf %239, %237 : f64
    %241 = llvm.mlir.addressof @op_name_045_permute_6 : !llvm.ptr
    call @record_timing(%241, %240) : (!llvm.ptr, f64) -> ()
    %242 = call @rtclock() : () -> f64
    %243 = call @rtclock() : () -> f64
    %244 = arith.subf %243, %242 : f64
    %245 = llvm.mlir.addressof @op_name_046_expand_2 : !llvm.ptr
    call @record_timing(%245, %244) : (!llvm.ptr, f64) -> ()
    %246 = call @rtclock() : () -> f64
    %247 = tosa.const_shape  {values = dense<[12, 40, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %248 = tosa.reshape %167, %247 : (tensor<1x12x40x128xf32>, !tosa.shape<3>) -> tensor<12x40x128xf32>
    %249 = call @rtclock() : () -> f64
    %250 = arith.subf %249, %246 : f64
    %251 = llvm.mlir.addressof @op_name_047_view_11 : !llvm.ptr
    call @record_timing(%251, %250) : (!llvm.ptr, f64) -> ()
    %252 = call @rtclock() : () -> f64
    %253 = call @rtclock() : () -> f64
    %254 = arith.subf %253, %252 : f64
    %255 = llvm.mlir.addressof @op_name_048_expand_3 : !llvm.ptr
    call @record_timing(%255, %254) : (!llvm.ptr, f64) -> ()
    %256 = call @rtclock() : () -> f64
    %257 = tosa.const_shape  {values = dense<[12, 128, 40]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %258 = tosa.reshape %238, %257 : (tensor<1x12x128x40xf32>, !tosa.shape<3>) -> tensor<12x128x40xf32>
    %259 = call @rtclock() : () -> f64
    %260 = arith.subf %259, %256 : f64
    %261 = llvm.mlir.addressof @op_name_049_view_12 : !llvm.ptr
    call @record_timing(%261, %260) : (!llvm.ptr, f64) -> ()
    %262 = call @rtclock() : () -> f64
    %263 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %264 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %265 = tosa.matmul %248, %258, %263, %264 : (tensor<12x40x128xf32>, tensor<12x128x40xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x40x40xf32>
    %266 = call @rtclock() : () -> f64
    %267 = arith.subf %266, %262 : f64
    %268 = llvm.mlir.addressof @op_name_050_bmm : !llvm.ptr
    call @record_timing(%268, %267) : (!llvm.ptr, f64) -> ()
    %269 = call @rtclock() : () -> f64
    %270 = tosa.const_shape  {values = dense<[1, 12, 40, 40]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %271 = tosa.reshape %265, %270 : (tensor<12x40x40xf32>, !tosa.shape<4>) -> tensor<1x12x40x40xf32>
    %272 = call @rtclock() : () -> f64
    %273 = arith.subf %272, %269 : f64
    %274 = llvm.mlir.addressof @op_name_051_view_13 : !llvm.ptr
    call @record_timing(%274, %273) : (!llvm.ptr, f64) -> ()
    %275 = call @rtclock() : () -> f64
    %cst_2 = arith.constant dense<0.0883883461> : tensor<1xf32>
    %276 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
    %277 = tosa.reshape %cst_2, %276 : (tensor<1xf32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
    %278 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %279 = tosa.mul %271, %277, %278 : (tensor<1x12x40x40xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x12x40x40xf32>
    %280 = call @rtclock() : () -> f64
    %281 = arith.subf %280, %275 : f64
    %282 = llvm.mlir.addressof @op_name_052_mul_2 : !llvm.ptr
    call @record_timing(%282, %281) : (!llvm.ptr, f64) -> ()
    %283 = call @rtclock() : () -> f64
    %284 = tosa.const_shape  {values = dense<[1, 1, 1, 40]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %285 = tosa.reshape %arg6, %284 : (tensor<1x40xi64>, !tosa.shape<4>) -> tensor<1x1x1x40xi64>
    %286 = call @rtclock() : () -> f64
    %287 = arith.subf %286, %283 : f64
    %288 = llvm.mlir.addressof @op_name_053_view_14 : !llvm.ptr
    call @record_timing(%288, %287) : (!llvm.ptr, f64) -> ()
    %289 = call @rtclock() : () -> f64
    %290 = tosa.cast %285 : (tensor<1x1x1x40xi64>) -> tensor<1x1x1x40xf32>
    %291 = call @rtclock() : () -> f64
    %292 = arith.subf %291, %289 : f64
    %293 = llvm.mlir.addressof @op_name_054_convert_element_type : !llvm.ptr
    call @record_timing(%293, %292) : (!llvm.ptr, f64) -> ()
    %294 = call @rtclock() : () -> f64
    %295 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1x1x40xf32>}> : () -> tensor<1x1x1x40xf32>
    %296 = tosa.sub %295, %290 : (tensor<1x1x1x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x1x1x40xf32>
    %297 = call @rtclock() : () -> f64
    %298 = arith.subf %297, %294 : f64
    %299 = llvm.mlir.addressof @op_name_055_sub : !llvm.ptr
    call @record_timing(%299, %298) : (!llvm.ptr, f64) -> ()
    %300 = call @rtclock() : () -> f64
    %cst_3 = arith.constant dense<-1.000000e+04> : tensor<1xf32>
    %301 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
    %302 = tosa.reshape %cst_3, %301 : (tensor<1xf32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
    %303 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %304 = tosa.mul %296, %302, %303 : (tensor<1x1x1x40xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x1x40xf32>
    %305 = call @rtclock() : () -> f64
    %306 = arith.subf %305, %300 : f64
    %307 = llvm.mlir.addressof @op_name_056_mul_3 : !llvm.ptr
    call @record_timing(%307, %306) : (!llvm.ptr, f64) -> ()
    %308 = call @rtclock() : () -> f64
    %309 = tosa.add %279, %304 : (tensor<1x12x40x40xf32>, tensor<1x1x1x40xf32>) -> tensor<1x12x40x40xf32>
    %310 = call @rtclock() : () -> f64
    %311 = arith.subf %310, %308 : f64
    %312 = llvm.mlir.addressof @op_name_057_add_1 : !llvm.ptr
    call @record_timing(%312, %311) : (!llvm.ptr, f64) -> ()
    %313 = call @rtclock() : () -> f64
    %314 = tosa.reduce_max %309 {axis = 3 : i32} : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x1xf32>
    %315 = call @rtclock() : () -> f64
    %316 = arith.subf %315, %313 : f64
    %317 = llvm.mlir.addressof @op_name_058_amax : !llvm.ptr
    call @record_timing(%317, %316) : (!llvm.ptr, f64) -> ()
    %318 = call @rtclock() : () -> f64
    %319 = tosa.sub %309, %314 : (tensor<1x12x40x40xf32>, tensor<1x12x40x1xf32>) -> tensor<1x12x40x40xf32>
    %320 = call @rtclock() : () -> f64
    %321 = arith.subf %320, %318 : f64
    %322 = llvm.mlir.addressof @op_name_059_sub_1 : !llvm.ptr
    call @record_timing(%322, %321) : (!llvm.ptr, f64) -> ()
    %323 = call @rtclock() : () -> f64
    %324 = tosa.exp %319 : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x40xf32>
    %325 = call @rtclock() : () -> f64
    %326 = arith.subf %325, %323 : f64
    %327 = llvm.mlir.addressof @op_name_060_exp : !llvm.ptr
    call @record_timing(%327, %326) : (!llvm.ptr, f64) -> ()
    %328 = call @rtclock() : () -> f64
    %329 = tosa.reduce_sum %324 {axis = 3 : i32} : (tensor<1x12x40x40xf32>) -> tensor<1x12x40x1xf32>
    %330 = call @rtclock() : () -> f64
    %331 = arith.subf %330, %328 : f64
    %332 = llvm.mlir.addressof @op_name_061_sum_1 : !llvm.ptr
    call @record_timing(%332, %331) : (!llvm.ptr, f64) -> ()
    %333 = call @rtclock() : () -> f64
    %334 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %335 = tosa.reciprocal %329 : (tensor<1x12x40x1xf32>) -> tensor<1x12x40x1xf32>
    %336 = tosa.mul %324, %335, %334 : (tensor<1x12x40x40xf32>, tensor<1x12x40x1xf32>, tensor<1xi8>) -> tensor<1x12x40x40xf32>
    %337 = call @rtclock() : () -> f64
    %338 = arith.subf %337, %333 : f64
    %339 = llvm.mlir.addressof @op_name_062_div : !llvm.ptr
    call @record_timing(%339, %338) : (!llvm.ptr, f64) -> ()
    %340 = call @rtclock() : () -> f64
    %341 = call @rtclock() : () -> f64
    %342 = arith.subf %341, %340 : f64
    %343 = llvm.mlir.addressof @op_name_063_expand_4 : !llvm.ptr
    call @record_timing(%343, %342) : (!llvm.ptr, f64) -> ()
    %344 = call @rtclock() : () -> f64
    %345 = tosa.const_shape  {values = dense<[12, 40, 40]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %346 = tosa.reshape %336, %345 : (tensor<1x12x40x40xf32>, !tosa.shape<3>) -> tensor<12x40x40xf32>
    %347 = call @rtclock() : () -> f64
    %348 = arith.subf %347, %344 : f64
    %349 = llvm.mlir.addressof @op_name_064_view_15 : !llvm.ptr
    call @record_timing(%349, %348) : (!llvm.ptr, f64) -> ()
    %350 = call @rtclock() : () -> f64
    %351 = call @rtclock() : () -> f64
    %352 = arith.subf %351, %350 : f64
    %353 = llvm.mlir.addressof @op_name_065_expand_5 : !llvm.ptr
    call @record_timing(%353, %352) : (!llvm.ptr, f64) -> ()
    %354 = call @rtclock() : () -> f64
    %355 = tosa.const_shape  {values = dense<[12, 40, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %356 = tosa.reshape %233, %355 : (tensor<1x12x40x128xf32>, !tosa.shape<3>) -> tensor<12x40x128xf32>
    %357 = call @rtclock() : () -> f64
    %358 = arith.subf %357, %354 : f64
    %359 = llvm.mlir.addressof @op_name_066_view_16 : !llvm.ptr
    call @record_timing(%359, %358) : (!llvm.ptr, f64) -> ()
    %360 = call @rtclock() : () -> f64
    %361 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %362 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %363 = tosa.matmul %346, %356, %361, %362 : (tensor<12x40x40xf32>, tensor<12x40x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x40x128xf32>
    %364 = call @rtclock() : () -> f64
    %365 = arith.subf %364, %360 : f64
    %366 = llvm.mlir.addressof @op_name_067_bmm_1 : !llvm.ptr
    call @record_timing(%366, %365) : (!llvm.ptr, f64) -> ()
    %367 = call @rtclock() : () -> f64
    %368 = tosa.const_shape  {values = dense<[1, 12, 40, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %369 = tosa.reshape %363, %368 : (tensor<12x40x128xf32>, !tosa.shape<4>) -> tensor<1x12x40x128xf32>
    %370 = call @rtclock() : () -> f64
    %371 = arith.subf %370, %367 : f64
    %372 = llvm.mlir.addressof @op_name_068_view_17 : !llvm.ptr
    call @record_timing(%372, %371) : (!llvm.ptr, f64) -> ()
    %373 = call @rtclock() : () -> f64
    %374 = tosa.transpose %369 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x12x40x128xf32>) -> tensor<1x40x12x128xf32>
    %375 = call @rtclock() : () -> f64
    %376 = arith.subf %375, %373 : f64
    %377 = llvm.mlir.addressof @op_name_069_permute_7 : !llvm.ptr
    call @record_timing(%377, %376) : (!llvm.ptr, f64) -> ()
    %378 = call @rtclock() : () -> f64
    %379 = call @rtclock() : () -> f64
    %380 = arith.subf %379, %378 : f64
    %381 = llvm.mlir.addressof @op_name_070_clone_2 : !llvm.ptr
    call @record_timing(%381, %380) : (!llvm.ptr, f64) -> ()
    %382 = call @rtclock() : () -> f64
    %383 = tosa.const_shape  {values = dense<[1, 40, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %384 = tosa.reshape %374, %383 : (tensor<1x40x12x128xf32>, !tosa.shape<3>) -> tensor<1x40x1536xf32>
    %385 = call @rtclock() : () -> f64
    %386 = arith.subf %385, %382 : f64
    %387 = llvm.mlir.addressof @op_name_071_view_18 : !llvm.ptr
    call @record_timing(%387, %386) : (!llvm.ptr, f64) -> ()
    %388 = call @rtclock() : () -> f64
    %389 = tosa.transpose %arg7 {perms = array<i32: 1, 0>} : (tensor<1536x1536xf32>) -> tensor<1536x1536xf32>
    %390 = call @rtclock() : () -> f64
    %391 = arith.subf %390, %388 : f64
    %392 = llvm.mlir.addressof @op_name_072_permute_8 : !llvm.ptr
    call @record_timing(%392, %391) : (!llvm.ptr, f64) -> ()
    %393 = call @rtclock() : () -> f64
    %394 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %395 = tosa.reshape %384, %394 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %396 = call @rtclock() : () -> f64
    %397 = arith.subf %396, %393 : f64
    %398 = llvm.mlir.addressof @op_name_073_view_19 : !llvm.ptr
    call @record_timing(%398, %397) : (!llvm.ptr, f64) -> ()
    %399 = call @rtclock() : () -> f64
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %400 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%395, %389 : tensor<40x1536xf32>, tensor<1536x1536xf32>) outs(%cst_4 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %401 = call @rtclock() : () -> f64
    %402 = arith.subf %401, %399 : f64
    %403 = llvm.mlir.addressof @op_name_074_mm_3 : !llvm.ptr
    call @record_timing(%403, %402) : (!llvm.ptr, f64) -> ()
    %404 = call @rtclock() : () -> f64
    %405 = tosa.const_shape  {values = dense<[1, 40, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %406 = tosa.reshape %400, %405 : (tensor<40x1536xf32>, !tosa.shape<3>) -> tensor<1x40x1536xf32>
    %407 = call @rtclock() : () -> f64
    %408 = arith.subf %407, %404 : f64
    %409 = llvm.mlir.addressof @op_name_075_view_20 : !llvm.ptr
    call @record_timing(%409, %408) : (!llvm.ptr, f64) -> ()
    %410 = call @rtclock() : () -> f64
    %411 = tosa.add %arg8, %406 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %412 = call @rtclock() : () -> f64
    %413 = arith.subf %412, %410 : f64
    %414 = llvm.mlir.addressof @op_name_076_add_2 : !llvm.ptr
    call @record_timing(%414, %413) : (!llvm.ptr, f64) -> ()
    %415 = call @rtclock() : () -> f64
    %416 = tensor.empty() : tensor<1x40x1536xf32>
    %c2_i32_5 = arith.constant 2 : i32
    %417 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%411 : tensor<1x40x1536xf32>) outs(%416 : tensor<1x40x1536xf32>) {
    ^bb0(%in: f32, %out: f32):
      %553 = math.fpowi %in, %c2_i32_5 : f32, i32
      linalg.yield %553 : f32
    } -> tensor<1x40x1536xf32>
    %418 = call @rtclock() : () -> f64
    %419 = arith.subf %418, %415 : f64
    %420 = llvm.mlir.addressof @op_name_077_pow_2 : !llvm.ptr
    call @record_timing(%420, %419) : (!llvm.ptr, f64) -> ()
    %421 = call @rtclock() : () -> f64
    %422 = tosa.reduce_sum %417 {axis = 2 : i32} : (tensor<1x40x1536xf32>) -> tensor<1x40x1xf32>
    %423 = "tosa.const"() <{values = dense<1.536000e+03> : tensor<1xf32>}> : () -> tensor<1xf32>
    %424 = tosa.reciprocal %423 : (tensor<1xf32>) -> tensor<1xf32>
    %425 = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
    %426 = tosa.reshape %424, %425 : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
    %427 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %428 = tosa.mul %426, %422, %427 : (tensor<1x1x1xf32>, tensor<1x40x1xf32>, tensor<1xi8>) -> tensor<1x40x1xf32>
    %429 = call @rtclock() : () -> f64
    %430 = arith.subf %429, %421 : f64
    %431 = llvm.mlir.addressof @op_name_078_mean_1 : !llvm.ptr
    call @record_timing(%431, %430) : (!llvm.ptr, f64) -> ()
    %432 = call @rtclock() : () -> f64
    %433 = "tosa.const"() <{values = dense<9.99999997E-7> : tensor<1x40x1xf32>}> : () -> tensor<1x40x1xf32>
    %434 = tosa.add %428, %433 : (tensor<1x40x1xf32>, tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %435 = call @rtclock() : () -> f64
    %436 = arith.subf %435, %432 : f64
    %437 = llvm.mlir.addressof @op_name_079_add_3 : !llvm.ptr
    call @record_timing(%437, %436) : (!llvm.ptr, f64) -> ()
    %438 = call @rtclock() : () -> f64
    %439 = tosa.rsqrt %434 : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %440 = call @rtclock() : () -> f64
    %441 = arith.subf %440, %438 : f64
    %442 = llvm.mlir.addressof @op_name_080_rsqrt_1 : !llvm.ptr
    call @record_timing(%442, %441) : (!llvm.ptr, f64) -> ()
    %443 = call @rtclock() : () -> f64
    %444 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %445 = tosa.mul %411, %439, %444 : (tensor<1x40x1536xf32>, tensor<1x40x1xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>
    %446 = call @rtclock() : () -> f64
    %447 = arith.subf %446, %443 : f64
    %448 = llvm.mlir.addressof @op_name_081_mul_4 : !llvm.ptr
    call @record_timing(%448, %447) : (!llvm.ptr, f64) -> ()
    %449 = call @rtclock() : () -> f64
    %450 = tosa.const_shape  {values = dense<[1, 1, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %451 = tosa.reshape %arg9, %450 : (tensor<1536xf32>, !tosa.shape<3>) -> tensor<1x1x1536xf32>
    %452 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %453 = tosa.mul %451, %445, %452 : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>, tensor<1xi8>) -> tensor<1x40x1536xf32>
    %454 = call @rtclock() : () -> f64
    %455 = arith.subf %454, %449 : f64
    %456 = llvm.mlir.addressof @op_name_082_mul_5 : !llvm.ptr
    call @record_timing(%456, %455) : (!llvm.ptr, f64) -> ()
    %457 = call @rtclock() : () -> f64
    %458 = tosa.transpose %arg10 {perms = array<i32: 1, 0>} : (tensor<8960x1536xf32>) -> tensor<1536x8960xf32>
    %459 = call @rtclock() : () -> f64
    %460 = arith.subf %459, %457 : f64
    %461 = llvm.mlir.addressof @op_name_083_permute_9 : !llvm.ptr
    call @record_timing(%461, %460) : (!llvm.ptr, f64) -> ()
    %462 = call @rtclock() : () -> f64
    %463 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %464 = tosa.reshape %453, %463 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %465 = call @rtclock() : () -> f64
    %466 = arith.subf %465, %462 : f64
    %467 = llvm.mlir.addressof @op_name_084_view_21 : !llvm.ptr
    call @record_timing(%467, %466) : (!llvm.ptr, f64) -> ()
    %468 = call @rtclock() : () -> f64
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %469 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%464, %458 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_6 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %470 = call @rtclock() : () -> f64
    %471 = arith.subf %470, %468 : f64
    %472 = llvm.mlir.addressof @op_name_085_mm_4 : !llvm.ptr
    call @record_timing(%472, %471) : (!llvm.ptr, f64) -> ()
    %473 = call @rtclock() : () -> f64
    %474 = tosa.const_shape  {values = dense<[1, 40, 8960]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %475 = tosa.reshape %469, %474 : (tensor<40x8960xf32>, !tosa.shape<3>) -> tensor<1x40x8960xf32>
    %476 = call @rtclock() : () -> f64
    %477 = arith.subf %476, %473 : f64
    %478 = llvm.mlir.addressof @op_name_086_view_22 : !llvm.ptr
    call @record_timing(%478, %477) : (!llvm.ptr, f64) -> ()
    %479 = call @rtclock() : () -> f64
    %480 = tosa.sigmoid %475 : (tensor<1x40x8960xf32>) -> tensor<1x40x8960xf32>
    %481 = call @rtclock() : () -> f64
    %482 = arith.subf %481, %479 : f64
    %483 = llvm.mlir.addressof @op_name_087_sigmoid : !llvm.ptr
    call @record_timing(%483, %482) : (!llvm.ptr, f64) -> ()
    %484 = call @rtclock() : () -> f64
    %485 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %486 = tosa.mul %475, %480, %485 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>, tensor<1xi8>) -> tensor<1x40x8960xf32>
    %487 = call @rtclock() : () -> f64
    %488 = arith.subf %487, %484 : f64
    %489 = llvm.mlir.addressof @op_name_088_mul_6 : !llvm.ptr
    call @record_timing(%489, %488) : (!llvm.ptr, f64) -> ()
    %490 = call @rtclock() : () -> f64
    %491 = tosa.transpose %arg11 {perms = array<i32: 1, 0>} : (tensor<8960x1536xf32>) -> tensor<1536x8960xf32>
    %492 = call @rtclock() : () -> f64
    %493 = arith.subf %492, %490 : f64
    %494 = llvm.mlir.addressof @op_name_089_permute_10 : !llvm.ptr
    call @record_timing(%494, %493) : (!llvm.ptr, f64) -> ()
    %495 = call @rtclock() : () -> f64
    %496 = tosa.const_shape  {values = dense<[40, 1536]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %497 = tosa.reshape %453, %496 : (tensor<1x40x1536xf32>, !tosa.shape<2>) -> tensor<40x1536xf32>
    %498 = call @rtclock() : () -> f64
    %499 = arith.subf %498, %495 : f64
    %500 = llvm.mlir.addressof @op_name_090_view_23 : !llvm.ptr
    call @record_timing(%500, %499) : (!llvm.ptr, f64) -> ()
    %501 = call @rtclock() : () -> f64
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<40x8960xf32>
    %502 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%497, %491 : tensor<40x1536xf32>, tensor<1536x8960xf32>) outs(%cst_7 : tensor<40x8960xf32>) -> tensor<40x8960xf32>
    %503 = call @rtclock() : () -> f64
    %504 = arith.subf %503, %501 : f64
    %505 = llvm.mlir.addressof @op_name_091_mm_5 : !llvm.ptr
    call @record_timing(%505, %504) : (!llvm.ptr, f64) -> ()
    %506 = call @rtclock() : () -> f64
    %507 = tosa.const_shape  {values = dense<[1, 40, 8960]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %508 = tosa.reshape %502, %507 : (tensor<40x8960xf32>, !tosa.shape<3>) -> tensor<1x40x8960xf32>
    %509 = call @rtclock() : () -> f64
    %510 = arith.subf %509, %506 : f64
    %511 = llvm.mlir.addressof @op_name_092_view_24 : !llvm.ptr
    call @record_timing(%511, %510) : (!llvm.ptr, f64) -> ()
    %512 = call @rtclock() : () -> f64
    %513 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %514 = tosa.mul %486, %508, %513 : (tensor<1x40x8960xf32>, tensor<1x40x8960xf32>, tensor<1xi8>) -> tensor<1x40x8960xf32>
    %515 = call @rtclock() : () -> f64
    %516 = arith.subf %515, %512 : f64
    %517 = llvm.mlir.addressof @op_name_093_mul_7 : !llvm.ptr
    call @record_timing(%517, %516) : (!llvm.ptr, f64) -> ()
    %518 = call @rtclock() : () -> f64
    %519 = tosa.transpose %arg12 {perms = array<i32: 1, 0>} : (tensor<1536x8960xf32>) -> tensor<8960x1536xf32>
    %520 = call @rtclock() : () -> f64
    %521 = arith.subf %520, %518 : f64
    %522 = llvm.mlir.addressof @op_name_094_permute_11 : !llvm.ptr
    call @record_timing(%522, %521) : (!llvm.ptr, f64) -> ()
    %523 = call @rtclock() : () -> f64
    %524 = tosa.const_shape  {values = dense<[40, 8960]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %525 = tosa.reshape %514, %524 : (tensor<1x40x8960xf32>, !tosa.shape<2>) -> tensor<40x8960xf32>
    %526 = call @rtclock() : () -> f64
    %527 = arith.subf %526, %523 : f64
    %528 = llvm.mlir.addressof @op_name_095_view_25 : !llvm.ptr
    call @record_timing(%528, %527) : (!llvm.ptr, f64) -> ()
    %529 = call @rtclock() : () -> f64
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<40x1536xf32>
    %530 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%525, %519 : tensor<40x8960xf32>, tensor<8960x1536xf32>) outs(%cst_8 : tensor<40x1536xf32>) -> tensor<40x1536xf32>
    %531 = call @rtclock() : () -> f64
    %532 = arith.subf %531, %529 : f64
    %533 = llvm.mlir.addressof @op_name_096_mm_6 : !llvm.ptr
    call @record_timing(%533, %532) : (!llvm.ptr, f64) -> ()
    %534 = call @rtclock() : () -> f64
    %535 = tosa.const_shape  {values = dense<[1, 40, 1536]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %536 = tosa.reshape %530, %535 : (tensor<40x1536xf32>, !tosa.shape<3>) -> tensor<1x40x1536xf32>
    %537 = call @rtclock() : () -> f64
    %538 = arith.subf %537, %534 : f64
    %539 = llvm.mlir.addressof @op_name_097_view_26 : !llvm.ptr
    call @record_timing(%539, %538) : (!llvm.ptr, f64) -> ()
    %540 = call @rtclock() : () -> f64
    %541 = tosa.add %411, %536 : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x1536xf32>
    %542 = call @rtclock() : () -> f64
    %543 = arith.subf %542, %540 : f64
    %544 = llvm.mlir.addressof @op_name_098_add_4 : !llvm.ptr
    call @record_timing(%544, %543) : (!llvm.ptr, f64) -> ()
    %545 = call @rtclock() : () -> f64
    %546 = call @rtclock() : () -> f64
    %547 = arith.subf %546, %545 : f64
    %548 = llvm.mlir.addressof @op_name_099_output : !llvm.ptr
    call @record_timing(%548, %547) : (!llvm.ptr, f64) -> ()
    %549 = call @rtclock() : () -> f64
    %550 = call @rtclock() : () -> f64
    %551 = arith.subf %550, %549 : f64
    %552 = llvm.mlir.addressof @op_name_100_output : !llvm.ptr
    call @record_timing(%552, %551) : (!llvm.ptr, f64) -> ()
    return %541 : tensor<1x40x1536xf32>
  }
}

