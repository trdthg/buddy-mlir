# Define bufferize options as variables to avoid CMake quote parsing issues
set(BUFFERIZE_FULL_OPTS "unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries")
set(BUFFERIZE_SIMPLE_OPTS "bufferize-function-boundaries")
set(TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

if(IS_RVV_CROSSCOMPILING)
  if(NOT DEFINED BUDDY_MLIR_BUILD_DIR OR "${BUDDY_MLIR_BUILD_DIR}" STREQUAL "")
    message(FATAL_ERROR "BuddyQwen3 cross-compiling requires -DBUDDY_MLIR_BUILD_DIR=...")
  endif()
  set(CMAKE_C_COMPILER ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin/clang)
  set(CMAKE_CXX_COMPILER ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin/clang++)
  foreach(REQUIRED_VAR RISCV_GNU_TOOLCHAIN RISCV_MLIR_C_RUNNER_UTILS RISCV_OMP_SHARED)
    if(NOT DEFINED ${REQUIRED_VAR} OR "${${REQUIRED_VAR}}" STREQUAL "")
      message(FATAL_ERROR "BuddyQwen3 cross-compiling requires -D${REQUIRED_VAR}=...")
    endif()
  endforeach()
  if(NOT EXISTS "${CMAKE_C_COMPILER}" OR NOT EXISTS "${CMAKE_CXX_COMPILER}")
    message(FATAL_ERROR
      "BuddyQwen3 cross-compiling requires clang/clang++ under "
      "${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin.")
  endif()
endif()

# Detect RISC-V Vector (RVV) platform
# Use HAVE_LOCAL_RVV from check_simd.cmake which is already called in the main CMakeLists.txt
if(HAVE_LOCAL_RVV)
  set(IS_RVV_PLATFORM ON)
  message(STATUS "RISC-V Vector (RVV) platform detected")
endif()

# Set platform-specific variables
set(BUDDY_QWEN3_DEFAULT_OPENMP_THREADS 48)
set(BUDDY_QWEN3_DEFAULT_BLIS_VECTOR_SIZE 32)
set(BUDDY_QWEN3_DEFAULT_BLIS_PARALLEL_OUTER_LOOP ON)
set(BUDDY_QWEN3_DEFAULT_BATCH_TRANSPOSE_VECTOR_SIZE 32)
set(BUDDY_QWEN3_DEFAULT_PREFILL_BLOCK_SIZE_Q 16)
set(BUDDY_QWEN3_DEFAULT_PREFILL_BLOCK_SIZE_KV 64)
set(BUDDY_QWEN3_DEFAULT_PREFILL_AFFINE_PARALLEL_MAX_NESTED "")
if(IS_RVV_PLATFORM)
  set(BUDDY_QWEN3_DEFAULT_OPENMP_THREADS 32)
  set(BUDDY_QWEN3_DEFAULT_BLIS_VECTOR_SIZE 8)
  set(BUDDY_QWEN3_DEFAULT_BATCH_TRANSPOSE_VECTOR_SIZE 16)
  set(BUDDY_QWEN3_DEFAULT_PREFILL_AFFINE_PARALLEL_MAX_NESTED 1)
  set(LLC_RISCV_ATTRS "-mattr=+m,+d,+v")
  set(RISCV_COMPILE_OPTS "-march=rv64gcv;-mabi=lp64d;-O3;-Wall")
  set(RISCV_LINK_OPTS "-march=rv64gcv;-mabi=lp64d")
else()
  set(LLC_RISCV_ATTRS "")
endif()

if(IS_RVV_CROSSCOMPILING)
  message(STATUS "Cross compiling BuddyQwen3 for RISC-V Vector (RVV)")
  set(BUDDY_QWEN3_DEFAULT_OPENMP_THREADS 8)
  set(BUDDY_QWEN3_DEFAULT_BLIS_VECTOR_SIZE 8)
  set(BUDDY_QWEN3_DEFAULT_BATCH_TRANSPOSE_VECTOR_SIZE 16)
  set(BUDDY_QWEN3_DEFAULT_PREFILL_AFFINE_PARALLEL_MAX_NESTED 1)
  set(LLC_RISCV_ATTRS "-march=riscv64;-mattr=+m,+d,+v")
  set(RISCV_COMPILE_OPTS
    -march=rv64gcv
    -mabi=lp64d
    --target=riscv64-unknown-linux-gnu
    --sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot
    --gcc-toolchain=${RISCV_GNU_TOOLCHAIN}
    -fPIC
  )
  set(RISCV_LINK_OPTS
    -march=rv64gcv
    -mabi=lp64d
    --target=riscv64-unknown-linux-gnu
    --sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot
    --gcc-toolchain=${RISCV_GNU_TOOLCHAIN}
  )
endif()

set(BUDDY_QWEN3_OPENMP_THREADS "${BUDDY_QWEN3_DEFAULT_OPENMP_THREADS}" CACHE STRING
  "Number of OpenMP threads encoded into the BuddyQwen3 lowering pipeline")
set(BUDDY_QWEN3_BLIS_VECTOR_SIZE "${BUDDY_QWEN3_DEFAULT_BLIS_VECTOR_SIZE}" CACHE STRING
  "Vector width used by matmul-vectorization-blis for BuddyQwen3")
set(BUDDY_QWEN3_BLIS_PARALLEL_OUTER_LOOP "${BUDDY_QWEN3_DEFAULT_BLIS_PARALLEL_OUTER_LOOP}" CACHE BOOL
  "Whether BuddyQwen3 should parallelize the outer jc loop in matmul-vectorization-blis")
set(BUDDY_QWEN3_BATCH_TRANSPOSE_VECTOR_SIZE "${BUDDY_QWEN3_DEFAULT_BATCH_TRANSPOSE_VECTOR_SIZE}" CACHE STRING
  "Vector width used by batchmatmul-transpose-b-vectorization for BuddyQwen3")
set(BUDDY_QWEN3_PREFILL_BLOCK_SIZE_Q "${BUDDY_QWEN3_DEFAULT_PREFILL_BLOCK_SIZE_Q}" CACHE STRING
  "Flash-attention prefill tile size along the query sequence for BuddyQwen3")
set(BUDDY_QWEN3_PREFILL_BLOCK_SIZE_KV "${BUDDY_QWEN3_DEFAULT_PREFILL_BLOCK_SIZE_KV}" CACHE STRING
  "Flash-attention prefill tile size along the key/value sequence for BuddyQwen3")
set(BUDDY_QWEN3_PREFILL_AFFINE_PARALLEL_MAX_NESTED
  "${BUDDY_QWEN3_DEFAULT_PREFILL_AFFINE_PARALLEL_MAX_NESTED}" CACHE STRING
  "Maximum nested affine.parallel depth for BuddyQwen3 prefill; empty keeps MLIR defaults")
if(BUDDY_QWEN3_BLIS_PARALLEL_OUTER_LOOP)
  set(BUDDY_QWEN3_BLIS_PARALLEL_OUTER_LOOP_OPT true)
else()
  set(BUDDY_QWEN3_BLIS_PARALLEL_OUTER_LOOP_OPT false)
endif()
set(BUDDY_QWEN3_BLIS_PASS_OPTS
  "vector-size=${BUDDY_QWEN3_BLIS_VECTOR_SIZE} parallel-outer-loop=${BUDDY_QWEN3_BLIS_PARALLEL_OUTER_LOOP_OPT}")
set(OPENMP_NUM_THREADS "num-threads=${BUDDY_QWEN3_OPENMP_THREADS}")
set(QWEN3_IMPORT_PYTHONPATH
  "${LLVM_BINARY_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_PYTHON_PACKAGES_DIR}:$ENV{PYTHONPATH}")
set(QWEN3_IMPORT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/import-qwen3.py)
if(TARGET python-package-buddy)
  list(APPEND QWEN3_IMPORT_DEPENDS python-package-buddy)
endif()
if(TARGET BuddyMLIRPythonModules)
  list(APPEND QWEN3_IMPORT_DEPENDS BuddyMLIRPythonModules)
endif()
if(BUDDY_QWEN3_PREFILL_AFFINE_PARALLEL_MAX_NESTED STREQUAL "")
  set(BUDDY_QWEN3_PREFILL_AFFINE_PARALLELIZE_PASS -affine-parallelize)
else()
  set(BUDDY_QWEN3_PREFILL_AFFINE_PARALLELIZE_PASS
    -affine-parallelize=max-nested=${BUDDY_QWEN3_PREFILL_AFFINE_PARALLEL_MAX_NESTED})
endif()

# Skip import commands on RVV platform
if(NOT IS_RVV_PLATFORM)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_0_6b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_0_6b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_0_6b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_0_6b.mlir
           ${CMAKE_CURRENT_BINARY_DIR}/arg0_0_6b.data
    COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${QWEN3_IMPORT_PYTHONPATH}"
            BUDDY_QWEN3_PREFILL_BLOCK_SIZE_Q=${BUDDY_QWEN3_PREFILL_BLOCK_SIZE_Q}
            BUDDY_QWEN3_PREFILL_BLOCK_SIZE_KV=${BUDDY_QWEN3_PREFILL_BLOCK_SIZE_KV}
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/import-qwen3.py
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}
            --precision f32
    DEPENDS ${QWEN3_IMPORT_DEPENDS}
    COMMENT "Generating forward_prefill.mlir, subgraph0_prefill.mlir and arg0.data..."
  )
endif()

add_custom_command(
  OUTPUT forward_prefill_0_6b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_0_6b.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -matmul-vectorization-blis=${BUDDY_QWEN3_BLIS_PASS_OPTS}
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            ${BUDDY_QWEN3_PREFILL_AFFINE_PARALLELIZE_PASS}
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_0_6b.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_prefill_0_6b.mlir
  COMMENT "Building forward_prefill.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_prefill_0_6b.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_0_6b.mlir
              -simplify-tosa-reshape |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -matmul-vectorization-blis=${BUDDY_QWEN3_BLIS_PASS_OPTS}
            -batchmatmul-optimize
            -batchmatmul-transpose-b-vectorization=vector-size=${BUDDY_QWEN3_BATCH_TRANSPOSE_VECTOR_SIZE}
            -convert-linalg-to-affine-loops
            ${BUDDY_QWEN3_PREFILL_AFFINE_PARALLELIZE_PASS}
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_prefill_0_6b.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_prefill_0_6b.mlir
    COMMENT "Building subgraph_prefill.o "
    VERBATIM)

add_custom_command(
  OUTPUT forward_decode_0_6b.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_0_6b.mlir
            -simplify-tosa-reshape |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -pass-pipeline ${TOSA_PIPELINE} |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -matmul-vectorization-blis=${BUDDY_QWEN3_BLIS_PASS_OPTS}
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-parallelize
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
        ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
        ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
          -o ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_0_6b.o
  DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/forward_decode_0_6b.mlir
  COMMENT "Building forward_decode.o "
  VERBATIM)

add_custom_command(
    OUTPUT subgraph_decode_0_6b.o
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_0_6b.mlir
              -simplify-tosa-reshape
              -simplify-tosa-matmul-scalar |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline ${TOSA_PIPELINE} |
            ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -convert-elementwise-to-linalg
            -one-shot-bufferize=${BUFFERIZE_SIMPLE_OPTS}
            -expand-strided-metadata
            -ownership-based-buffer-deallocation
            -canonicalize
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -cse
            -canonicalize
            -optimize-allocation-liveness
            -eliminate-memref-copy
            -assume-tight-memref-layout
            -staticize-memref-layout
            -matmul-vectorization-decode=vector-size=128
            -batch-matmul-vectorization-decode=vector-size=128
            -batchmatmul-transpose-b-vectorization=vector-size=16
            -convert-linalg-to-affine-loops
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-openmp=${OPENMP_NUM_THREADS}
            -cse
            -memref-expand
            -arith-expand
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc ${LLC_RISCV_ATTRS} -filetype=obj -relocation-model=pic -O3
            -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph_decode_0_6b.o
    DEPENDS buddy-opt ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_decode_0_6b.mlir
    COMMENT "Building subgraph_decode.o "
    VERBATIM)


add_library(QWEN3_0_6B STATIC forward_prefill_0_6b.o subgraph_prefill_0_6b.o forward_decode_0_6b.o subgraph_decode_0_6b.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  QWEN3_0_6B
  PROPERTIES
  LINKER_LANGUAGE C)


add_executable(buddy-qwen3-0.6b-run buddy-qwen3-0.6b-main.cpp)

if(IS_RVV_CROSSCOMPILING)
  if(NOT DEFINED QWEN3_0_6B_EXAMPLE_PATH OR "${QWEN3_0_6B_EXAMPLE_PATH}" STREQUAL "")
    set(QWEN3_0_6B_EXAMPLE_PATH ".")
  endif()
  if(NOT DEFINED QWEN3_0_6B_EXAMPLE_BUILD_PATH OR "${QWEN3_0_6B_EXAMPLE_BUILD_PATH}" STREQUAL "")
    set(QWEN3_0_6B_EXAMPLE_BUILD_PATH ".")
  endif()
  set(QWEN3_0_6B_EXAMPLE_PATH_RESOLVED "${QWEN3_0_6B_EXAMPLE_PATH}")
  set(QWEN3_0_6B_EXAMPLE_BUILD_PATH_RESOLVED "${QWEN3_0_6B_EXAMPLE_BUILD_PATH}")
else()
  set(QWEN3_0_6B_EXAMPLE_PATH_RESOLVED ${CMAKE_CURRENT_SOURCE_DIR})
  set(QWEN3_0_6B_EXAMPLE_BUILD_PATH_RESOLVED ${CMAKE_CURRENT_BINARY_DIR})
endif()

target_compile_definitions(buddy-qwen3-0.6b-run PRIVATE
  QWEN3_0_6B_EXAMPLE_PATH="${QWEN3_0_6B_EXAMPLE_PATH_RESOLVED}/"
  QWEN3_0_6B_EXAMPLE_BUILD_PATH="${QWEN3_0_6B_EXAMPLE_BUILD_PATH_RESOLVED}/"
)

if(IS_RVV_PLATFORM OR IS_RVV_CROSSCOMPILING)
  target_compile_options(buddy-qwen3-0.6b-run PRIVATE ${RISCV_COMPILE_OPTS})
  target_link_options(buddy-qwen3-0.6b-run PRIVATE ${RISCV_LINK_OPTS})
endif()

if(NOT IS_RVV_CROSSCOMPILING)
  target_link_directories(buddy-qwen3-0.6b-run PRIVATE ${LLVM_LIBRARY_DIR})
endif()

if(IS_RVV_CROSSCOMPILING)
  set(BUDDY_QWEN3_0_6B_LIBS
    QWEN3_0_6B
    ${RISCV_MLIR_C_RUNNER_UTILS}
    ${RISCV_OMP_SHARED}
  )
else()
  set(BUDDY_QWEN3_0_6B_LIBS
    QWEN3_0_6B
    mlir_c_runner_utils
    omp
  )
endif()

if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_QWEN3_0_6B_LIBS mimalloc)
endif()

target_link_libraries(buddy-qwen3-0.6b-run ${BUDDY_QWEN3_0_6B_LIBS})

if(IS_RVV_CROSSCOMPILING)
  get_filename_component(RISCV_MLIR_C_RUNNER_UTILS_DIR ${RISCV_MLIR_C_RUNNER_UTILS} DIRECTORY)
  get_filename_component(RISCV_OMP_SHARED_DIR ${RISCV_OMP_SHARED} DIRECTORY)
  set(RISCV_MLIR_FLOAT16_UTILS ${RISCV_MLIR_C_RUNNER_UTILS_DIR}/libmlir_float16_utils.so.22.0git)
  set(RISCV_MLIR_RUNNER_UTILS ${RISCV_MLIR_C_RUNNER_UTILS_DIR}/libmlir_runner_utils.so.22.0git)
  if(NOT EXISTS ${RISCV_MLIR_FLOAT16_UTILS})
    message(FATAL_ERROR
      "BuddyQwen3 cross-compiling package requires ${RISCV_MLIR_FLOAT16_UTILS}.")
  endif()
  set_target_properties(buddy-qwen3-0.6b-run PROPERTIES
    BUILD_RPATH "\$ORIGIN;${RISCV_MLIR_C_RUNNER_UTILS_DIR};${RISCV_OMP_SHARED_DIR}"
    INSTALL_RPATH "\$ORIGIN"
  )

  set(QWEN3_RVV_PKG_NAME "buddy-qwen3-0.6b-rvv-package")
  set(QWEN3_RVV_PKG_DIR ${CMAKE_CURRENT_BINARY_DIR}/${QWEN3_RVV_PKG_NAME})
  set(QWEN3_RVV_PKG_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/${QWEN3_RVV_PKG_NAME}.tgz)
  set(QWEN3_RVV_PKG_ARCHIVE_TMP ${QWEN3_RVV_PKG_ARCHIVE}.tmp)

  add_custom_target(buddy-qwen3-0.6b-rvv-package DEPENDS ${QWEN3_RVV_PKG_ARCHIVE})
  add_custom_command(
    OUTPUT ${QWEN3_RVV_PKG_ARCHIVE}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${QWEN3_RVV_PKG_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/arg0_0_6b.data
      ${CMAKE_BINARY_DIR}/bin/buddy-qwen3-0.6b-run
      ${CMAKE_SOURCE_DIR}/examples/BuddyQwen3/vocab.txt
      ${RISCV_MLIR_C_RUNNER_UTILS}
      ${RISCV_MLIR_FLOAT16_UTILS}
      ${RISCV_MLIR_RUNNER_UTILS}
      ${RISCV_OMP_SHARED}
      ${QWEN3_RVV_PKG_DIR}/
    COMMAND ${CMAKE_COMMAND} -E rm -f ${QWEN3_RVV_PKG_ARCHIVE_TMP}
    COMMAND sh -c "tar -czf ${QWEN3_RVV_PKG_ARCHIVE_TMP} ${QWEN3_RVV_PKG_NAME}"
    COMMAND ${CMAKE_COMMAND} -E rename ${QWEN3_RVV_PKG_ARCHIVE_TMP} ${QWEN3_RVV_PKG_ARCHIVE}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${QWEN3_RVV_PKG_DIR}
    COMMENT "Creating RVV package: ${QWEN3_RVV_PKG_ARCHIVE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/arg0_0_6b.data
      ${CMAKE_BINARY_DIR}/bin/buddy-qwen3-0.6b-run
      ${CMAKE_SOURCE_DIR}/examples/BuddyQwen3/vocab.txt
      ${RISCV_MLIR_C_RUNNER_UTILS}
      ${RISCV_MLIR_FLOAT16_UTILS}
      ${RISCV_MLIR_RUNNER_UTILS}
      ${RISCV_OMP_SHARED}
    USES_TERMINAL
    VERBATIM
  )
endif()
