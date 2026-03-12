# 交叉编译

conda activate buddy-mlir-py310

export WORKSPACE=/home/yanmingzhu/repo/ruyi-ai/worktree/trdthg/feature/cross-compile/
export BUDDY_MLIR_BUILD_DIR=$WORKSPACE/build
export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
export BUILD_LOCAL_LLVM_DIR=$WORKSPACE/llvm/build
export BUILD_RISCV_GNU_TOOLCHAIN_DIR=$WORKSPACE/build/thirdparty/riscv-gnu-toolchain/
export RISCV_GNU_TOOLCHAIN_SYSROOT_DIR=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}/sysroot/
export BUILD_CROSS_MLIR_DIR=$WORKSPACE/llvm/build-cross-mlir-rv
export RISCV_MLIR_C_RUNNER_UTILS=${BUILD_CROSS_MLIR_DIR}/lib/libmlir_c_runner_utils.so.22.0git
export BUDDY_DEEPSEEKR1_LLVMOPT=${BUILD_CROSS_MLIR_DIR}/lib/libLLVMSupport.a
export RISCV_OMP_SHARED=${BUILD_LOCAL_LLVM_DIR}/../build-omp-shared-rv/libomp.so

## 1. Build Local LLVM/MLIR

cd $WORKSPACE/llvm/build

cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_ENABLE_RUNTIMES=openmp \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
ninja check-clang check-mlir openmp
export BUILD_LOCAL_LLVM_DIR=$WORKSPACE/llvm/build


## 2. Build Local buddy-mlir

cd $WORKSPACE/build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_RISCV_GNU_TOOLCHAIN=ON \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
ninja
ninja check-buddy
export BUILD_RISCV_GNU_TOOLCHAIN_DIR=$WORKSPACE/build/thirdparty/riscv-gnu-toolchain/
export RISCV_GNU_TOOLCHAIN_SYSROOT_DIR=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}/sysroot/

## 3. Build Cross-Compiled MLIR

mkdir -p $WORKSPACE/llvm/build-cross-mlir-rv
cd $WORKSPACE/llvm/build-cross-mlir-rv

cmake -G Ninja ../../llvm/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DCMAKE_CROSSCOMPILING=True \
    -DLLVM_TARGET_ARCH=RISCV64 \
    -DLLVM_TARGETS_TO_BUILD=RISCV \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_NATIVE_ARCH=RISCV \
    -DLLVM_HOST_TRIPLE=riscv64-unknown-linux-gnu \
    -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
    -DCMAKE_C_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang \
    -DCMAKE_CXX_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang++ \
    -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
    -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
    -DMLIR_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-tblgen \
    -DLLVM_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/llvm-tblgen \
    -DMLIR_SRC_SHARDER_TABLEGEN_EXE=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-src-sharder \
    -DMLIR_LINALG_ODS_YAML_GEN=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-linalg-ods-yaml-gen \
    -DMLIR_PDLL_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-pdll \
    -DLLVM_ENABLE_ZSTD=Off \
    -DMLIR_IRDL_TO_CPP_EXE=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-irdl-to-cpp
ninja
export BUILD_CROSS_MLIR_DIR=$WORKSPACE/llvm/build-cross-mlir-rv
export RISCV_MLIR_C_RUNNER_UTILS=${BUILD_CROSS_MLIR_DIR}/lib/libmlir_c_runner_utils.so.22.0git
export BUDDY_DEEPSEEKR1_LLVMOPT=${BUILD_CROSS_MLIR_DIR}/lib/libLLVMSupport.a

cd ${BUILD_LOCAL_LLVM_DIR}/../
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XEsAhOcMioN9gdufuyO9OrHIdR0UtHh2' -O build-omp-shared-rv.tar.gz
mkdir build-omp-shared-rv && tar -xzf build-omp-shared-rv.tar.gz -C build-omp-shared-rv && rm build-omp-shared-rv.tar.gz
export RISCV_OMP_SHARED=${BUILD_LOCAL_LLVM_DIR}/../build-omp-shared-rv/libomp.so

## Build for the target platform

cd $WORKSPACE/build

cmake -G Ninja .. \
    -DRISCV_GNU_TOOLCHAIN=${BUILD_RISCV_GNU_TOOLCHAIN_DIR} \
    -DDEEPSEEKR1_EXAMPLE_PATH=. \
    -DDEEPSEEKR1_EXAMPLE_BUILD_PATH=. \
    -DBUDDY_DEEPSEEKR1_EXAMPLES=ON \
    -DBUDDY_QWEN3_EXAMPLES=OFF \
    -DIS_RVV_CROSSCOMPILING=ON \
    -DRISCV_OMP_SHARED=${RISCV_OMP_SHARED} \
    -DRISCV_MLIR_C_RUNNER_UTILS=${RISCV_MLIR_C_RUNNER_UTILS} \
    -DBUILD_CROSS_MLIR_DIR=${BUILD_CROSS_MLIR_DIR} \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}
ninja buddy-deepseek-r1-run
ninja buddy-deepseek-r1-rvv-package

scp ./examples/BuddyDeepSeekR1/buddy-deepseek-r1-rvv-package.tgz k1:/home/work/repo/cross-compile

## Build BuddyQwen3 for the target platform

`BuddyQwen3` 现在和 `BuddyDeepSeekR1` 一样，在 `IS_RVV_CROSSCOMPILING=ON` 时会自动切到本机 LLVM 的 `clang/clang++`，不需要额外清理 cache，也不需要在命令前手动加 `CC/CXX`。

cd $WORKSPACE/build

cmake -G Ninja .. \
    -DRISCV_GNU_TOOLCHAIN=${BUILD_RISCV_GNU_TOOLCHAIN_DIR} \
    -DQWEN3_0_6B_EXAMPLE_PATH=. \
    -DQWEN3_0_6B_EXAMPLE_BUILD_PATH=. \
    -DBUDDY_QWEN3_EXAMPLES=ON \
    -DBUDDY_DEEPSEEKR1_EXAMPLES=OFF \
    -DIS_RVV_CROSSCOMPILING=ON \
    -DRISCV_OMP_SHARED=${RISCV_OMP_SHARED} \
    -DRISCV_MLIR_C_RUNNER_UTILS=${RISCV_MLIR_C_RUNNER_UTILS} \
    -DBUDDY_QWEN3_OPENMP_THREADS=8 \
    -DBUILD_CROSS_MLIR_DIR=${BUILD_CROSS_MLIR_DIR} \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}

ninja buddy-qwen3-0.6b-run
ninja buddy-qwen3-0.6b-rvv-package

scp ./examples/BuddyQwen3/buddy-qwen3-0.6b-rvv-package.tgz k1:/home/work/repo/cross-compile
