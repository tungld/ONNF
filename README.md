# ONNF
Open Neural Network Frontend : an ONNX frontend for MLIR.

[![CircleCI](https://circleci.com/gh/clang-ykt/ONNF.svg?style=svg)](https://circleci.com/gh/clang-ykt/ONNF)

## Installation

Firstly, install MLIR (as a part of LLVM-Project):

[same-as-file]: <> (utils/install-mlir.sh)
``` bash
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON

cmake --build . --target
cmake --build . --target check-mlir
```

Two environment variables need to be set:
- LLVM_PROJ_SRC should point to the llvm-project src directory (e.g., llvm-project/).
- LLVM_PROJ_BUILD should point to the llvm-project build directory (e.g., llvm-project/build).

To build ONNF, use the following command:

[same-as-file]: <> ({"ref": "utils/install-onnf.sh", "skip-doc": 2})
```
git clone --recursive git@github.com:clang-ykt/ONNF.git

# Export environment variables pointing to LLVM-Projects.
export LLVM_PROJ_SRC=$(pwd)/llvm-project/
export LLVM_PROJ_BUILD=$(pwd)/llvm-project/build

mkdir ONNF/build && cd ONNF/build
cmake ..
cmake --build . --target onnf

# Run FileCheck tests:
export LIT_OPTS=-v
cmake --build . --target check-mlir-lit
```

After the above commands succeed, an `onnf` executable should appear in the `bin` directory. 

## Using ONNF

The usage of `onnf` is as such:
```
OVERVIEW: ONNF MLIR modular optimizer driver

USAGE: onnf [options] <input file>

OPTIONS:

Generic Options:

  --help        - Display available options (--help-hidden for more)
  --help-list   - Display list of available options (--help-list-hidden for more)
  --version     - Display the version of this program

ONNF Options:
These are frontend options.

  Choose target to emit:
      --EmitONNXIR - Ingest ONNX and emit corresponding ONNX dialect.
      --EmitMLIR   - Lower model to MLIR built-in transformation dialect.
      --EmitLLVMIR - Lower model to LLVM IR (LLVM dialect).
      --EmitLLVMBC - Lower model to LLVM IR and emit (to file) LLVM bitcode for model.
```

## Example

For example, to lower an ONNX model (e.g., add.onnx) to ONNX dialect, use the following command:
```
./onnf --EmitONNXIR add.onnx
```
The output should look like:
```
module {
  func @main_graph(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
    return %0 : tensor<10x10x10xf32>
  }
}
```
