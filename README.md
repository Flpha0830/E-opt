# E-Opt

## Building

This setup assumes that you build the project as part of a monolithic LLVM build via the `LLVM_EXTERNAL_PROJECTS` mechanism.
To build LLVM, MLIR and the project
```sh
mkdir build && cd build
cmake -G "Unix Makefiles" `$LLVM_SRC_DIR/llvm` \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_EXTERNAL_PROJECTS=toy-dialect -DLLVM_EXTERNAL_TOY_DIALECT_SOURCE_DIR=../
cmake --build .
```
Here, `$LLVM_SRC_DIR` needs to point to the root of the monorepo.

## Testing
```sh
./build/bin/e-opt --e-graph rewrite-test/Tosa/randomTest1.mlir
```

## Random Test Generate
```sh
python3 ./genTest/main.py
```
