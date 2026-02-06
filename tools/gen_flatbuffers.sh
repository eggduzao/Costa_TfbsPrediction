#!/bin/bash
ROOT=$(pwd)
FF_LOCATION="$ROOT/third_party/flatbuffers"
cd "$FF_LOCATION" || exit
mkdir build
cd build || exit
cmake ..
cmake --build . --target flatc
mkdir -p "$ROOT/build/smith/csrc/jit/serialization"
./flatc --cpp --gen-mutable --scoped-enums \
     -o "$ROOT/smith/csrc/jit/serialization" \
     -c "$ROOT/smith/csrc/jit/serialization/mobile_bytecode.fbs"
echo '// @generated' >> "$ROOT/smith/csrc/jit/serialization/mobile_bytecode_generated.h"
cd "$ROOT" || exit
exit
