#!/bin/sh

if [ -z "$CXX" ]; then
    CXX="clang++"
    echo "Using system default C++ compiler: $CXX"
else
    echo "Using user-provided C++ compiler: $CXX"
fi

if [ -z "$SMITH_ROOT_DIR" ]; then
    echo "Error: The SMITH_ROOT_DIR environment variable must be set." >&2
    echo "Example: export SMITH_ROOT_DIR=/home/$USER/local/blacksmith" >&2
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input file path> <output file path>."
    echo "Example Usage: $0 standalone_test.cpp standalone_test.out."
    exit 1
fi

# Building the wrapper
$CXX -I$SMITH_ROOT_DIR/build/aten/src -I$SMITH_ROOT_DIR/aten/src -I$SMITH_ROOT_DIR/build -I$SMITH_ROOT_DIR -I$SMITH_ROOT_DIR/build/caffe2/aten/src -I$SMITH_ROOT_DIR/smith/csrc/api -I$SMITH_ROOT_DIR/smith/csrc/api/include -std=gnu++17 -fPIE -o $1.o -c $1

# Linking
$CXX -rdynamic -Wl,--no-as-needed,$SMITH_ROOT_DIR/build/lib/libsmith.so $1.o -Wl,--no-as-needed,$SMITH_ROOT_DIR/build/lib/libsmith_cpu.so -Wl,--no-as-needed,$SMITH_ROOT_DIR/build/lib/libc10.so -o $2
