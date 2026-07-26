#!/usr/bin/env bash
# Build the C bridge shared library for the llama.cpp backbone.
#
# Usage:
#   bash build_bridge.sh /path/to/llama.cpp
#
# Prerequisites:
#   1. llama.cpp must be compiled first (cmake --build build --config Release)
#   2. GCC or Clang with -shared/-fPIC support
#
# The output library (libbackbone_bridge.so) is placed in the same
# directory as this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_SRC="${SCRIPT_DIR}/backbone_bridge.c"
LIB_OUT="${SCRIPT_DIR}/libbackbone_bridge.so"

if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/llama.cpp"
    echo ""
    echo "  llama.cpp must be already built (cmake --build build --config Release)"
    exit 1
fi

LLAMA_CPP_DIR="$(cd "$1" && pwd)"

INCLUDE_DIRS=(
    "-I${LLAMA_CPP_DIR}/include"
    "-I${LLAMA_CPP_DIR}/ggml/include"
)

LIB_DIR="${LLAMA_CPP_DIR}/build/bin"
if [ ! -d "${LIB_DIR}" ]; then
    LIB_DIR="${LLAMA_CPP_DIR}/build/src"
fi
if [ ! -d "${LIB_DIR}" ]; then
    LIB_DIR="${LLAMA_CPP_DIR}/build"
fi

echo "Bridge source:  ${BRIDGE_SRC}"
echo "llama.cpp dir:  ${LLAMA_CPP_DIR}"
echo "Library dir:    ${LIB_DIR}"
echo "Output:         ${LIB_OUT}"
echo ""

gcc -shared -fPIC -O2 \
    -o "${LIB_OUT}" \
    "${BRIDGE_SRC}" \
    "${INCLUDE_DIRS[@]}" \
    -L"${LIB_DIR}" \
    -lllama \
    -Wl,-rpath,"${LIB_DIR}"

echo "Successfully built: ${LIB_OUT}"
echo "Verify with: ldd ${LIB_OUT}"
