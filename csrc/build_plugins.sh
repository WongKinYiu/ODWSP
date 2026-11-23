#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Error: nvcc not found in PATH. Install CUDA Toolkit >= 12.0." >&2
  exit 1
fi

GENCODE_FLAGS=(
  -gencode arch=compute_80,code=sm_80
  -gencode arch=compute_86,code=sm_86
  -gencode arch=compute_89,code=sm_89
  -gencode arch=compute_90,code=sm_90
)

if nvcc --help 2>/dev/null | grep -q "compute_120"; then
  GENCODE_FLAGS+=(-gencode arch=compute_120,code=sm_120)
fi

echo "[1/2] Building libwsp_portable_codesign.so ..."
nvcc -O3 --use_fast_math "${GENCODE_FLAGS[@]}" \
  --shared -Xcompiler -fPIC -lcublas \
  wsp_portable_codesign.cu -o libwsp_portable_codesign.so

echo "[2/2] Building libwsp_sm120_codesign.so ..."
nvcc -O3 --use_fast_math "${GENCODE_FLAGS[@]}" \
  --shared -Xcompiler -fPIC -lcublas \
  wsp_sm120_codesign.cu -o libwsp_sm120_codesign.so

echo "Successfully built custom CUDA C++ TensorRT IPluginV3 libraries:"
ls -lh libwsp_*.so
