#!/bin/bash
set -e
# MicroGAN Vertical Slice Pipeline Runner (macOS Ultra-Safe Mode)

# 1. Threading and Mutex Safeguards
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1

# 2. Logging and GPU Safeguards
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""
export TF_ENABLE_ONEDNN_OPTS=0

# Activate environment
source venv_compat/bin/activate

echo "--- Phase 1: Training (PyTorch) ---"
microgan train --epochs 1 --output-dir build_vertical

echo "--- Phase 2a: Export to ONNX (PyTorch) ---"
microgan export-onnx --checkpoint build_vertical/generator_final.pt --output-dir build_vertical

echo "--- Phase 2b: Convert ONNX to TFLite (TensorFlow) ---"
# We run this in its own subshell to be extra safe
(
  export OMP_NUM_THREADS=1
  microgan onnx-to-tflite --onnx-path build_vertical/generator.onnx --output-dir build_vertical
)

echo "--- Phase 3: Header Conversion (TensorFlow) ---"
microgan convert --tflite build_vertical/generator_quantized.tflite --output-dir build_vertical

echo "--- Pipeline Complete ---"
