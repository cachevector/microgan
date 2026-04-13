# Contributing to MicroGAN

This document details the project structure and technical stack for those who want to contribute to the codebase.

## Technical Stack

*   Python 3.9+ for the training and conversion pipeline.
*   PyTorch for model definition and training.
*   ONNX and TFLite for model optimization and quantization.
*   onnx2tf for stable ONNX to TFLite conversion on macOS/Linux.
*   C99 for the embedded inference runtime.

## Project Structure

*   tinygen/: The main Python CLI and library.
    *   train/: GAN architectures (DCGAN, StyleGAN) and training logic.
    *   compress/: Model quantization and pruning tools.
    *   convert/: ONNX, TFLite, and C header export utilities.
    *   validate/: PC-side C runtime testing and validation.
    *   hardware/: Hardware profiles and flashing tools.
*   runtime/: The C/C++ source for the on-device inference engine.
    *   backends/: Optimized kernels for specific architectures (CMSIS-NN, Xtensa).
*   firmware_templates/: Scaffolding for different boards and frameworks (PlatformIO, Arduino).
*   model_zoo/: Pre-trained weights for various image domains.
*   datasets/: Scripts for downloading and preprocessing sample data.
*   tests/: Python and C unit tests.
