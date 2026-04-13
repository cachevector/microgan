# MicroGAN

MicroGAN is an end to end pipeline for training compact generative adversarial networks on a laptop and deploying them fully on a microcontroller. It allows you to synthesize novel images on demand using a learned latent space, making your devices generative rather than merely retrieval based. 

## Features

### Hardware Budget Aware Training
The trainer automatically caps the number of generator parameters based on your target device's SRAM and flash budget. It estimates the post quantization size before training starts to ensure the model will actually fit on the chip.

### GAN Aware Quantization
Most TinyML tools focus on classifiers. MicroGAN uses quantization aware training (QAT) and layer sensitivity analysis specifically for generators. This prevents the "mode collapse" that often happens when GANs are aggressively quantized to INT8.

### Static Memory C Export
The exported C runtime uses a fixed scratch buffer (arena) and static weight arrays in flash. There is zero heap usage, which is critical for reliability on bare metal targets like the RP2040 or STM32.

### Latent Space Animation
The runtime includes an engine to interpolate between different random seeds in the latent space. This produces smooth visual animations of images morphing into one another, calculated entirely on device with integer arithmetic.

## Usage Guide

### 1. Training
Use the CLI to train a model on your own dataset or a sample dataset. 

```bash
microgan train --epochs 200 --latent-dim 32 --channels 1 --output-dir build
```

This creates a PyTorch checkpoint in the build directory. If you do not provide a dataset path, it will generate a dummy dataset for testing the pipeline.

### 2. Compression and Conversion
To deploy on a microcontroller, the model must be converted to ONNX and then to a quantized TFLite format.

```bash
# Export to ONNX
microgan export-onnx --checkpoint build/generator_final.pt --output-dir build

# Convert to TFLite
microgan onnx-to-tflite --onnx-path build/generator.onnx --output-dir build
```

### 3. C Header Generation
Finally, convert the TFLite model into a C header file containing the weights and quantization parameters.

```bash
microgan convert --tflite build/generator_quantized.tflite --output-dir build
```

This generates `MicroGAN_weights.h`. You can include this file in your embedded project.

### 4. Deployment
The C runtime is located in the runtime directory. To use it, initialize the arena and call the generation function.

```c
#include "MicroGAN_runtime.h"
#include "MicroGAN_weights.h"

uint8_t arena[1024 * 64]; // Static scratch space
uint8_t output[32 * 32];  // Buffer for the generated image

void setup() {
    MicroGAN_init(arena, sizeof(arena));
}

void loop() {
    uint8_t seed = get_random_seed();
    MicroGAN_generate(seed, 0, output);
    display_image(output);
}
```

## Hardware Targets

MicroGAN is designed for:
* ESP32 and ESP32-S3
* STM32F4 and STM32F7 (via CMSIS-NN)
* Raspberry Pi Pico (RP2040)
* Arduino Uno R4
