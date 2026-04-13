# MicroGAN

Train and deploy tiny GANs on microcontrollers.

MicroGAN trains a compact generative adversarial network on your laptop and exports it as a single C header file you can drop into any embedded project. Your microcontroller generates novel images on-device from a learned distribution — no cloud, no network, no runtime allocations.

## Quickstart

Run the entire pipeline end to end:

```bash
./run_pipeline.sh
```

This trains a generator, converts it through ONNX to quantized TFLite, and produces a C header with the weights baked in. Takes about 30 seconds.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Requires Python 3.9+ with PyTorch, TensorFlow, and ONNX. All dependencies are handled by `pip install -e .`.

## Usage

### Train a Generator

```bash
microgan train \
  --data ./my_dataset \
  --epochs 200 \
  --latent-dim 32 \
  --channels 1 \
  --output-dir build
```

Trains a DCGAN that learns to generate 32x32 images from your dataset. Omit `--data` to run with a built-in dummy dataset for testing the pipeline.

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | None | Path to image dataset (omit for dummy data) |
| `--epochs` | 10 | Training epochs |
| `--latent-dim` | 32 | Size of the latent noise vector |
| `--channels` | 1 | 1 for grayscale, 3 for RGB |
| `--output-dir` | `build` | Output directory for all artifacts |

### Export and Convert

Once training is done, three commands take you from a PyTorch checkpoint to a C header:

```bash
# PyTorch -> ONNX
microgan export-onnx \
  --checkpoint build/generator_final.pt \
  --output-dir build

# ONNX -> quantized TFLite
microgan onnx-to-tflite \
  --onnx-path build/generator.onnx \
  --output-dir build

# TFLite -> C header with weights and quantization parameters
microgan convert \
  --tflite build/generator_quantized.tflite \
  --output-dir build
```

The final output is `MicroGAN_weights.h`.

### Deploy to Your Microcontroller

Copy `MicroGAN_weights.h`, `MicroGAN_runtime.h`, and `MicroGAN_runtime.c` into your embedded project:

```c
#include "MicroGAN_runtime.h"
#include "MicroGAN_weights.h"

static uint8_t arena[1024 * 64];  // scratch space for activations
static uint8_t output[32 * 32];   // 32x32 grayscale image buffer

void setup() {
    MicroGAN_init(arena, sizeof(arena));
}

void loop() {
    uint8_t seed = get_random_seed();
    MicroGAN_generate(seed, 0, output);
    display_image(output);
}
```

Different seeds produce different images from the learned distribution. The runtime uses zero heap allocation — all memory comes from the static arena you provide.

## Hardware Targets

- ESP32 / ESP32-S3
- STM32F4 / STM32F7 (via CMSIS-NN)
- Raspberry Pi Pico (RP2040)
- Arduino Uno R4

## License

See [LICENSE](LICENSE) for details.
