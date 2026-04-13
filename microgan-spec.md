# MicroGAN: Generative AI for Microcontrollers
### Comprehensive Project Specification

**Version:** 0.1.0-draft  
**Status:** Pre-development  
**Stack:** Python · C/C++ · PyTorch · TFLite Micro · Arduino/PlatformIO

---

## Table of Contents

1. [Vision & Differentiation](#1-vision--differentiation)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Unique Edge: What Makes MicroGAN Different](#4-unique-edge-what-makes-MicroGAN-different)
5. [System Architecture](#5-system-architecture)
6. [Pipeline Stages — Detailed Specification](#6-pipeline-stages--detailed-specification)
   - 6.1 Training Stage
   - 6.2 Compression Stage
   - 6.3 Conversion Stage
   - 6.4 Deployment & Inference Stage
7. [MicroGAN Studio — The GUI Control Center](#7-MicroGANgan-studio--the-gui-control-center)
8. [Conditional Generation & Seed Control](#8-conditional-generation--seed-control)
9. [Model Zoo & Prebuilt Profiles](#9-model-zoo--prebuilt-profiles)
10. [Hardware Targets & Constraints](#10-hardware-targets--constraints)
11. [Project Structure](#11-project-structure)
12. [API & Interface Design](#12-api--interface-design)
13. [Data Requirements](#13-data-requirements)
14. [Testing & Validation Strategy](#14-testing--validation-strategy)
15. [Performance Benchmarks & Targets](#15-performance-benchmarks--targets)
16. [Security & Reproducibility](#16-security--reproducibility)
17. [Documentation Plan](#17-documentation-plan)
18. [Milestones & Roadmap](#18-milestones--roadmap)
19. [Open Questions & Risks](#19-open-questions--risks)

---

## 1. Vision & Differentiation

MicroGAN is an **end-to-end pipeline** for training compact generative adversarial networks on a laptop and deploying them fully on a microcontroller — no cloud, no Linux, no OS, no WiFi required at inference time.

The vision is deceptively simple but technically deep: **a $5 chip that creates images it has never stored.** Rather than replaying sprite sheets from flash memory, MicroGAN chips synthesize novel images on demand using a learned latent space — making them generative rather than merely retrieval-based.

This has immediate applications in IoT badges that show a unique face per boot, embedded games with procedural texture variety, hardware CAPTCHAs, and educational kits for learning TinyML.

---

## 2. Problem Statement

The current landscape of generative AI on embedded hardware is nearly nonexistent as a cohesive toolkit. Practitioners face:

- **No unified pipeline:** Training frameworks (PyTorch), quantization tools (ONNX, TFLite), and embedded runtimes (CMSIS-NN, TFLite Micro) exist in separate ecosystems with no connecting glue.
- **GAN-specific compression research stays in papers:** Most TinyML work focuses on classifiers and object detectors. GANs have fundamentally different compression behaviors (the generator must produce perceptually good output at very low bit-widths — a harder problem than classifier quantization) and there are no dedicated, maintained tools for this.
- **No hardware-aware design loop:** Existing GAN training ignores the target device's SRAM/flash limits. You train first, discover it doesn't fit second.
- **C export is painful and undocumented:** Converting PyTorch models to flat C arrays that compile on ARM Cortex-M0+ without dynamic memory allocation requires arcane knowledge not captured in any single project.
- **No latent space interface for end-users:** Even if someone gets inference running, there is no way to control *what* is generated — no seed injection, no class conditioning, no animation support.

MicroGAN solves all five of these problems in a single cohesive toolkit.

---

## 3. Goals & Non-Goals

### Goals

- Train a GAN whose generator fits within a configurable flash/SRAM budget (default target: ≤150 KB flash, ≤32 KB SRAM at inference).
- Produce output images at 32×32 pixels (grayscale or RGB), extensible to 64×64 for capable targets.
- Automate the full pipeline from raw image dataset → deployable C firmware in a single CLI command.
- Support STM32F4/F7, ESP32/ESP32-S3, and Raspberry Pi Pico (RP2040) as primary deployment targets.
- Provide a desktop GUI (MicroGAN Studio) for non-experts to train, compress, preview, and flash without CLI knowledge.
- Enable conditional generation (class-conditioned or label-conditioned) so the device can generate specific categories of images on demand.
- Provide a latent space "animation" mode that interpolates between seeds over time to produce smooth visual variation on-device.
- Ship a pre-trained model zoo (faces, icons, textures, emoji-style sprites) so users can deploy in minutes with no training required.
- Include hardware-in-the-loop (HIL) validation to confirm the deployed model's output matches the PC reference output within a tolerance threshold.

### Non-Goals (v1)

- Discriminator deployment on-device (training happens only on PC).
- Video generation or temporal sequence synthesis.
- Diffusion models or VAEs (future versions may add these).
- Support for MCUs with less than 128 KB flash or less than 20 KB SRAM.
- Real-time retraining or federated learning on-device.
- Color depth beyond 8-bit per channel (24-bit RGB maximum).

---

## 4. Unique Edge: What Makes MicroGAN Different

This section defines what no comparable project currently does, and therefore what gives MicroGAN its competitive advantage.

### 4.1 Hardware-Budget-Aware Training from the Start

Most TinyML workflows train first, compress second, and then discover the model doesn't fit the target device. MicroGAN introduces a **Hardware Profile** concept baked into the training configuration. Before training begins, the user selects a target device (or defines custom SRAM/flash constraints), and the trainer automatically:

- Caps the number of generator parameters to stay within budget.
- Estimates post-quantization size and reports it before training starts.
- Emits a warning if the configuration cannot possibly compress below budget, allowing reconfiguration before wasting compute.

This is a feedback loop that no other tool provides.

### 4.2 GAN-Aware Quantization (Not Just Classifier Quantization)

Standard quantization tools treat all models identically. GANs have a known failure mode under aggressive quantization: low-bit weights collapse the generator's output to a few nearly identical patterns. MicroGAN addresses this with:

- **Layer sensitivity analysis** that identifies which generator layers are most sensitive to bit-width reduction and keeps them at INT8 while quantizing less sensitive layers to INT4.
- **Quantization-aware GAN training (QA-GAN):** The discriminator remains in full precision during training, but the generator is trained with a quantization simulation (fake-quant nodes). This means the generator learns to produce good images *under the constraints it will face at deployment* — a critical difference from post-training quantization alone.
- **Perceptual quality metric tracking during compression:** MicroGAN tracks FID (Fréchet Inception Distance) proxies during the compression phase to detect when compression has degraded visual quality past an acceptable threshold.

### 4.3 Static Memory C Export

Microcontrollers like RP2040 have no heap allocator in their typical firmware configuration. Most ONNX/TFLite export tools assume dynamic allocation. MicroGAN exports a generator that uses:

- **Static weight arrays** (const arrays in flash via `PROGMEM` or linker section attributes).
- **Fixed scratch buffer** (a single stack-allocated arena for intermediate activations, sized at compile time).
- **Zero heap usage** — validated by a linker script check that errors if the heap segment is non-zero.

This is critical for reliability on bare-metal targets and is not provided by any existing tool.

### 4.4 Seed-to-Image CLI and Hardware API

Every deployed MicroGAN chip exposes a simple, documented interface:

- **On PC (validation):** `tinygen --seed 42 --class icons` produces a PNG to compare against hardware output.
- **On hardware:** A single function call `MicroGAN_generate(uint8_t seed, uint8_t class_id, uint8_t* output_buffer)` fills a 32×32 pixel buffer.
- **Over UART/USB:** The firmware accepts a seed byte over serial and immediately DMA-transfers the resulting image buffer to the host for capture or display.

This makes MicroGAN hardware trivially integrable into any larger embedded system.

### 4.5 Latent Space Animation Mode

A unique feature: the firmware can generate a smooth visual animation by interpolating between two random seeds in the generator's latent space. Because the latent space is continuous, points between two seeds produce visually coherent intermediate images. On a 32×32 display, this creates a hypnotic slow-morphing effect from one generated image to another, using nothing but integer arithmetic and the static weight arrays. No other embedded GAN project supports this.

### 4.6 One-Command Pipeline

```bash
tinygen train --data ./my_icons --target esp32 --output ./build
```

This single command runs training, compression, conversion, C code generation, and optionally flashing, with sensible defaults for every step. The pipeline is fully resumable (each stage checkpoints its outputs) and reproducible (a `tinygen.lock` file records all hyperparameters, random seeds, and tool versions used).

### 4.7 Visual Quality Preview Before Flash

MicroGAN Studio shows a side-by-side comparison of:
- Full-precision PC-generated samples.
- INT8 quantized samples (emulated on PC).
- INT4 quantized samples (emulated on PC, if selected).

The user approves quality before flashing, preventing "looks fine on PC, garbage on hardware" surprises.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Developer Machine (PC/Laptop)           │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │  Dataset │──▶│   Training   │──▶│   Compression     │   │
│  │  (PNG/   │   │   Engine     │   │   Engine          │   │
│  │  JPEG)   │   │  (PyTorch)   │   │  (QA-GAN + PTQ)   │   │
│  └──────────┘   └──────────────┘   └─────────┬─────────┘   │
│                                               │             │
│  ┌─────────────────────────────┐   ┌──────────▼─────────┐   │
│  │     MicroGAN Studio (GUI)    │   │   Conversion       │   │
│  │  - Dataset import           │   │   Engine           │   │
│  │  - Training config          │   │  (ONNX/TFLite →    │   │
│  │  - Quality preview          │   │   C array export)  │   │
│  │  - Flash tool               │   └──────────┬─────────┘   │
│  └─────────────────────────────┘              │             │
│                                               │             │
│  ┌─────────────────────────────────────────────▼──────────┐ │
│  │              CLI (tinygen)                              │ │
│  │  train | compress | convert | preview | flash | bench  │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬─────────────────────────┘
                                    │ USB/UART (flash + validate)
                 ┌──────────────────▼───────────────────┐
                 │         Microcontroller Firmware       │
                 │                                        │
                 │  ┌──────────────────────────────────┐  │
                 │  │     MicroGAN_runtime (C/C++)       │  │
                 │  │  - Static weight arrays (flash)   │  │
                 │  │  - Fixed arena allocator (SRAM)   │  │
                 │  │  - INT8/INT4 matmul kernels        │  │
                 │  │  - Seed → latent → image pipeline  │  │
                 │  │  - Animation interpolation engine  │  │
                 │  │  - UART command interface          │  │
                 │  └──────────────────────────────────┘  │
                 │                                        │
                 │  Targets: STM32F4/F7, ESP32/ESP32-S3,  │
                 │           RP2040, Arduino Uno R4       │
                 └────────────────────────────────────────┘
```

### Data Flow Summary

1. User provides a directory of 32×32 PNG images (or lets MicroGAN auto-resize).
2. Training engine trains a DCGAN or Mini-StyleGAN generator + discriminator.
3. Compression engine applies QAT (Quantization-Aware Training) and optional layer-wise pruning.
4. Conversion engine exports to TFLite FlatBuffer or ONNX, then converts to a C header (`MicroGAN_weights.h`) containing a flat `const uint8_t` array.
5. The runtime firmware C library + the generated header are compiled into a PlatformIO/Arduino project.
6. The firmware is flashed to the target device.
7. (Optional) HIL validation sends a seed to the device over UART and compares the device's output buffer to the PC reference output.

---

## 6. Pipeline Stages — Detailed Specification

### 6.1 Training Stage

#### Architecture Options

| Architecture | Parameters (approx.) | Use Case |
|---|---|---|
| MicroDCGAN | 80K–200K | Fast training, basic shapes & textures |
| MiniStyleGAN | 200K–600K | Better quality, style mixing, more diverse output |
| PatchGAN-Tiny | 60K–150K | Texture generation, no global structure needed |

**Default recommendation:** MicroDCGAN for ≤150 KB targets; MiniStyleGAN for ESP32-S3 (≤300 KB targets).

#### MicroDCGAN Generator Architecture

```
Latent z (dim=32, INT8 on device, float32 during training)
    │
    ▼
Linear(32 → 4*4*128)  → ReLU → Reshape(128, 4, 4)
    │
    ▼
ConvTranspose2d(128→64, k=4, s=2, p=1)  → BatchNorm → ReLU   [8×8]
    │
    ▼
ConvTranspose2d(64→32, k=4, s=2, p=1)   → BatchNorm → ReLU   [16×16]
    │
    ▼
ConvTranspose2d(32→C, k=4, s=2, p=1)    → Tanh               [32×32×C]

C = 1 (grayscale) or 3 (RGB)
```

Total parameters: ~110K (RGB), ~95K (grayscale). Post INT8 quantization: ~110 KB / ~95 KB.

#### MicroDCGAN Discriminator Architecture (Training Only — Not Deployed)

```
Input [32×32×C]
    │
Conv2d(C→32, k=4, s=2, p=1)   → LeakyReLU(0.2)   [16×16]
Conv2d(32→64, k=4, s=2, p=1)  → BN → LeakyReLU   [8×8]
Conv2d(64→128, k=4, s=2, p=1) → BN → LeakyReLU   [4×4]
Flatten → Linear(128*4*4 → 1) → Sigmoid
```

#### Conditional Generation Extension

When class labels are provided (e.g., 5 icon categories), the generator and discriminator are conditioned with label embeddings:

- Generator: `z_cond = concat(z, embed(label))` where `embed` is a learned embedding of dimension 8.
- Discriminator: label embedding is projected and added to feature maps after the first conv layer.
- On-device: The `class_id` argument to `MicroGAN_generate()` selects from a lookup table of pre-computed class embedding vectors stored in flash.

#### Training Configuration File (`tinygen.yaml`)

```yaml
training:
  architecture: micro_dcgan          # micro_dcgan | mini_stylegan | patchtiny
  latent_dim: 32                     # Latent vector dimensionality
  image_size: 32                     # 32 or 64
  channels: 3                        # 1 (grayscale) or 3 (RGB)
  conditional: false                 # Enable class conditioning
  num_classes: 1                     # Ignored if conditional: false
  epochs: 200
  batch_size: 64
  lr_g: 0.0002
  lr_d: 0.0002
  beta1: 0.5
  beta2: 0.999
  seed: 42                           # Global random seed (for reproducibility)
  augmentation:
    horizontal_flip: true
    random_crop: true
    color_jitter: false              # Keep false for grayscale

compression:
  target_size_kb: 150                # Maximum flash budget for the generator
  quantization_mode: qat             # qat (recommended) | ptq
  qat_epochs: 50                     # Additional epochs with fake-quant nodes
  weight_bits: 8                     # 8 (INT8) or 4 (INT4, experimental)
  activation_bits: 8
  pruning_enabled: false             # Optional unstructured pruning before QAT
  pruning_sparsity: 0.2              # 20% weight sparsity if enabled

hardware:
  target: esp32                      # esp32 | esp32s3 | stm32f4 | stm32f7 | rp2040
  flash_kb: 4096                     # Total flash available
  sram_kb: 512                       # Total SRAM available
  display: spi_lcd_st7735            # Display driver hint (optional, for firmware codegen)

output:
  dir: ./build
  generate_firmware: true
  firmware_framework: platformio     # platformio | arduino | espidf | bare_metal
```

#### Training Output Artifacts

```
build/
  checkpoints/
    generator_epoch_100.pt
    generator_epoch_200.pt          ← Best checkpoint by FID proxy
  logs/
    training_loss.csv
    sample_grid_epoch_*.png
  final/
    generator.pt                    ← Full-precision final generator
    discriminator.pt
    training_summary.json           ← Hyperparams, loss curves, FID, total time
```

---

### 6.2 Compression Stage

#### Stage Goals

Reduce generator from full-precision float32 (~440 KB for 110K params) to ≤150 KB while maintaining perceptually acceptable output quality. Acceptable is defined as **FID proxy degradation ≤ 20%** vs. the full-precision model.

#### Step 1 — Optional Pruning

If `pruning_enabled: true`, apply unstructured magnitude pruning before QAT:

```python
# Implemented using torch.nn.utils.prune
prune.l1_unstructured(module, name='weight', amount=sparsity)
```

Sparse weights are zeroed but remain in the weight tensor. During C export, zero weights are stored as INT8 zeros and the runtime can skip their multiplications with a mask (providing a speed benefit on CMSIS-NN with sparse compute paths).

#### Step 2 — Quantization-Aware Training (QAT)

After standard training, the generator is fine-tuned with fake-quantization nodes inserted:

```python
generator.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(generator, inplace=True)
# Fine-tune for qat_epochs with discriminator in eval mode (frozen)
torch.quantization.convert(generator, inplace=True)
```

The discriminator is kept frozen at full precision during QAT. This is the QA-GAN approach: the generator is pushed to learn outputs that are robust to quantization noise, guided by the full-precision discriminator's unchanged quality signal.

#### Step 3 — Size Validation

After QAT, the tool reports:

```
Generator size (float32):    440 KB
Generator size (INT8):       110 KB  ✓ Under 150 KB budget
SRAM for activations:         18 KB  ✓ Under 32 KB budget
Estimated FID degradation:    +8.3%  ✓ Under 20% threshold
```

If any budget is exceeded, the tool suggests architectural changes (reducing latent dim, reducing channel multiplier) before proceeding.

#### Step 4 — TFLite Export

```python
# PyTorch INT8 → ONNX → TFLite FlatBuffer
torch.onnx.export(generator_int8, dummy_input, "generator.onnx", ...)
# onnx → tflite via onnx-tf + TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
tflite_model = converter.convert()
```

#### Compression Output Artifacts

```
build/
  compressed/
    generator_int8.pt               ← PyTorch INT8 model
    generator.onnx                  ← ONNX intermediate
    generator.tflite                ← TFLite FlatBuffer
    compression_report.json         ← Size, FID proxy, layer sensitivities
    sample_comparison/
      full_precision_samples.png
      int8_samples.png
      int4_samples.png (if applicable)
```

---

### 6.3 Conversion Stage

The conversion stage produces a single C header file and a runtime C source file that together constitute the entire on-device inference engine.

#### C Header Generation: `MicroGAN_weights.h`

```c
// Auto-generated by MicroGAN v0.1.0
// Model: micro_dcgan | Target: esp32 | Quantization: INT8
// Generator params: 110,592 | Flash: 108 KB | SRAM arena: 18 KB

#pragma once
#include <stdint.h>

#define MicroGAN_LATENT_DIM      32
#define MicroGAN_OUTPUT_WIDTH    32
#define MicroGAN_OUTPUT_HEIGHT   32
#define MicroGAN_OUTPUT_CHANNELS  3
#define MicroGAN_NUM_CLASSES      1
#define MicroGAN_ARENA_SIZE   18432   // bytes, must be allocated by caller

// Quantization scale/zero-point per layer
#define MicroGAN_SCALE_FC1   0.003921f
#define MicroGAN_ZP_FC1      128
// ... (all layers)

// Weight data (stored in flash via PROGMEM or linker section)
extern const int8_t MicroGAN_WEIGHTS[];
#define MicroGAN_WEIGHTS_LEN  110592

// Optional: Class conditioning embedding lookup table
extern const int8_t MicroGAN_CLASS_EMBEDDINGS[];
```

#### Runtime C Library: `MicroGAN_runtime.c / .h`

Public API (stable, versioned):

```c
// Initialize the inference engine with a user-provided static arena buffer.
// arena must be MicroGAN_ARENA_SIZE bytes, aligned to 8 bytes.
MicroGAN_status_t MicroGAN_init(uint8_t* arena, size_t arena_size);

// Generate a 32x32 pixel image into output_rgb888.
// seed: 0-255 deterministic seed (expands to latent vector via PRNG)
// class_id: 0 to MicroGAN_NUM_CLASSES-1 (use 0 if not conditional)
// output: caller-allocated buffer of size W*H*C bytes (3072 bytes for 32x32 RGB)
MicroGAN_status_t MicroGAN_generate(uint8_t seed,
                                   uint8_t class_id,
                                   uint8_t* output_rgb888);

// Generate interpolated frame between two seeds.
// alpha: 0 = seed_a, 255 = seed_b, 128 = midpoint
MicroGAN_status_t MicroGAN_interpolate(uint8_t seed_a,
                                      uint8_t seed_b,
                                      uint8_t alpha,
                                      uint8_t* output_rgb888);

// Retrieve last error string (useful for debugging over UART)
const char* MicroGAN_last_error(void);

// Report memory usage to a callback (for diagnostics)
void MicroGAN_memory_report(void (*print_fn)(const char*));
```

#### Compute Backends

The runtime selects a compute backend at compile time via preprocessor flags:

| Flag | Backend | Targets |
|---|---|---|
| `MicroGAN_BACKEND_CMSIS` | ARM CMSIS-NN (optimized for Cortex-M) | STM32, RP2040 |
| `MicroGAN_BACKEND_XTENSA` | Xtensa DSP intrinsics | ESP32 |
| `MicroGAN_BACKEND_GENERIC` | Pure C99 reference (portable) | All |

The generic backend is always included as a fallback. A CI test confirms that all backends produce identical output for the same seed.

#### Firmware Template Generation

The conversion stage generates a complete PlatformIO project:

```
build/firmware/
  platformio.ini                    ← Board config, dependencies
  src/
    main.cpp                        ← Example: generate on boot, send over UART
    MicroGAN_runtime.c
    MicroGAN_runtime.h
  include/
    MicroGAN_weights.h               ← Auto-generated, do not edit
  test/
    test_inference.cpp              ← Unity test: seed 0 produces expected checksum
```

---

### 6.4 Deployment & Inference Stage

#### Flashing

```bash
# PlatformIO (recommended)
tinygen flash --target esp32 --port /dev/ttyUSB0

# Arduino CLI
tinygen flash --framework arduino --target rp2040 --port /dev/ttyACM0

# OpenOCD (STM32)
tinygen flash --framework openocd --target stm32f4 --port /dev/stlink
```

#### Hardware-in-the-Loop (HIL) Validation

After flashing, MicroGAN validates the deployment:

```bash
tinygen validate --port /dev/ttyUSB0 --seeds 0,1,2,42,100
```

The tool sends each seed to the device over UART, receives the 3072-byte output buffer, and compares it to the PC-computed reference using MSE. Pass threshold: MSE < 5.0 per pixel (accounting for minor fixed-point rounding differences).

```
Validating deployment on /dev/ttyUSB0...
  Seed 0:   MSE = 1.24   ✓ PASS
  Seed 1:   MSE = 0.87   ✓ PASS
  Seed 42:  MSE = 2.11   ✓ PASS
  Seed 100: MSE = 3.45   ✓ PASS

  Inference time (seed→image): 47 ms
  Peak SRAM usage:             18.2 KB
  Flash usage (weights+code):  122 KB

All validation checks passed.
```

#### UART Command Protocol (for Integration)

The default firmware implements a simple binary command protocol over UART (115200 baud):

| Byte | Meaning |
|---|---|
| `0x47` | Magic byte (G for Generate) |
| `seed` | 1 byte seed value |
| `class_id` | 1 byte class ID |
| `alpha` | 1 byte interpolation alpha (0 = no interpolation, use seed directly) |
| `seed_b` | 1 byte second seed (ignored if alpha = 0) |

Response: `3072` bytes of raw RGB888 pixel data (32×32×3), MSB first, row-major.

---

## 7. MicroGAN Studio — The GUI Control Center

MicroGAN Studio is a cross-platform desktop application (Python + Dear PyGui or Tkinter) that provides a visual interface for the entire pipeline. It is designed for students and hobbyists who may not be comfortable with a CLI.

### Screens

#### Screen 1: Dataset Wizard
- Drag-and-drop a folder of images.
- Auto-preview of resized 32×32 thumbnails.
- Dataset statistics: count, aspect ratio distribution, class label detection.
- "Download Sample Dataset" button (fetches MNIST, CIFAR-10 icons subset, or custom sprite pack).

#### Screen 2: Train & Compress
- Hardware target selector (dropdown: ESP32, STM32F4, RP2040, etc.) — automatically sets the size budget.
- Architecture recommendation based on budget.
- Live training dashboard: G loss, D loss, FID proxy, sample grid (refreshes every 10 epochs).
- "Stop and Compress" button triggers the compression pipeline.

#### Screen 3: Quality Preview
- Side-by-side comparison: Full Precision | INT8 | INT4.
- Seed slider: drag to preview different generated images.
- Class selector (if conditional model).
- "Approve and Export" button.

#### Screen 4: Flash & Validate
- Port detection (auto-scans serial ports, identifies likely targets).
- Flash progress bar.
- Live UART output window.
- "Run HIL Validation" button with pass/fail summary.

---

## 8. Conditional Generation & Seed Control

### Seed Expansion

A raw `uint8_t seed` (0–255) is insufficient as a latent vector (dimension 32). MicroGAN uses a deterministic PRNG expansion:

```c
// Expand 1-byte seed to MicroGAN_LATENT_DIM INT8 values
// Uses xoshiro128** seeded with the user seed, producing reproducible latents
void MicroGAN_expand_seed(uint8_t seed, int8_t* latent_out, int latent_dim);
```

The PRNG is fixed (not hardware RNG) to ensure identical outputs for the same seed across all devices. If true randomness is desired, the user can seed from `esp_random()` or similar before calling `MicroGAN_generate`.

### Class Conditioning

```c
// Conditional generation: generate class_id=2 (e.g., "star icon")
MicroGAN_generate(seed=42, class_id=2, output_buffer);
```

Class embeddings are stored as a small lookup table in flash:

```c
// 8-dimensional INT8 embedding per class, stored in flash
const int8_t MicroGAN_CLASS_EMBEDDINGS[NUM_CLASSES * 8] = { ... };
```

The selected embedding is concatenated with the expanded latent before the first layer of the generator.

### Latent Interpolation (Animation)

```c
// Smoothly morph between seed 10 and seed 200
for (uint8_t alpha = 0; alpha < 255; alpha++) {
    MicroGAN_interpolate(seed_a=10, seed_b=200, alpha=alpha, output_buffer);
    display_frame(output_buffer);
    delay_ms(33); // ~30 fps
}
```

The latent vectors for both seeds are pre-expanded and linearly interpolated in INT8 arithmetic before passing through the generator. This produces the characteristic GAN "latent walk" animation entirely on-device.

---

## 9. Model Zoo & Prebuilt Profiles

MicroGAN ships prebuilt, ready-to-flash model binaries for common use cases. Users can deploy these without training anything.

| Model Name | Dataset | Output | Size | Classes | Use Case |
|---|---|---|---|---|---|
| `faces-grayscale` | CelebA-HQ (resized) | 32×32 grayscale | 95 KB | 1 | IoT badge avatar |
| `icons-rgb` | OpenMoji subset | 32×32 RGB | 108 KB | 8 | Embedded UI icons |
| `textures-rgb` | Describable Textures | 32×32 RGB | 112 KB | 5 | Procedural game textures |
| `symbols-grayscale` | Custom symbol set | 32×32 grayscale | 88 KB | 16 | Display glyphs |
| `noise-patterns` | Generated fractals | 32×32 grayscale | 76 KB | 1 | CAPTCHA / unique ID art |

Each prebuilt model ships with:
- The `.tflite` and `.h` files.
- A complete PlatformIO firmware project.
- A `preview_samples.png` grid showing 64 generated samples.
- A `model_card.md` describing training data, quality metrics, and intended use.

---

## 10. Hardware Targets & Constraints

| Target | Flash | SRAM | CPU | Max Model Size | Notes |
|---|---|---|---|---|---|
| ESP32 (original) | 4 MB | 520 KB | Xtensa LX6, 240 MHz | 300 KB | Xtensa DSP backend |
| ESP32-S3 | 8 MB | 512 KB | Xtensa LX7, 240 MHz | 400 KB | Vector extension support |
| STM32F407 | 1 MB | 192 KB | Cortex-M4F, 168 MHz | 150 KB | CMSIS-NN backend |
| STM32F746 | 1 MB | 320 KB | Cortex-M7, 216 MHz | 200 KB | CMSIS-NN + FPU |
| RP2040 | 2 MB (external) | 264 KB | Cortex-M0+, 133 MHz | 150 KB | No FPU — pure INT8 only |
| Arduino Uno R4 (RA4M1) | 256 KB | 32 KB | Cortex-M4, 48 MHz | 80 KB | Tight budget — grayscale only |

### Memory Layout (ESP32 example)

```
Flash Map (4 MB total):
  [0x000000 - 0x010000]  Bootloader         (64 KB)
  [0x010000 - 0x0F0000]  Application code   (896 KB)
  [0x0F0000 - 0x1B0000]  MicroGAN weights    (112 KB)  ← MicroGAN_WEIGHTS in flash
  [0x1B0000 - 0x400000]  Remaining / OTA    (2.3 MB)

SRAM Map (520 KB total):
  [Stack]                                   (8 KB)
  [Heap — intentionally zero for MicroGAN]   (0 KB)
  [MicroGAN arena (static)]                  (18 KB)   ← User provides this buffer
  [User application]                        (remaining)
```

---

## 11. Project Structure

```
MicroGAN/
├── README.md
├── spec.md                          ← This file
├── pyproject.toml                   ← Python package (tinygen CLI + training)
├── tinygen/                         ← Python package
│   ├── __init__.py
│   ├── cli.py                       ← Click-based CLI entry point
│   ├── train/
│   │   ├── __init__.py
│   │   ├── dcgan.py                 ← MicroDCGAN model definition
│   │   ├── stylegan_tiny.py         ← MiniStyleGAN model definition
│   │   ├── trainer.py               ← Training loop, logging, checkpointing
│   │   ├── fid_proxy.py             ← Lightweight FID estimation (no Inception)
│   │   └── augment.py               ← Dataset augmentation pipeline
│   ├── compress/
│   │   ├── __init__.py
│   │   ├── qat.py                   ← Quantization-aware training
│   │   ├── ptq.py                   ← Post-training quantization fallback
│   │   ├── pruning.py               ← Magnitude-based unstructured pruning
│   │   ├── sensitivity.py           ← Layer sensitivity analysis
│   │   └── size_estimator.py        ← Pre-training budget checker
│   ├── convert/
│   │   ├── __init__.py
│   │   ├── to_onnx.py
│   │   ├── to_tflite.py
│   │   ├── to_c_array.py            ← TFLite flatbuffer → C header
│   │   └── firmware_gen.py          ← PlatformIO/Arduino project generator
│   ├── validate/
│   │   ├── __init__.py
│   │   ├── hil.py                   ← Hardware-in-the-loop validator
│   │   └── pc_reference.py          ← PC-side reference inference
│   ├── hardware/
│   │   ├── profiles/
│   │   │   ├── esp32.yaml
│   │   │   ├── esp32s3.yaml
│   │   │   ├── stm32f4.yaml
│   │   │   ├── stm32f7.yaml
│   │   │   └── rp2040.yaml
│   │   └── flash.py                 ← Platform-specific flash tool wrapper
│   └── studio/
│       ├── __init__.py
│       ├── app.py                   ← Dear PyGui app entry point
│       └── screens/
│           ├── dataset_wizard.py
│           ├── train_compress.py
│           ├── quality_preview.py
│           └── flash_validate.py
├── runtime/                         ← C/C++ embedded runtime
│   ├── CMakeLists.txt
│   ├── MicroGAN_runtime.h
│   ├── MicroGAN_runtime.c            ← Core inference engine
│   ├── MicroGAN_prng.c               ← xoshiro128** seed expansion
│   ├── backends/
│   │   ├── backend_cmsis.c          ← ARM CMSIS-NN
│   │   ├── backend_xtensa.c         ← ESP32 Xtensa DSP
│   │   └── backend_generic.c        ← Pure C99 reference
│   └── test/
│       ├── test_inference.c
│       └── test_interpolate.c
├── firmware_templates/              ← PlatformIO/Arduino project templates
│   ├── esp32_basic/
│   ├── esp32_uart_server/
│   ├── stm32_tft_display/
│   └── rp2040_basic/
├── model_zoo/                       ← Prebuilt model headers
│   ├── faces_grayscale/
│   ├── icons_rgb/
│   └── textures_rgb/
├── datasets/                        ← Sample dataset download scripts
│   ├── download_mnist_sprites.py
│   └── download_openmoji_subset.py
├── tests/                           ← Python test suite
│   ├── test_training.py
│   ├── test_compression.py
│   ├── test_conversion.py
│   └── test_cli.py
├── docs/
│   ├── getting_started.md
│   ├── hardware_guide.md
│   ├── architecture_guide.md
│   ├── api_reference.md
│   └── model_zoo.md
└── examples/
    ├── 01_train_icons.ipynb
    ├── 02_compress_and_flash_esp32.ipynb
    └── 03_latent_animation.ipynb
```

---

## 12. API & Interface Design

### CLI Commands

```bash
# Full pipeline in one command
tinygen train --config tinygen.yaml

# Individual stages
tinygen train   --data ./dataset --target esp32 --epochs 200 --output ./build
tinygen compress --checkpoint ./build/checkpoints/best.pt --target esp32
tinygen convert  --tflite ./build/compressed/generator.tflite --target esp32
tinygen flash    --build-dir ./build/firmware --port /dev/ttyUSB0
tinygen validate --port /dev/ttyUSB0 --seeds 0,1,42

# Preview without hardware
tinygen preview  --checkpoint ./build/final/generator.pt --seeds 0,1,2,3,4,5,6,7

# Benchmark (runs on PC using the C reference backend via ctypes)
tinygen bench --weights ./build/firmware/include/MicroGAN_weights.h

# Use a model zoo entry
tinygen flash --zoo icons-rgb --target esp32 --port /dev/ttyUSB0

# GUI
tinygen studio
```

### Python Library API

```python
from tinygen import MicroGAN

# Training
gan = MicroGAN(config="tinygen.yaml")
gan.train(data_dir="./my_icons")
gan.compress()
gan.convert()
gan.flash(port="/dev/ttyUSB0")

# Direct inference (PC, for testing)
from tinygen.validate import PCReference
ref = PCReference("./build/compressed/generator_int8.pt")
image = ref.generate(seed=42, class_id=0)   # Returns numpy array (32, 32, 3)
image_pil = ref.to_pil(image)
image_pil.save("preview.png")
```

---

## 13. Data Requirements

### Minimum Dataset Size

- Recommended: ≥ 500 images per class for conditional models, ≥ 1000 images total for unconditional.
- Acceptable: ≥ 200 images (training will still converge but with lower diversity).

### Accepted Formats

- JPEG, PNG, BMP, WebP (any color depth — auto-converted to target channels).
- Automatic resizing to 32×32 (center-crop or pad, configurable).
- Folder-based class labeling: `dataset/class_a/*.png`, `dataset/class_b/*.png`.

### Bundled Sample Datasets

MicroGAN ships download scripts for:

- **MNIST sprites** — 70,000 grayscale digit images, resized to 32×32. Good for verifying the pipeline works.
- **OpenMoji subset** — 500 color emoji-style icons. Good for the icons-rgb model.
- **Describable Textures Dataset (DTD) subset** — 500 texture crops at 32×32. Good for the textures-rgb model.

---

## 14. Testing & Validation Strategy

### Unit Tests (pytest)

- `test_dcgan_output_shape`: Generator output is (batch, C, 32, 32) for various batch sizes.
- `test_qat_size`: INT8-converted model fits under budget for each hardware profile.
- `test_c_array_roundtrip`: C array export → load → inference matches TFLite inference within tolerance.
- `test_seed_determinism`: Same seed always produces the same output (PC reference).
- `test_interpolation_monotone`: Interpolated images smoothly vary between seed_a and seed_b outputs.

### Integration Tests

- `test_full_pipeline_esp32_sim`: Runs the complete pipeline (train 5 epochs, compress, convert, validate) against the ESP32 QEMU emulator.
- `test_cli_all_targets`: Runs `tinygen convert` for all 5 hardware profiles and confirms output is valid C.

### Hardware-in-the-Loop Tests (Manual / CI with hardware)

- Seed determinism test: same seed → same pixel output, verified byte-for-byte.
- MSE validation: all seeds within the HIL validation suite pass MSE < 5.0.
- Timing test: inference time measured over 100 runs, reported as mean ± std dev.
- Arena overflow test: firmware with instrumented stack canaries confirms no SRAM overflow.

### CI Pipeline (GitHub Actions)

```yaml
jobs:
  test-python:     # runs pytest on Ubuntu, macOS, Windows
  test-c-runtime:  # compiles runtime with gcc and clang, runs unit tests
  test-pipeline:   # runs full pipeline with 5-epoch training on MNIST sprites
  lint:            # flake8, mypy (Python), cppcheck (C)
  size-check:      # confirms model zoo entries are under their documented sizes
```

---

## 15. Performance Benchmarks & Targets

### PC Training (reference: NVIDIA RTX 3060 or equivalent)

| Architecture | Dataset (1K images) | Epochs | Training Time |
|---|---|---|---|
| MicroDCGAN | 32×32 grayscale | 200 | ~4 min |
| MicroDCGAN | 32×32 RGB | 200 | ~6 min |
| MiniStyleGAN | 32×32 RGB | 200 | ~18 min |

On CPU only: ~3–6× slower.

### On-Device Inference Targets

| Target | Architecture | Resolution | Target Inference Time |
|---|---|---|---|
| ESP32 (240 MHz) | MicroDCGAN INT8 | 32×32 RGB | ≤ 80 ms |
| ESP32-S3 (240 MHz) | MicroDCGAN INT8 | 32×32 RGB | ≤ 50 ms |
| STM32F407 (168 MHz) | MicroDCGAN INT8 | 32×32 RGB | ≤ 120 ms |
| RP2040 (133 MHz) | MicroDCGAN INT8 | 32×32 grayscale | ≤ 200 ms |

These targets allow latent animation at 5–30 fps depending on hardware.

### Flash & SRAM Budget Targets

| Model | Flash (weights) | SRAM (arena) |
|---|---|---|
| MicroDCGAN INT8 RGB | ≤ 115 KB | ≤ 20 KB |
| MicroDCGAN INT8 grayscale | ≤ 95 KB | ≤ 16 KB |
| MiniStyleGAN INT8 RGB | ≤ 280 KB | ≤ 40 KB |

---

## 16. Security & Reproducibility

### Reproducibility

Every MicroGAN training run produces a `tinygen.lock` file:

```json
{
  "MicroGAN_version": "0.1.0",
  "python_version": "3.11.4",
  "torch_version": "2.1.0",
  "config_hash": "sha256:a3f4...",
  "dataset_hash": "sha256:b2e1...",
  "global_seed": 42,
  "training_completed": "2025-06-01T14:23:00Z",
  "final_checkpoint": "sha256:c7a9..."
}
```

This file is sufficient to reproduce the exact same trained model (given identical hardware, which affects floating-point ops on some platforms).

### Model Provenance

All model zoo entries include a `model_card.md` with training data description, intended use, known limitations, and performance metrics. This follows the Model Cards for Model Reporting standard.

### No Network Access Required at Inference

The deployed firmware has zero network dependencies. No telemetry, no OTA updates, no DNS calls. The device is fully air-gapped post-flash.

---

## 17. Documentation Plan

| Document | Audience | Format |
|---|---|---|
| `README.md` | All users | Markdown, quick-start in under 5 minutes |
| `docs/getting_started.md` | Beginners | Step-by-step walkthrough: icons dataset → ESP32 in 20 minutes |
| `docs/hardware_guide.md` | Hardware engineers | Wiring, power, display connections, UART testing |
| `docs/architecture_guide.md` | ML practitioners | GAN architecture choices, QAT details, compression tradeoffs |
| `docs/api_reference.md` | Python developers | Full CLI and library API reference |
| `docs/c_runtime_api.md` | Firmware engineers | C runtime API, memory model, backend porting guide |
| `docs/model_zoo.md` | All users | Prebuilt model cards, preview samples, flash instructions |
| Jupyter notebooks (3) | Students / educators | Hands-on examples with explanations |
| YouTube walkthrough (planned) | Beginners | End-to-end demo: laptop → ESP32 displaying generated icons |

---

## 18. Milestones & Roadmap

### Phase 1 — Core Pipeline (Weeks 1–4)

- [ ] MicroDCGAN training (PyTorch, unconditional, grayscale)
- [ ] Basic PTQ compression to INT8
- [ ] TFLite export + C array generation
- [ ] Generic C runtime (no CMSIS, pure C99)
- [ ] Working firmware on ESP32 (PlatformIO)
- [ ] CLI: `train`, `compress`, `convert`, `flash`
- [ ] Unit tests passing, CI configured

### Phase 2 — Quality & Compression (Weeks 5–8)

- [ ] QA-GAN (quantization-aware training)
- [ ] Layer sensitivity analysis
- [ ] Hardware budget-aware training config
- [ ] RGB support (3-channel)
- [ ] CMSIS-NN backend for STM32 / RP2040
- [ ] HIL validation tool
- [ ] FID proxy metric

### Phase 3 — Features (Weeks 9–12)

- [ ] Conditional generation (class conditioning)
- [ ] Seed interpolation / latent animation
- [ ] MiniStyleGAN support
- [ ] RP2040 and STM32 targets validated
- [ ] Model zoo (3 prebuilt models)
- [ ] `tinygen preview` command

### Phase 4 — Polish & Launch (Weeks 13–16)

- [ ] MicroGAN Studio GUI (MVP: dataset wizard + train + preview + flash)
- [ ] All documentation complete
- [ ] Jupyter notebooks
- [ ] Full model zoo (5 models)
- [ ] GitHub release with binaries
- [ ] Demo video

### Future (Post-v1)

- INT4 support (experimental, for ultra-constrained targets)
- 64×64 output on ESP32-S3
- VAE support (deterministic latent encoding for image compression use cases)
- WASM export (run MicroGAN generator in a browser)
- Online model fine-tuning via the Studio (adapt pretrained model to custom dataset with 50 images in 10 minutes)

---

## 19. Open Questions & Risks

### Technical Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| INT8 quantization causes GAN mode collapse | Medium | QA-GAN training, layer sensitivity gating, INT8 floor for sensitive layers |
| TFLite Micro doesn't support all required ops for transposed convolutions | Medium | Custom op fallback in generic C backend; ONNX Runtime Micro as alternative |
| RP2040 (no FPU, 133 MHz) is too slow for acceptable generation | Low-Medium | Profile early; fall back to grayscale only; accept 200 ms as sufficient |
| PyTorch → ONNX → TFLite conversion pipeline is brittle across versions | High | Pin all tool versions in `pyproject.toml`; integration test the full conversion on every CI run |

### Open Design Questions

1. **PRNG choice for seed expansion:** xoshiro128** is proposed. Should we use a simpler LCG for tiny targets (RP2040 RAM pressure) or is the added quality of xoshiro128** worth the ~64 bytes of state?

2. **FID proxy metric:** True FID requires Inception features (too heavy for this context). Options: (a) a tiny classifier-based proxy trained on the dataset, (b) SSIM-based diversity metric, (c) just track G/D loss ratio and visual inspection. Decision needed before Phase 2.

3. **GUI framework:** Dear PyGui is fast and lightweight but less familiar. Tkinter is universally available but dated. A lightweight web-based GUI (FastAPI + React embedded in an Electron shell) would be more maintainable long-term but is more complex to build. Decision needed before Phase 4.

4. **Model zoo licensing:** Generated images from GANs trained on CelebA-HQ are subject to CelebA's non-commercial license. Should the faces model be excluded from the default zoo, or trained on a more permissive dataset (e.g., FFHQ subset)?

5. **Xtensa DSP backend:** ESP32 Xtensa intrinsics can provide significant speedup but require Espressif's proprietary toolchain. Is the maintenance burden worth it, or should ESP32 use the generic C backend (still INT8, just less optimized)?

---

*End of MicroGAN Specification v0.1.0-draft*
