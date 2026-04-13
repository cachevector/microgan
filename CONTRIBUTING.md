# Contributing to MicroGAN

Thank you for your interest in contributing to MicroGAN. This project aims to make generative AI accessible on low power microcontrollers, and community help is vital for expanding hardware support and optimizing the pipeline.

## How to contribute

There are several ways you can help improve the project:

### Reporting bugs
If you find a bug, please check the existing issues to see if it has already been reported. If not, open a new issue with a clear description of the problem and steps to reproduce it.

### Suggesting features
New ideas for hardware targets, model architectures, or compression techniques are always welcome. Open an issue to discuss your proposal before starting implementation.

### Code contributions
Submit a pull request for bug fixes or new features. For significant changes, please discuss them in an issue first to ensure they align with the project goals.

### Documentation
Improving the guides, adding examples, or fixing typos is a great way to contribute.

## Getting started

To set up your development environment:

1. Fork the repository and clone it to your machine.
2. Create a virtual environment using Python 3.9.
3. Install the dependencies and the package in editable mode:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. Run the existing tests to ensure everything is set up correctly.

## Technical stack

The project uses the following technologies:

* Python 3.9+ for the training and conversion pipeline.
* PyTorch for model definition and training.
* ONNX and TFLite for model optimization and quantization.
* onnx2tf for stable ONNX to TFLite conversion on macOS and Linux.
* C99 for the embedded inference runtime.

## Project structure

Understanding the directory layout helps in navigating the codebase:

* tinygen/: The main Python CLI and library.
  * train/: GAN architectures and training logic.
  * compress/: Model quantization and pruning tools.
  * convert/: Export utilities for ONNX, TFLite, and C headers.
  * validate/: Testing tools for the PC side C runtime.
  * hardware/: Hardware profiles and flashing tools.
* runtime/: The C/C++ source for the on-device inference engine.
  * backends/: Optimized kernels for specific architectures like CMSIS-NN or Xtensa.
* firmware_templates/: Scaffolding for different boards and frameworks.
* model_zoo/: Pre-trained weights for various image domains.
* datasets/: Scripts for downloading and preprocessing sample data.
* tests/: Python and C unit tests.

## Development workflow

When working on a change:

1. Create a new branch for your work.
2. Keep your changes focused. If you have multiple unrelated fixes, use separate pull requests.
3. Write or update tests for any new logic.
4. Ensure the code follows the existing style and conventions.
5. Provide a clear and concise description in your pull request.

Thank you for helping make MicroGAN better.
