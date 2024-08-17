# Overview of Project Components

This project is divided into three distinct parts, each exploring different aspects of machine learning model management and optimization. Below are brief descriptions and links to detailed instructions for each part.

## Quick Setup

Before diving into the parts, ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```

## Part 1: Model Quantization and Benchmarking

Objective: Quantize a complex vision model and benchmark its inference performance on the Tiny ImageNet dataset.

Details: Part 1 README
Key Tasks:

- Download and prepare a subset of Tiny ImageNet.
- Apply dynamic quantization to the model.
- Measure inference time and accuracy differences.

## Part 2: Automated Hyperparameter Tuning

Objective: Optimize hyperparameters for a CNN on the Tiny ImageNet training dataset.

Details: Part 2 README
Key Tasks:

- Define and setup a CNN for the CIFAR-100 dataset.
- Employ a hyperparameter tuning framework.
- Document the tuning process and outcomes.

## Part 3: Model Conversion to TensorRT and ONNX

Objective: Convert and optimize a trained model for fast inference on Nvidia GPUs using TensorRT and ONNX.

Details: Part 3 README
Key Tasks:

- Export a pre-trained model to ONNX format.
- Convert the ONNX model to TensorRT.
- Benchmark and report the inference performance.

