# Model Quantization and Benchmarking

Entry point to code is DINO.ipynb

## Objective

This project aims to apply dynamic quantization to a complex Vision Transformer (DINO model) and benchmark its inference speed and accuracy on a subset of the Tiny ImageNet dataset.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- PyTorch 1.8+
- torchvision
- tqdm

## Dataset Download

Download the Tiny ImageNet dataset using the provided utility script:

```bash
python ./utils/download_tiny_imagenet.py
```

## Dynamic Quantization

Apply dynamic quantization to the Vision Transformer model focusing on linear and convolutional layers, considering their compatibility with quantization techniques. The code below demonstrates this process:

```python
import torch

# Load and prepare the model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
model.eval().cuda()

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
```

## Quantizable Components

Identified components suitable for dynamic quantization include:

- Linear Layers: Fundamental to reducing model size and potentially speeding up operations.
- Convolutional Layers: Similar in behavior to linear layers but applied to spatial data.


## Non-Quantizable Components

Some layers do not benefit from quantization or cannot be quantized effectively:

- Normalization Layers (LayerNorm): These are sensitive to quantization as they rely on statistics that can be skewed by reduced precision.
- Dropout Layers: Non-functional during inference, thus quantization is unnecessary.
- Activation Functions (GELU): Typically require quantization-aware training to maintain effectiveness when quantized.

## Quantisation results

🍩 Original model (gpu):

> Inference Speed: 0.0077 seconds per frame

🍩 Original model (cpu):

> Inference Speed: 0.2014 seconds per frame

🚀 Quantized model 1: Linear Layers only

> Inference Speed: 0.1940 seconds per frame

🌟 Quantized model 2: Linear and Conv Layers

> Inference Speed: 0.1892 seconds per frame


Accuracy results are not shared because the recommended torch dino model is not trained on the Tiny ImageNet model. From experience, I can say that the there is only a slight degradation in model accuracy.

## Understanding Quantization and GPU Limitations

### Why Quantize if GPU is Not Supported?

Quantization primarily benefits CPU inference in several ways:

- Reduced Model Size: Quantization reduces the bit-width of model weights and activations from floating-point representations (typically 32 bits) to lower-precision formats (like INT8, which uses only 8 bits). This reduction can significantly decrease the model size, which is crucial for deploying models in bandwidth or storage-constrained environments.

- Increased Inference Speed on CPUs: Many CPUs can execute INT8 calculations more quickly than floating-point calculations due to hardware optimizations for low-precision arithmetic. This makes quantized models particularly beneficial for real-time applications on edge devices or mobile phones, where GPU resources are not available or limited.

- Lower Power Consumption: Quantization can reduce the power consumption of inference, an important factor in battery-powered devices.

### Limitations

- GPU Support: Current dynamic quantization implementations in PyTorch are limited to CPU. This is because most modern GPUs are optimized for high-throughput 32-bit calculations and do not yet benefit as much from INT8 precision. For scenarios where GPU use is critical, and high precision is required, unquantized models are preferable.

- Performance Trade-offs: While quantization can accelerate inference and reduce model size, it might also lead to a slight degradation in model accuracy, which is a critical consideration depending on the application's tolerance for error.

## Note on Dynamic Quantization

Dynamic quantization is quick and easy to apply as it does not necessitate retraining. However, it generally offers less reduction in model size and smaller speed improvements compared to static quantization or quantization-aware training. For certain layers like activations or custom operations, quantization-aware training might provide a more balanced approach to preserve accuracy.

## Conclusion

This project explores the feasibility and impacts of dynamically quantizing a Vision Transformer model for practical deployment scenarios, offering insights into the trade-offs involved in using this technique on complex neural network architectures.
