# To set a batch size or not to set a batch size? That is the question

## Understanding EXPLICIT_BATCH

The use of EXPLICIT_BATCH means that the network created by TensorRT will expect the batch size to be explicitly specified when you run an inference session, rather than being fixed at the time of engine creation. This approach offers flexibility, especially useful in production environments where input batch sizes might vary depending on the incoming data.

## Issues with optimisation strategy where batch size is dynamic

The error message you are seeing from TensorRT indicates that the network you are trying to convert has dynamic or shape inputs, and you have not defined an optimization profile for it. In TensorRT, when dealing with networks that have dynamic input sizes, you need to define at least one optimization profile so that TensorRT can optimize the engine accordingly.

Here's how you can address this by creating and setting an optimization profile for your network:

Step 1: Define an Optimization Profile
Optimization profiles specify the minimum, optimal, and maximum dimensions for each dynamic input tensor. This allows TensorRT to prepare optimized kernels for varying input sizes.





## Key Steps and Their Implications:

Initialization:

You initialize the TensorRT Builder and specify the use of EXPLICIT_BATCH. This indicates that your network will explicitly handle batch sizes in the shapes defined by the optimization profile.
Configuration:

You set a memory pool limit for the workspace to 1 MiB, which is very small; usually, a larger size (like 1 GiB) is recommended to give TensorRT enough temporary workspace for its optimization processes.
Optimization Profile:

You create an optimization profile that defines the minimum, optimal, and maximum input shapes for the network:
Minimum Shape: (1, 3, 32, 32) — suitable for single-image inference.
Optimal Shape: (32, 3, 32, 32) — optimized for batch processing of 32 images at once.
Maximum Shape: (64, 3, 32, 32) — the engine can handle up to 64 images at once, but this is the upper limit.
This profile allows the engine to dynamically adjust to different batch sizes within these limits, optimizing performance based on actual usage.
FP16 Optimization:

If the GPU supports fast FP16 computations, this flag is set, allowing TensorRT to use half-precision arithmetic where possible, increasing performance.
Building the Engine:

The ONNX model is parsed, and assuming no errors, a TensorRT engine is built. This engine is serialized and saved to a file, which can then be loaded for inference tasks.
Fixed or Dynamic Batch Size?
Your engine uses a dynamic batch size as defined by the optimization profile. It can handle varying batch sizes from 1 to 64. The actual batch size for inference can be set dynamically based on the input data provided during each inference call.
GPU Compatibility:
GPU Requirements: The engine uses FP16 if available, which is supported by NVIDIA GPUs with Tensor Cores (e.g., Volta, Turing, Ampere architectures).
Compatibility: The engine is generally compatible with any recent NVIDIA GPU that supports the CUDA and TensorRT version used to build the engine. However, performance optimizations are specific to the GPU on which the engine was built. For best performance, rebuild the engine on the target deployment GPU or at least a similar GPU architecture.
Deployment Considerations: While the engine can be deployed on different NVIDIA GPUs, performance and stability are best when the engine is used on the same type of GPU or a similar one to which it was originally built. This is because specific optimizations that TensorRT applies may be tailored to the specific GPU architecture.
