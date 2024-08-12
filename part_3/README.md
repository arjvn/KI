## READ ME IS YET TO BE UPDATED

FOR SAM

Entry point is part_3/part_3_model_conversion.ipynb

Summary:

SimpleCNN(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=4096, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=256, bias=True)
  (fc3): Linear(in_features=256, out_features=100, bias=True)
)

### findings

🐮 ONNX Run
 >>> Accuracy: 62.99%
 >>> Average inference time per batch: 0.016461 seconds
 >>> Model file size: 10.50 MB

 🐍 PyTorch Run
 >>> Accuracy: 62.99%
 >>> Average inference time per batch: 0.003703 seconds
 >>> Model file size: 21.02 MB

⚡ TensorRT Run
 >>> Accuracy: 62.99%
 >>> Average inference time per batch: 0.000714 seconds
 >>> Model file size: 10.26 MB


## Comparative Analysis of Model Performance Across Different Frameworks

Three frameworks were used to evaluate the performance of a convolutional neural network model on the CIFAR-100 dataset: ONNX, PyTorch, and TensorRT. The model accuracy, inference time, and file size were compared across these frameworks with the following findings:

### Accuracy

All three frameworks achieved an identical accuracy of 62.99%. This consistency across different platforms suggests that the model conversion and execution were correctly implemented, maintaining the integrity of the model predictions.

### Inference Time

- ONNX: Averaged an inference time of 0.016461 seconds per batch.
- PyTorch: Demonstrated a faster performance with an average of 0.003703 seconds per batch.
- TensorRT: Showed the fastest inference times, averaging 0.000714 seconds per batch. This substantial reduction (5.2x) in inference time highlights TensorRT's optimization capabilities, particularly its efficient use of hardware acceleration.

### Model File Size

- ONNX: The model file size was 10.50 MB.
- PyTorch: Had a larger file size at 21.02 MB, likely reflecting the framework's storage of additional state management and computation graph data.
- TensorRT: The model was compacted down to 10.26 MB, slightly smaller than the ONNX file and significantly smaller than the PyTorch file. This indicates efficient serialization of the model suitable for deployment in constrained environments.

### Conclusion

The evaluation illustrates the effectiveness of TensorRT in optimizing model performance for deployment, achieving significantly faster inference times with minimal impact on storage. While PyTorch offers ease of use and flexibility during the development and training phases, TensorRT provides superior performance optimizations for inference, particularly in production environments. ONNX stands as a versatile intermediary that supports model interoperability between various frameworks.

This analysis provides clear insights into the trade-offs between model development convenience and operational efficiency, guiding the choice of framework based on the deployment context and performance requirements.

## ISSUE WITH TENSORRT CODE

 this is the error: 
 [08/12/2024-18:12:52] [TRT] [E] IExecutionContext::enqueueV3: Error Code 3: API Usage Error (Parameter check failed, condition: inputDimensionSpecified && inputShapesSpecified. Not all shapes are specified. Following input tensors' dimensions are not specified: input.)

I observed a few things

1) with batch size = 32

Binding: input, Size: 98304, Dtype: <class 'numpy.float32'>
Binding: output, Size: 3200, Dtype: <class 'numpy.float32'>
[08/12/2024-18:12:52] [TRT] [E] IExecutionContext::enqueueV3: Error Code 3: API Usage Error (Parameter check failed, condition: inputDimensionSpecified && inputShapesSpecified. Not all shapes are specified. Following input tensors' dimensions are not specified: input.)
output shape: (3200,)

why is the output shape (3200,)?? 32 * 100 where 100 is the number of classes

2) with batch size = 1

Binding: input, Size: 3072, Dtype: <class 'numpy.float32'>
Binding: output, Size: 100, Dtype: <class 'numpy.float32'>
[08/12/2024-18:15:36] [TRT] [E] IExecutionContext::enqueueV3: Error Code 3: API Usage Error (Parameter check failed, condition: inputDimensionSpecified && inputShapesSpecified. Not all shapes are specified. Following input tensors' dimensions are not specified: input.)
output shape: (100,)
