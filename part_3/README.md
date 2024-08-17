# Results and Observations 📊

Entry point is part_3/part_3_model_conversion.ipynb

### 🐮 ONNX Run

- Accuracy: 62.99%
- Average inference time per batch: 0.016461 seconds
- Model file size: 10.50 MB

### 🐍 PyTorch Run

- Accuracy: 62.99%
- Average inference time per batch: 0.003703 seconds
- Model file size: 21.02 MB

### ⚡ TensorRT Run

- Accuracy: 62.99%
- Average inference time per batch: 0.000714 seconds
- Model file size: 10.26 MB

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

## Issue faced with tensorrt 10.0 inference code

There has been a recent update to Tensorrt 10.0 and I am facing some issue with my inference code with my generated tensorrt model file. I am confident that my file is generated correctly and that it is a parameter issue somewhere in my inference code. The bug is definitely just once function or parameter that I am entering incorrectly...

The error message tells me that I am not entering my Input shape but I have! - atleast to my understanding of the updated Tensorrt 10.0 docs. I have even tried digging through the source code to try and figure out what is it that I am doing wrong.

Regardless, I have been able to complete the task with the help of a library which abstracts the inference code for me. So the results presented above stand. This exercise could have easily be done with tensorrt 8.0 as well since it very well documented, however I made the decision when I started the exercise that I would try out the new tensorrt 10.0 library.

this is the error: 

[08/12/2024-18:12:52] [TRT] [E] IExecutionContext::enqueueV3: Error Code 3: API Usage Error (Parameter check failed, condition: inputDimensionSpecified && inputShapesSpecified. Not all shapes are specified. Following input tensors' dimensions are not specified: input.)

I observed a few things

### with batch size = 32

- Binding: input, Size: 98304, Dtype: <class 'numpy.float32'>
- Binding: output, Size: 3200, Dtype: <class 'numpy.float32'>

[08/12/2024-18:12:52] [TRT] [E] IExecutionContext::enqueueV3: Error Code 3: API Usage Error (Parameter check failed, condition: inputDimensionSpecified && inputShapesSpecified. Not all shapes are specified. Following input tensors' dimensions are not specified: input.)
output shape: (3200,)

why is the output shape (3200,)?? 32 * 100 where 100 is the number of classes?

### with batch size = 1

- Binding: input, Size: 3072, Dtype: <class 'numpy.float32'>
- Binding: output, Size: 100, Dtype: <class 'numpy.float32'>
[08/12/2024-18:15:36] [TRT] [E] IExecutionContext::enqueueV3: Error Code 3: API Usage Error (Parameter check failed, condition: inputDimensionSpecified && inputShapesSpecified. Not all shapes are specified. Following input tensors' dimensions are not specified: input.)
output shape: (100,)
