# Instructions

## Part 1: Model Quantisation and Benchmarking

Objective: Take a complex computer vision model from Torch Hub, quantise it, and benchmark the speed of inference on the test subset of tiny-ImageNet dataset.

To download the dataset, use the following utility script:

python ./utils/download_tiny_imagenet.py
Instructions 📃
Select a complex computer vision model from Torch Hub (e.g., Dinov2).
Prepare a small subset of ImageNet images for inference.
Apply dynamic quantisation to the model.
Measure and compare the inference time of the original and quantized models.
Report the inference times and any differences in accuracy.
Submission 💻
Python script with code for loading, quantising, and benchmarking the model.
A brief analysis report (ipynb, markdown or PDF) with inference time comparisons and accuracy differences.

------------------------------------------------------------------------------

## Part 2: Automated Hyperparameter Tuning

Objective: Conduct automated hyperparameter tuning to identify the optimal hyperparameters for a small CNN trained on the tiny-ImageNet dataset training dataset. (Refer to instructions in Part 1 for downloading the data)

Instructions 📃
Define a small CNN architecture of choice for the CIFAR-100 dataset.
Set up a training loop for the CNN model.
Choose hyperparameters to tune (e.g., learning rate, batch size, number of layers, etc.).
Use a hyperparameter optimization library (e.g., Optuna, Hyperopt, or Scikit-Optimize) to find the best hyperparameters.
Train the model using the optimal hyperparameters and report the final accuracy.
Submission 💻
Python script with the model definition, training loop, and hyperparameter tuning setup.
A brief report (markdown or PDF) detailing the hyperparameter tuning process and final model accuracy.

------------------------------------------------------------------------------

## Part 3: Model Conversion to TensorRT and ONNX

Objective: Convert a trained model to TensorRT format and serialize it in ONNX for fast inference on Nvidia GPUs.

Instructions 📃
Train or use a pre-trained model (it can be the model from Part 1 or another model).
Export the model to ONNX format.
Convert the ONNX model to TensorRT using TensorRT tools.
Measure the inference time of the TensorRT model on an Nvidia GPU.
Report the inference times and any speedup achieved.
Submission 💻
Python script with code for model training/loading, ONNX export, and TensorRT conversion.
A brief report (markdown or PDF) with inference time benchmarks and any observed improvements.
General Submission Guidelines
Ensure all code is well-documented and follows best practices.
Include a requirements.txt file with all dependencies required to run your code.
Submit your code and reports in a zip file or through a GitHub repository link.
Evaluation Criteria
Correctness: Does the code achieve the desired outcomes?
Efficiency: Are the implementations optimized for performance?
Clarity: Is the code well-structured and documented?
Reporting: Are the reports clear and do they adequately explain the results?
Good luck, and we look forward to seeing your solutions!