import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms as tt
from torch.utils.data import DataLoader

import os
import json
import wandb
import numpy as np
from torchviz import make_dot
from matplotlib import pyplot as plt
from tqdm import tqdm

import logging

'''
Step 0: Set up Weights & Biases, logging, get trasnform info and save model visualization
'''

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Get wandb sweep config
with open("sweep_config.json", "r") as file:
	sweep_config = json.load(file)

sweep_id = wandb.sweep(sweep_config, entity='arjvn', project='SimpleCNN')

# Get transform info
def get_transform_info():
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True)

    x = np.concatenate([np.asarray(trainset[i][0]) for i in range(len(trainset))])
    mean = np.mean(x, axis=(0, 1))/255
    std = np.std(x, axis=(0, 1))/255
    # the the mean and std
    mean=mean.tolist()
    std=std.tolist()

    return mean, std

def create_dot_graph(model):
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 32, 32)
    model_output = model(dummy_input)
    dot = make_dot(model_output, params=dict(model.named_parameters()))
    dot.render('SimpleCNN_visualization', format='png')  # This will save the visualization as a PNG file


'''
Step 1: Download dataset & set up data loader
Luckily no custom dataset is needed, we can use CIFAR100 dataset from torchvision
'''

# Define the transformation using the calculated mean and std

mean, std = get_transform_info()
transform_train = tt.Compose([
                         tt.RandomCrop(32, padding=4,padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(mean,std,inplace=True)
                         ])

transform_test = tt.Compose([
                            tt.ToTensor(), 
                             tt.Normalize(mean,std)
                             ])

# define the datasets
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
'''
Step 2: Define the model
'''

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=config.num_filters_1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=config.num_filters_1, out_channels=config.num_filters_2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=config.num_filters_2, out_channels=config.num_filters_3, kernel_size=3, padding=1)
        
		# Single batch normalization layer after the last conv layer
        self.bn = nn.BatchNorm2d(config.num_filters_3)
    
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Fully connected layers
        final_size = config.num_filters_3 * 4 * 4
        self.fc1 = nn.Linear(in_features=final_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=100)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.bn(self.conv3(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
Step 3: Define the training loop
'''

def train_model(config=None, watch_model=False):
    # Initialize Weights & Biases
    with wandb.init(config=config):
        config = wandb.config

        # Setup directories for saving models
        model_dir = os.path.join(wandb.run.dir, "models")
        best_model_dir = os.path.join(wandb.run.dir, "best_model")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)
        wandb.log({"model_dir": model_dir, "best_model_dir": best_model_dir})

        # Initialize the best accuracy - to save the best model
        best_val_accuracy = 0

        # Initialize the data loaders
        train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=8)

        # Initialize the model
        model = SimpleCNN(config)
        create_dot_graph(model, config)
        
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

        if watch_model: # large overhead - turn off if not needed
            wandb.watch(model, criterion, log="gradients", log_freq=100)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in tqdm(range(config.epochs), desc="Epochs"):
            # Training Phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            logging.info(f'Epoch {epoch+1}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy:.2f}%')

            # Validation Phase
            model.eval()
            running_val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    val_loss = criterion(outputs, labels)
                    running_val_loss += val_loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            validation_loss = running_val_loss / len(test_loader)
            validation_accuracy = 100 * correct / total

            # Save the model every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch+1}.pth'))

            # Check if it is the best model
            if validation_accuracy > best_val_accuracy:
                best_val_accuracy = validation_accuracy
                best_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_accuracy': validation_accuracy
                }
                torch.save(best_checkpoint, os.path.join(best_model_dir, 'best_model.pth'))
            logging.info(f'Epoch {epoch+1}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy:.2f}%')

            # Log metrics to Weights & Biases
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'validation_loss': validation_loss,
                'validation_accuracy': validation_accuracy
            })
            
        wandb.finish()

'''
Step 4: Run the training loop via the wandb agent
'''

# train_model(num_epochs=30, watch_model=True)
wandb.agent(sweep_id, lambda: train_model(watch_model=True), count=30)