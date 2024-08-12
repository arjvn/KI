import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms as tt

from matplotlib import pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Get transform info
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True)

# Stick all the images together to form a 1600000 X 32 X 3 array
x = np.concatenate([np.asarray(trainset[i][0]) for i in range(len(trainset))])

# calculate the mean and std along the (0, 1) axes
mean = np.mean(x, axis=(0, 1))/255
std = np.std(x, axis=(0, 1))/255
# the the mean and std
mean=mean.tolist()
std=std.tolist()


'''
Step 1: Download dataset & set up data loader
Luckly no custom dataset is needed, we can use CIFAR100 dataset from torchvision
'''

# Define the transformation using the calculated mean and std
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

# define the datasets
train_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8)
test_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=8)


'''
Step 2: Define the model
'''

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
		# Single batch normalization layer after the last conv layer
        self.bn = nn.BatchNorm2d(128)
    
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
		# Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=512)
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

model = SimpleCNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, project_name='My_Project'):
    # Initialize Weights & Biases
    wandb.init(project=project_name, entity='arjvn', name="SimpleCNN")
    wandb.watch(model, criterion, log="gradients", log_freq=100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the appropriate device

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
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
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validation_loss = running_val_loss / len(val_loader)
        validation_accuracy = 100 * correct / total
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

# Assuming model, train_loader, val_loader, criterion, optimizer are defined
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30, project_name='CIFAR_Training')