'''
Step 1: Read & understand the data
'''

import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou

import matplotlib.patches as patches
from matplotlib import pyplot as plt

import wandb
from tqdm import tqdm

from db_tools import id_to_name


'''
Step 2: Data Preprocessing & Data loader
'''
import os
from PIL import Image
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TinyImageNetDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Path to the dataset directory.
        transform: Optional transforms to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.bboxes = []
        self.labels = []

        self.classes = sorted(os.listdir(root_dir))
        # ids_to_ints = parse_ids_to_int('data/tiny-imagenet/words.txt')
        ids_to_ints = {}
        list_classes = os.listdir("data/tiny-imagenet/train")

        for cc, cls in enumerate(list_classes):
            ids_to_ints[cls] = cc
        
        # Iterate over all class folders, ensuring they are directories
        cc = 0
        for class_id in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_id, 'images')
            bbox_file = os.path.join(self.root_dir, class_id, f'{class_id}_boxes.txt')
            
            if os.path.isdir(class_dir) and os.path.isfile(bbox_file):  # Ensure both are valid: helps ensure that no corrupted or not needed files are read eg. .DSTORE
                class_int = ids_to_ints[class_id]
                # Read bounding box annotations
                with open(bbox_file, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        filename = parts[0]
                        bbox = tuple(map(int, parts[1:]))  # Convert strings to integers
                        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                            cc += 1
                            logger.debug(f"{cc}: Invalid bounding box found in {bbox_file} for {filename}.")

                            continue  # Skip this bounding box
                        self.images.append(os.path.join(class_dir, filename))
                        self.bboxes.append(bbox)
                        self.labels.append(int(class_int))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        
        # Get bounding box and label
        bbox = self.bboxes[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        # Create target dictionary expected by Faster R-CNN
        target = {}
        target['boxes'] = torch.tensor([bbox], dtype=torch.float32)  # Faster R-CNN expects float type
        target['labels'] = torch.tensor([label], dtype=torch.int64)  # Ensure labels are torch.int64

        return image, target
    
def save_checkpoint(state, is_best, epoch, folder='checkpoints'):
    """Save checkpoint if a new best is achieved"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = os.path.join(folder, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, filename)  
    
    if is_best:
        best_filename = os.path.join(folder, 'best_model.pth')
        torch.save(state, best_filename)
        
def calculate_ap(iou, thresholds=[0.5]):
    """A simple AP calculation implementation"""
    # Assume binary classification: IoU above threshold is a TP
    true_positives = (iou >= thresholds[0]).float()
    false_positives = (iou < thresholds[0]).float()
    # Simplified AP calculation: TP / (TP + FP)
    return true_positives.sum() / (true_positives.sum() + false_positives.sum())

'''
Step 3: Training the model
'''


'''
model definition 1
'''

# Load a pre-trained model for classification and return only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280  # MobilenetV2's feature size for FPN

# Define RPN (Region Proposal Network)
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=(0.5, 1.0, 2.0))

# Define RoI (Region of Interest) pooling for feature maps to bounding box transformation
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# Piece together the model using the parts
my_model = FasterRCNN(backbone,
                   num_classes=201+1,  # including the background
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

'''
model definition 2: train from scratch
'''

# model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=len(train_dataset.classes) + 1)

'''
model definition 3: resnet 50 backbone 1
'''
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_faster_rcnn_resnet50_fpn(num_classes, pretrained_backbone=True):
    backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone)
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=num_classes)
    return model

# model = get_faster_rcnn_resnet50_fpn(num_classes = len(train_dataset.classes) + 1, pretrained_backbone=True)

'''
mdeol definition 4: resnet 50 backbone 2
'''

from torchvision.models.detection.rpn import AnchorGenerator

# Load a pre-trained model for classification and return only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280  # MobilenetV2's feature size for FPN

# Define RPN (Region Proposal Network)
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=(0.5, 1.0, 2.0))

# Define RoI (Region of Interest) pooling for feature maps to bounding box transformation
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# Piece together the model using the parts
# model = FasterRCNN(backbone,
#                    num_classes=201,  # including the background
#                    rpn_anchor_generator=anchor_generator,
#                    box_roi_pool=roi_pooler)

'''
training loop
'''
EPOCHS = 10

# Initialize a new run

config={
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "epochs": EPOCHS,
    "batch_size": 8
}

wandb.init(project="tiny-imagenet-detection", entity="arjvn", config=config)

transform = transforms.Compose([
    transforms.Resize((800, 800)),  # Adjust size depending on model requirements
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = TinyImageNetDataset(root_dir='data/tiny-imagenet/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

# model = get_faster_rcnn_resnet50_fpn(num_classes = len(train_dataset.classes) + 1, pretrained_backbone=True)
model = FasterRCNN(backbone,
                   num_classes=len(train_dataset.classes) + 1,  # including the background
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
wandb.watch(model, log='all')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f"Using device: {device}")
model.to(device)
model.train()

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    running_loss = 0.0
    iou_scores = []

    for images, targets in tqdm(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        # Calculate IoU and accumulate data for AP calculation
        for i, target in enumerate(targets):
            gt_boxes = target['boxes']
            pred_boxes = model(images[i])['boxes']
            iou = box_iou(gt_boxes, pred_boxes)
            iou_scores.append(iou)

        wandb.log({"batch_loss": losses.item()})
        wandb.log({"batch iou": torch.cat(iou_scores).mean()})
    
    avg_loss = running_loss / len(data_loader)
    ap = calculate_ap(torch.cat(iou_scores))
    wandb.log({"epoch_loss": avg_loss})
    wandb.log({"AP": ap})
    
    return avg_loss, ap

best_loss = float('inf')

for epoch in range(EPOCHS):
    epoch_loss = train_one_epoch(model, optimizer, train_loader, device)
    logger.info(f'Epoch: {epoch+1}, Loss: {epoch_loss}')
    wandb.log({"Epoch": epoch + 1, "Loss": epoch_loss})

    # Determine if this is the best model so far
    is_best = epoch_loss < best_loss
    best_loss = min(epoch_loss, best_loss)
    
    # Save checkpoint
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': epoch_loss,
    }, is_best, epoch + 1)

wandb.finish()