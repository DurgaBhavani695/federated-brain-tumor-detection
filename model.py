import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MRIDataset(Dataset):
    """
    Custom Dataset to load grayscale brain MRI images from normal/tumor subfolders.
    """
    def __init__(self, dir_path):
        super().__init__()
        self.samples = []
        self.labels = []
        
        normal_dir = os.path.join(dir_path, "normal")
        tumor_dir = os.path.join(dir_path, "tumor")
        
        if os.path.exists(normal_dir):
            for f in os.listdir(normal_dir):
                if f.endswith('.png'):
                    self.samples.append(os.path.join(normal_dir, f))
                    self.labels.append(0)  # 0 for normal
                    
        if os.path.exists(tumor_dir):
            for f in os.listdir(tumor_dir):
                if f.endswith('.png'):
                    self.samples.append(os.path.join(tumor_dir, f))
                    self.labels.append(1)  # 1 for tumor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load grayscale image
        img = Image.open(img_path).convert('L')
        
        # Convert to numpy array and normalize to [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to float tensor with shape [1, H, W]
        img_tensor = torch.tensor(arr).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return img_tensor, label_tensor


class BrainTumorCNN(nn.Module):
    """
    Convolutional Neural Network for binary brain tumor classification.
    Input size: 1x128x128
    """
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        
        # Conv block 1: 1 -> 8 channels, output: 8x64x64
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv block 2: 8 -> 16 channels, output: 16x32x32
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv block 3: 16 -> 16 channels, output: 16x16x16
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers (16 channels * 16 * 16 = 4096 features)
        self.fc1 = nn.Linear(4096, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x.squeeze(1)


def train_one_epoch(model, data_loader, optimizer, criterion):
    """
    Performs standard single-epoch local training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc


def evaluate_model(model, data_loader, criterion=nn.BCELoss()):
    """
    Evaluates the model on validation or test sets.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for p, l in zip(preds, labels):
                if p == 1 and l == 1:
                    true_positives += 1
                elif p == 1 and l == 0:
                    false_positives += 1
                elif p == 0 and l == 1:
                    false_negatives += 1
                elif p == 0 and l == 0:
                    true_negatives += 1
                    
    loss_val = running_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "loss": float(loss_val),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "tp": int(true_positives),
            "fp": int(false_positives),
            "fn": int(false_negatives),
            "tn": int(true_negatives)
        }
    }


def apply_differential_privacy(original_weights, trained_weights, clip_norm=1.0, noise_multiplier=0.1):
    """
    Applies Local Differential Privacy on client updates.
    Clips the client weight updates (trained - original) and adds Gaussian noise.
    
    Parameters:
        original_weights (dict): State dict of model before local training.
        trained_weights (dict): State dict of model after local training.
        clip_norm (float): Maximum allowed L2 norm of the update.
        noise_multiplier (float): Noise multiplier relative to sensitivity.
        
    Returns:
        dict: The privacy-preserved weight dictionary.
        float: Calculated L2 norm of the update before clipping.
    """
    dp_weights = copy.deepcopy(original_weights)
    
    # 1. Compute updates (deltas) for all layers and calculate global L2 norm of update
    deltas = {}
    sum_sq = 0.0
    for name in original_weights.keys():
        # Only apply to floating point tensors (weights/biases), ignore buffers like running means
        if original_weights[name].is_floating_point():
            delta = trained_weights[name] - original_weights[name]
            deltas[name] = delta
            sum_sq += torch.sum(delta ** 2).item()
        else:
            deltas[name] = None
            
    update_l2_norm = np.sqrt(sum_sq)
    
    # 2. Compute clipping factor
    clip_factor = 1.0
    if update_l2_norm > clip_norm:
        clip_factor = clip_norm / update_l2_norm
        
    # 3. Apply clipping and add Gaussian noise
    for name in original_weights.keys():
        if deltas[name] is not None:
            # Clip
            clipped_delta = deltas[name] * clip_factor
            
            # Generate Gaussian noise matching the parameter shape
            # Sensitivity = clip_norm
            # Standard deviation = noise_multiplier * sensitivity (clip_norm)
            std_dev = noise_multiplier * clip_norm
            noise = torch.normal(mean=0.0, std=std_dev, size=clipped_delta.size())
            
            # Update state dict
            dp_weights[name] = original_weights[name] + clipped_delta + noise
        else:
            # Copy non-floating buffers directly
            dp_weights[name] = trained_weights[name]
            
    return dp_weights, float(update_l2_norm)
