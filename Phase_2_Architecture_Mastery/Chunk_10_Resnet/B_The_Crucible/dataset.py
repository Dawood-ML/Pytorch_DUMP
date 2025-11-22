"""
Chunk 10 Crucible: CIFAR-100 Professional Data Pipeline
-----------------------------------------------------

This module demonstrates the 'Custom Dataset Pattern'. 
Instead of relying on pre-built wrappers, we define our own class.
This allows us to:
1. Split data INDICES first (Train vs Val).
2. Assign DIFFERENT transforms to the same underlying data source.
3. Debug exactly what goes into the model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Constants
MEAN = [0.5071, 0.4867, 0.4408] # We calculated these in eda.ipynb
STD  = [0.2675, 0.2565, 0.2761] # We calculated these in eda.ipynb

class CIFAR_100_CUSTOM(Dataset):
    """
    A professional wrapper around data arrays

    In a real project (e:g Medical imaging), 'data' might be a list of file paths
    ['img1.png', 'img2.png] and 'targets' might be [0, 1]

    Here, 'data' is the numpy array of CIFAR images [N, 32, 32, 3]
    """

    def __init__(self, data, targets, transform=None):
        """
        Args:
            data (np.ndarray): The image data
            targets (list/np.ndarray): The Labels
            transform (callable, optional): the transforms to apply
        """
        self.data      = data
        self.targets   = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        The most critical function.
        Fetches one sample, applies transforms, returns tensor
        """
        # 1. Get raw data
        img = self.data[idx] # shape: [32, 32, 3], Type: uint8
        target = self.targets[idx]

        # 2. Apply transform (if any)
        if self.transform:
            img = self.transform(img)

        # 3. Return tuple
        return img, target
    
def get_dataloaders(
        data_root = './data',
        batch_size=128,
        num_workers=4,
        val_split=0.1):
    # 2. Define Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(), # Often needed when starting from numpy arrays
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)        
    ])

    eval_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
    )
    
    # Load RAW data (Source)
    # We will ise torchvision just to download and extract the bits to disk/
    # We do not use the dataset object it returns for training directly
    raw_train = datasets.CIFAR100(root=data_root, train=True, download=True)
    raw_test  = datasets.CIFAR100(root=data_root, train=False, download=True)

    # Extract the underlying numpy arrays
    # # In a custom project, this is where you'd read your csv or glob your file paths

    X_train_full = raw_train.data # [50000, 32, 32, 3]
    y_train_full = raw_train.targets # [5000]

    X_test = raw_test.data
    y_test = raw_test.targets


    # 4. Stratified Split (professional practice)
    # We split the ARRAYS, not the dataset objects
    # Stratify ensures equal class distribution in Train and Val (Crucial for small classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_split,
        stratify=y_train_full,
        random_state=42
    )
    print(f"Train Shape : {X_train.shape}")
    print(F"Val Shape : {X_val.shape}")

    # 5. INstantiate Custom Classes
    train_dataset = CIFAR_100_CUSTOM(X_train, y_train, transform=train_transform)
    val_dataset   = CIFAR_100_CUSTOM(X_val, y_val, transform=eval_transform)
    test_dataset  = CIFAR_100_CUSTOM(X_test, y_test, transform=eval_transform)

    # 6. DataLoaders
    train_loader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=True)
    val_loader   = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    
    return train_loader,val_loader,  test_loader, raw_train.classes

# SAnity check 
if __name__ == "__main__":
    train, val, test, classes = get_dataloaders()
    img, label = next(iter(train))
    print(f"Output Tensor Shape: {img.shape}")
    print("Custom Dataset Pipeline Verified.")    
        
    