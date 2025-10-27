import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os

def get_xray_dataloaders(data_dir, 
                        image_size = 128, 
                        Validation_split = 0.15,
                        batch_size=512):
    normalize = transforms.Normalize(mean=[0.5],
                                     std = [0.5])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(degrees=10,
                                    translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ]
    )
    # Validation and Test Transform MUST NOT include data augmentation
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        normalize
    ])

    # Load full trining data and then split it
    full_train_dataset = datasets.ImageFolder(root=os.path.normpath(os.path.join(data_dir, 'train')),
                                             transform=train_transform)

    # The validation set should have the val_test_transform.
    # This is a bit tricky with random split. A common practice is to split indices and create new Subset
    # datasets 
    num_train = len(full_train_dataset)
    val_size  = int(num_train * Validation_split)
    train_size = num_train - val_size

    # Temporarily set the transform for the validation split
    full_train_dataset.transform = val_test_transform
    val_dataset, _ = random_split(full_train_dataset, [val_size, train_size],
                                  generator=torch.Generator().manual_seed(42))
    
    # Reset transform for the training split
    full_train_dataset.transform = train_transform
    _, train_dataset  = random_split(full_train_dataset, [val_size, train_size],
                                     generator=torch.Generator().manual_seed(42))
    
    test_dataset = datasets.ImageFolder(root=os.path.normpath(os.path.join(data_dir, 'test')),
                                    transform=val_test_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    ) 
    val_loader   = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    DATA_DIR  = './data/chest_xray'
    get_xray_dataloaders(DATA_DIR)