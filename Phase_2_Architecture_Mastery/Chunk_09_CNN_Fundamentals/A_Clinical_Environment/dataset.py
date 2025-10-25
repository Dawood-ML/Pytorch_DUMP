import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

def get_cifar10_dataloaders(batch_size=64, validation_split=0.1, data_dir='./data'):
    """
    Prepares CIFAR-10 train, validation, and test DataLoaders.
    """
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load the full training dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    
    # Create the test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    # Split the full training dataset into training and validation sets
    num_train = len(full_train_dataset)
    val_size = int(num_train * validation_split)
    train_size = num_train - val_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Full training set size: {num_train}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(batch_size=128)