# Imports
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Constants
# These are the standard MEAN and STD for the CIFAR-10 dataset.
# They are computed from the training set and are crucial for Normalization
MEAN        = (0.49, 0.48, 0.44)
STD         = (0.24, 0.24, 0.26)
BATCH_SIZE  = 128
VALID_SPLIT = 0.2   # 20% of the training data will be used for validation

class DatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset=subset
            self.transform = transform
        
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

def get_dataloaders(batch_size = BATCH_SIZE):
    """
    Prepares the Cifar-10 dataset, applies Transformations, Splots the
    training data into training and validation sets, and creates DataLoaders

    Args :
        batch_size (int): The number of samples per batch
    
    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader).
    """
    # 1. Define Transformations : 
    
    #######################################################################

    # As we identified in our EDA, we need to apply transformations.
    # We will only apply augmentation to the training set to help the model
    # generalize better. Validation and test sets should be untouched
    # to get a realistic measure of performance.

    # Transformations for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),    # Adds padding and then randomly crops back to 32x32.
        transforms.RandomHorizontalFlip(),       # Flips the image horizontally with a 50% probability.
        transforms.ToTensor(),                   # Converts PIL Image to tensor and scales values to [0, 1].
        transforms.Normalize(mean=MEAN, std=STD) # Normalizes with dataset's mean and std.
    ])

    transform_test  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    ##############################################


    # 2. Load the Datasets
    
    ##############################################
    # We load full training dataset first.
    full_trainset = datasets.CIFAR10(root='./data',
                                     train=True,
                                     download=True,)
    
    # We load the test dataset
    full_testset = datasets.CIFAR10(root='./data',
                                     train=False,
                                     download=True,)
    #################################################
    
    # 3. Create Train / Validation Split

    ################################################# 
    
    # The "Sacred" split. We will make a validation set from the
    # Full training set to monitor our model's performance on unseen data
    # during training

    num_train_samples = len(full_trainset)
    num_val_samples   = int(VALID_SPLIT * num_train_samples)
    num_train_samples = num_train_samples - num_val_samples

    # Use random_split for a reproducible split.
    # A fixed generator seed ensures we get the same split every time.
    train_subset, val_subset = random_split(
        full_trainset,
        [num_train_samples, num_val_samples],
        generator=torch.Generator().manual_seed(42) # For reproducibility
    )
    ###################################################

    # 4. Apply the correct transforms to the subsets
    # This is a key step. We need to wrap our subsets in a custom class or
    # manually assign the transforms. A simple way is to use a small wrapper.
    # class DatasetWrapper(torch.utils.data.Dataset):
    #     def __init__(self, subset, transform=None):
    #         self.subset=subset
    #         self.transform = transform
        
    #     def __getitem__(self, index):
    #         x, y = self.subset[index]
    #         if self.transform:
    #             x = self.transform(x)
    #         return x, y
    #     def __len__(self):
    #         return len(self.subset)
    
    train_dataset = DatasetWrapper(subset=train_subset, transform=transform_train)
    val_dataset  = DatasetWrapper(subset=val_subset,
                                   transform=transform_test)
    test_dataset = DatasetWrapper(subset=full_testset, transform=transform_test)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(full_testset)}")

    ##########################################################

    # 5. Create DataLoaders
    # DataLoaders handle batching, shuffling, and parallel data loading
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    ################################################
    return train_loader, val_loader, test_loader


# Main block for standalone testing
# THis is a very good thing to do. It enables you to test this script directly 
# an validate it early in project.

if __name__ == '__main__':
    # Get the dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=32)
    print("\n--- Testing the train_loader ---")
    images, labels = next(iter(train_dl))

    # Print shapes to verify
    print(f"Images batch shape: {images.shape}") # Should be [batch_size, 3, 32, 32]
    print(f"Labels batch shape: {labels.shape}") # Should be [batch_size]

    # Check the data range (should be normalized, i.e., not [0, 1])
    print(f"Min pixel value: {images.min():.2f}")
    print(f"Max pixel value: {images.max():.2f}")
    print(f"Mean pixel value: {images.mean():.2f}") # Should be close to 0
    print(f"Std pixel value: {images.std():.2f}")   # Should be close to 1

# SUMMARY BELOW
#
######################################################################################
######################################################################################


# Constants: We define the pre-computed MEAN and STD for CIFAR-10. Hard-coding these is standard practice. 
# Normalizing with the correct statistics is critical for model performance, especially when using pre-trained models later.



# Separate Transforms: Note the crucial difference between transform_train and transform_test. 
# We only augment the training data. 
# Augmenting validation/test data would give us an artificially inflated and incorrect measure of our model's true performance.



# Reproducible Split: Using torch.utils.data.random_split with a fixed generator seed is the professional way to create our validation set. 
# This guarantees that our validation set remains consistent across all experiments, which is essential for fair model comparison.



# DatasetWrapper: The random_split function returns Subset objects which don't have a .transform attribute. 
# By creating this small wrapper class, we can cleanly apply the correct set of transformations to our new train_dataset and val_dataset.



# DataLoader Best Practices: We use shuffle=True only for the training loader. 
# num_workers=2 uses subprocesses to load data in the background, preventing the GPU from waiting for data. 
# pin_memory=True can speed up data transfer from CPU to GPU.



# if __name__ == '__main__':: This block is your sanity check. 
# Running python dataset.py in your terminal should execute this code and show you that 
# your data pipeline is working as expected before you even start writing the model or training loop.