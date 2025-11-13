# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser
from tqdm import tqdm
import time

# Import our custom modules
from dataset import get_dataloaders
from model import ResNet18

# Argument Parsing
# This allows us to run the script from the command line with different settings
def parse_args():
    parser = ArgumentParser(description="Train a ResNet-18 on Cifar-10")
    parser.add_argument('--epochs', type=int, default=20, help='Number of Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save the best model')
    return parser.parse_args()

# Main training script
def main():
    args = parse_args()
    # 1. device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device : {device}")

    # 2. Loading the data
    print('\nNow loading the data')
    train_loader, val_loader, _ = get_dataloaders(args.batch_size) # Only train and val. not test loader
    print("data loaded successfully")

    # 3. Initialize the model, loss, optimizer
    print("\nCreating Model Instance now")
    model = ResNet18().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    # Using SGD with momentum is a classic choice for ResNets.
    # Weight decay is a form of L2 regularization
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=5e-4)
    
    # Learning Rate Schedular
    # Reduces the learning rate when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  mode='min',
                                  factor=0.1,
                                  patience=25,
                                  verbose=True)
    
    print('Model Initialized')

    # 4. training Loop
    best_val_accuracy = 0.0
    start_time = time.time()
    print("Starting training")
    for epoch in range(args.epochs):
        # Training phase
        model.train() # set the model to training mode
        running_loss  = 0.0
        correct_train = 0
        total_train   = 0

        # Using tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch +1}/{args.epochs} [Train]")

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            # Backward pas and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update tqdm progress bar description
            train_pbar.set_postfix({'loss': loss.item(), 'acc': f"{(predicted == labels).sum().item()/labels.size(0):.3f}"})
        
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train

    # Validation

    model.eval() # Set the model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")

    with torch.no_grad(): # No Gradient tracking during validation
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, dim=1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            val_pbar.set_postfix({'loss': loss.item(), 'acc' : f"{(predicted == labels).sum().item()/labels.size(0):.3f}"})
    val_loss /= len(val_loader.dataset)
    val_accuracy = 100 * correct_val / total_val

    # Update LR schedular based on validation loss
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    #   Save the best model
    # We will save the best model's state dict if it has the best validation accuracy so far.
    if val_accuracy > best_val_accuracy:
        print(f"Validation accuracy improved ({best_val_accuracy:.2f}%  - --  > {val_accuracy:.2f}%). Saving model")
        torch.save(model.state_dict(), args.save_path)
        best_val_accuracy = val_accuracy

    end_time =  time.time()
    print("\nTraining Finished")
    
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    main()
