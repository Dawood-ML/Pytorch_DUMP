import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from tqdm import tqdm
from model import SimpleVGG
from dataset import get_cifar10_dataloaders

def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    pbar = tqdm(train_loader, desc="Training epoch", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(preds.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(loss=loss.item(), accuracy = 100. * correct / total)

    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix(accuracy=100. * correct / total)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(42)

    # Get DataLoaders (now returns train, val, test)
    train_loader, val_loader, _ = get_cifar10_dataloaders(args.batch_size)

    model = SimpleVGG(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        
        print(f"Epoch {epoch+1:02d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving model...")
            torch.save(model.state_dict(), 'best_cifar10_vgg.pth')
            best_val_acc = val_acc
                        
    end_time = time.time()
    print(f"\nTraining finished in {(end_time - start_time)/60:.2f} minutes.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a simple VGG on CIFAR-10.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    args = parser.parse_args()
    main(args) 