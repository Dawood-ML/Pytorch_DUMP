import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import Counter
from tqdm import tqdm

# UPGRADED: We need sklearn to calculate better metrics
from sklearn.metrics import f1_score

from model import VGG_XRay
from dataset_xray import get_xray_dataloaders

def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc="Training Epoch", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix(loss=loss.item(), accuracy=f"{100 * correct / total:.2f}%")
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# UPGRADED: The validate function now calculates F1-score
def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    # CHANGED: Calculate macro F1-score, which is robust to imbalance
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    # We can still calculate accuracy for informational purposes
    epoch_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    return epoch_loss, epoch_acc, epoch_f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, _ = get_xray_dataloaders(args.data_dir, batch_size=args.batch_size)

    train_labels = [train_loader.dataset.dataset.samples[i][1] for i in train_loader.dataset.indices]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(class_counts)
    class_weights = torch.tensor([total_samples / (num_classes * class_counts[i]) for i in range(num_classes)], dtype=torch.float32).to(device)
    print(f"Class counts in training set: {class_counts}")
    print(f"Calculated class weights: {class_weights}")

    model = VGG_XRay(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # CHANGED: We track the best F1-score, not accuracy
    best_val_f1 = 0.0
    
    print("Starting X-Ray model training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        
        # CHANGED: validate now returns f1_score
        val_loss, val_acc, val_f1 = validate(model, device, val_loader, criterion)
        
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1-Score: {val_f1:.4f}")
        
        # CHANGED: The condition for saving the model is now based on the F1-score
        if val_f1 > best_val_f1:
            print(f"Validation F1-score improved from {best_val_f1:.4f} to {val_f1:.4f}. Saving model...")
            torch.save(model.state_dict(), 'best_xray_vgg.pth')
            best_val_f1 = val_f1
            
    print(f"\nTraining finished. Best validation F1-Score: {best_val_f1:.4f}")
    print("Model saved to 'best_xray_vgg.pth'. Run inference notebook for final evaluation.")

if __name__ == '__main__':
    # ... (parser arguments remain the same) ...
    parser = argparse.ArgumentParser(description='Train an adapted VGG on Chest X-Ray data.')
    parser.add_argument('--data_dir', type=str, default='./data/chest_xray', help='Path to dataset.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    args = parser.parse_args()
    main(args)