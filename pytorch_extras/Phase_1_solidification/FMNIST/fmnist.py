import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Load FashionMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split train into train/validation (80/20)
train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Simple classifier
class FashionNet(nn.Module):
    def __init__(self):
        super(FashionNet, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# Early Stopping class (implemented from scratch)
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

# Training function
def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, reg_type):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=10)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f"\nTraining with {reg_type} (weight_decay={weight_decay})...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Manual L1 regularization (if needed)
            if 'L1' in reg_type and weight_decay > 0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss = loss + weight_decay * l1_penalty
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model)
            break
    
    return train_losses, val_losses, train_accs, val_accs

# Train 4 different models
epochs = 100
results = {}

# 1. No regularization
model_no_reg = FashionNet()
results['No Regularization'] = train_model(model_no_reg, train_loader, val_loader, epochs, lr=0.001, weight_decay=0, reg_type='None')

# 2. L2 regularization (weight_decay)
model_l2 = FashionNet()
results['L2 (weight_decay=0.001)'] = train_model(model_l2, train_loader, val_loader, epochs, lr=0.001, weight_decay=0.001, reg_type='L2')

# 3. L2 stronger
model_l2_strong = FashionNet()
results['L2 (weight_decay=0.01)'] = train_model(model_l2_strong, train_loader, val_loader, epochs, lr=0.001, weight_decay=0.01, reg_type='L2')

# 4. L1 regularization (manual)
model_l1 = FashionNet()
results['L1 (weight_decay=0.0001)'] = train_model(model_l1, train_loader, val_loader, epochs, lr=0.001, weight_decay=0.0001, reg_type='L1')

# Test all models
print("\n" + "="*70)
print("FINAL TEST SET RESULTS:")
print("="*70)

for name, (model, data) in zip(results.keys(), 
                                [(model_no_reg, results['No Regularization']),
                                 (model_l2, results['L2 (weight_decay=0.001)']),
                                 (model_l2_strong, results['L2 (weight_decay=0.01)']),
                                 (model_l1, results['L1 (weight_decay=0.0001)'])]):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f"{name:30s} | Test Accuracy: {test_acc:.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training Loss
for name, (train_losses, val_losses, _, _) in results.items():
    axes[0, 0].plot(train_losses, label=name, alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Validation Loss (KEY: shows overfitting)
for name, (_, val_losses, _, _) in results.items():
    axes[0, 1].plot(val_losses, label=name, alpha=0.7)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Validation Loss (Overfitting Check)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Training Accuracy
for name, (_, _, train_accs, _) in results.items():
    axes[1, 0].plot(train_accs, label=name, alpha=0.7)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_title('Training Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Validation Accuracy
for name, (_, _, _, val_accs) in results.items():
    axes[1, 1].plot(val_accs, label=name, alpha=0.7)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].set_title('Validation Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'regularization_comparison.png'")
plt.show()

print("\nKEY INSIGHTS:")
print("- Watch validation loss: if it increases while training decreases = overfitting")
print("- L2 regularization (weight_decay) penalizes large weights")
print("- L1 regularization creates sparse weights (many become exactly 0)")
print("- Early stopping prevents wasting time on overfitted epochs")