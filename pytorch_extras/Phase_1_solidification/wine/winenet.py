import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Using device : {device}")

wine = load_wine()
X,y = wine.data, wine.target

print(f"Wine dataset : {X.shape[0]} samples, {X.shape[1]} features,\n{len(set(y))} classes")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("")
print(f"Total shape of X : {X.shape}")
print(f"Total shape of y : {y.shape}")
print("")
print(f"Shape of X_train : {X_train.shape}")
print(f"Shape of y_train : {y_train.shape}")
print()
print(f"Shape of X_val : {X_val.shape}")
print(f"Shape of y_val : {y_val.shape}")
print()
print(f"Shape of X_test : {X_test.shape}")
print(f"Shape of y_test : {y_test.shape}")

# Normalize data
scaler  = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Convert to Pytorch
X_train_t =  torch.FloatTensor(X_train).to(device)
X_val_t   =  torch.FloatTensor(X_val).to(device)
X_test_t  =  torch.FloatTensor(X_test).to(device)
y_train_t =  torch.LongTensor(y_train).to(device)
y_val_t   =  torch.LongTensor(y_val).to(device)
y_test_t  =  torch.LongTensor(y_test).to(device)


class WineNet(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(WineNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_units, hidden_units//2),
            nn.ReLU(),
            nn.Linear(hidden_units//2, 3)
        )

    def forward(self, x):
        return self.network(x)


def train_and_evaluate(lr, hidden_units, epochs=100, verbose=True):
    model = WineNet(input_dim=X_train.shape[1], hidden_units=hidden_units).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    train_dataset  = TensorDataset(X_train_t, y_train_t)
    train_loader   = DataLoader(train_dataset, shuffle=True,
                                batch_size=128)
    best_val_acc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()
            outputs  = model(X_batch)
            loss     = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs  = model(X_val_t)
            _, predicted = torch.max(val_outputs.data, 1)
            val_acc = 100 * (predicted == y_val_t).sum().item() / len(y_val_t)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    # Test Accuracy
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs.data, 1)

        test_acc = 100 * (predicted == y_test_t).sum().item() / len(y_test_t)


    if verbose:
        print(f"LR = {lr:.5f}, Hidden Unites = {hidden_units:3d} | Val : {best_val_acc:.2f}% | test: {test_acc:.2f}")
    return best_val_acc, test_acc


# Grid Search Setup

learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
hidden_units_options = [32, 64, 128, 256]

print("\n" + "="*80)
print("GRID SEARCH: Testing all combinations...")
print("="*80)
print(f"Total combinations: {len(learning_rates)} LRs × {len(hidden_units_options)} Hidden Units = {len(learning_rates) * len(hidden_units_options)}")
print()

# Run Grid Search
results = []
for lr, hidden_units in itertools.product(learning_rates, hidden_units_options):
    val_acc, test_acc = train_and_evaluate(lr, hidden_units, epochs=100, verbose=True)
    results.append({
        'learning_rate': lr,
        'hidden_units': hidden_units,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc
    })

df = pd.DataFrame(results)
best_idx = df['val_accuracy'].idxmax()
best_config = df.loc[best_idx]


print("\n" + "="*80)
print("BEST CONFIGURATION (based on validation accuracy):")
print("="*80)
print(f"Learning Rate:   {best_config['learning_rate']}")
print(f"Hidden Units:    {int(best_config['hidden_units'])}")
print(f"Val Accuracy:    {best_config['val_accuracy']:.2f}%")
print(f"Test Accuracy:   {best_config['test_accuracy']:.2f}%")
print("="*80)

# Detailed results table
print("\nFULL RESULTS (sorted by validation accuracy):")
print(df.sort_values('val_accuracy', ascending=False).to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap: Val Accuracy
pivot_val = df.pivot(index='hidden_units', columns='learning_rate', values='val_accuracy')
sns.heatmap(pivot_val, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[0], cbar_kws={'label': 'Val Accuracy (%)'})
axes[0].set_title('Validation Accuracy Heatmap', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Learning Rate')
axes[0].set_ylabel('Hidden Units')

# Heatmap: Test Accuracy
pivot_test = df.pivot(index='hidden_units', columns='learning_rate', values='test_accuracy')
sns.heatmap(pivot_test, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[1], cbar_kws={'label': 'Test Accuracy (%)'})
axes[1].set_title('Test Accuracy Heatmap', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Learning Rate')
axes[1].set_ylabel('Hidden Units')

plt.tight_layout()
plt.savefig('grid_search_results.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'grid_search_results.png'")
plt.show()

# Additional analysis
print("\n" + "="*80)
print("INSIGHTS FROM GRID SEARCH:")
print("="*80)

# Best learning rate across all hidden units
best_lr = df.groupby('learning_rate')['val_accuracy'].mean().idxmax()
print(f"Best average learning rate: {best_lr}")

# Best hidden units across all LRs
best_hidden = df.groupby('hidden_units')['val_accuracy'].mean().idxmax()
print(f"Best average hidden units: {int(best_hidden)}")

# Worst performers
worst_idx = df['val_accuracy'].idxmin()
worst_config = df.loc[worst_idx]
print(f"\nWorst config: LR={worst_config['learning_rate']}, Hidden={int(worst_config['hidden_units'])}, Val Acc={worst_config['val_accuracy']:.2f}%")

print("\nKEY TAKEAWAYS:")
print("- Grid search is exhaustive but slow (grows exponentially with hyperparameters)")
print("- Always use validation set to pick best config, then report test accuracy")
print("- For small datasets like Wine, even 'bad' configs can get decent accuracy")
print("- Manual logging helps you understand which hyperparameters matter most")