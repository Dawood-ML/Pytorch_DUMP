import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else 'cpu'


print("Loading Covtype dataset...")
covtype = fetch_covtype()
X, y = covtype.data[:50000], covtype.target[:50000] - 1  # Subset for speed

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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


# Normalize
scalar = StandardScaler()
X_train  = scalar.fit_transform(X_train)
X_val    = scalar.transform(X_val)
X_test   = scalar.transform(X_test)

# Convert to Pytorch
X_train_t = torch.FloatTensor(X_train).to(device)
X_val_t   = torch.FloatTensor(X_val).to(device)
X_test_t  = torch.FloatTensor(X_test).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
y_val_t   = torch.LongTensor(y_val).to(device)
y_test_t  = torch.LongTensor(y_test).to(device)

print()
print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
print(f"Features: {X_train.shape[1]}, Classes: {len(set(y))}\n")
print()


# Flexible network architecture
class FlexibleNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_units, dropout_rate, activation):
        super(FlexibleNet, self).__init__()

        layers = []
        in_features = input_dim

        # Build hidden layers
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_features=in_features, out_features=hidden_units))

            if activation   == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())

            layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.Dropout(dropout_rate))

            in_features  = hidden_units
            hidden_units = hidden_units // 2 # Decrease size each layer

        # Output layer
        layers.append(nn.Linear(in_features, 7))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
# Objective function for optuna

def objective(trial):
    # suggest hyperparameters
    lr            = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    hidden_units  = trial.suggest_categorical('hidden_units', [64, 128, 256, 512])
    dropout_rate  = trial.suggest_float('dropout_rate', 0.1, 0.6)
    weight_decay  = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    activation    = trial.suggest_categorical('activation', ['relu', 'tanh','leaky_relu'])

    # Build Model
    model = FlexibleNet(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        activation=activation
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True)
    
    # Train for limited epochs
    epochs = 30
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            _, predicted = torch.max(val_outputs, 1)
            val_acc = 100 * (predicted == y_val_t).sum().item() / len(y_val_t)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        trial.report(val_acc, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    return best_val_acc


# Run Optuna optimization
print("="*80)
print("OPTUNA HYPERPARAMETER OPTIMIZATION")
print("="*80)
print("Optuna will intelligently search the hyperparameter space.")
print("Bad trials get pruned early to save time.\n")

study = optuna.create_study(
    direction='maximize',
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
)

# Optimize
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Results
print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial: #{study.best_trial.number}")
print(f"Best validation accuracy: {study.best_value:.2f}%")
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key:20s}: {value}")

# Train final model with best hyperparameters
print("\n" + "="*80)
print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS...")
print("="*80)

best_params = study.best_params
final_model = FlexibleNet(
    input_dim=X_train.shape[1],
    hidden_layers=best_params['hidden_layers'],
    hidden_units=best_params['hidden_units'],
    dropout_rate=best_params['dropout_rate'],
    activation=best_params['activation']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    final_model.parameters(),
    lr=best_params['lr'],
    weight_decay=best_params['weight_decay']
)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(
    train_dataset,
    batch_size=best_params['batch_size'],
    shuffle=True
)

# Train for more epochs
for epoch in range(50):
    final_model.train()
    for X_batch, y_batch in train_loader:
        # Data already on device
        optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Final test evaluation
final_model.eval()
with torch.no_grad():
    test_outputs = final_model(X_test_t)
    _, predicted = torch.max(test_outputs.data, 1)
    test_acc = 100 * (predicted == y_test_t).sum().item() / len(y_test_t)

print(f"\nFINAL TEST ACCURACY: {test_acc:.2f}%")

# Visualizations
print("\nGenerating Optuna visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Optimization history
try:
    from optuna.visualization.matplotlib import plot_optimization_history as plot_opt_hist_mpl
    plot_opt_hist_mpl(study, ax=axes[0, 0])
    axes[0, 0].set_title('Optimization History', fontsize=14, fontweight='bold')
except:
    trial_numbers = [t.number for t in study.trials]
    trial_values = [t.value for t in study.trials if t.value is not None]
    axes[0, 0].plot(trial_numbers[:len(trial_values)], trial_values, 'o-', alpha=0.7)
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Validation Accuracy (%)')
    axes[0, 0].set_title('Optimization History')
    axes[0, 0].grid(True, alpha=0.3)

# 2. Parameter importances
try:
    from optuna.visualization.matplotlib import plot_param_importances as plot_param_imp_mpl
    plot_param_imp_mpl(study, ax=axes[0, 1])
    axes[0, 1].set_title('Parameter Importances', fontsize=14, fontweight='bold')
except:
    axes[0, 1].text(0.5, 0.5, 'Parameter importance\nrequires multiple trials', 
                     ha='center', va='center', fontsize=12)
    axes[0, 1].set_title('Parameter Importances')

# 3. Learning rate vs accuracy
lrs = [t.params['lr'] for t in study.trials if t.value is not None]
accs = [t.value for t in study.trials if t.value is not None]
axes[1, 0].scatter(lrs, accs, alpha=0.6, s=50)
axes[1, 0].set_xlabel('Learning Rate')
axes[1, 0].set_ylabel('Validation Accuracy (%)')
axes[1, 0].set_xscale('log')
axes[1, 0].set_title('Learning Rate vs Accuracy')
axes[1, 0].grid(True, alpha=0.3)

# 4. Hidden units vs accuracy
hidden = [t.params['hidden_units'] for t in study.trials if t.value is not None]
axes[1, 1].scatter(hidden, accs, alpha=0.6, s=50)
axes[1, 1].set_xlabel('Hidden Units')
axes[1, 1].set_ylabel('Validation Accuracy (%)')
axes[1, 1].set_title('Hidden Units vs Accuracy')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optuna_optimization_results.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'optuna_optimization_results.png'")
plt.show()

# Summary statistics
print("\n" + "="*80)
print("OPTUNA ADVANTAGES OVER GRID SEARCH:")
print("="*80)
print("✓ Intelligent sampling: Focuses on promising regions of hyperparameter space")
print("✓ Pruning: Stops bad trials early to save computation")
print("✓ Scales better: Can handle 10+ hyperparameters efficiently")
print("✓ Parameter importance: Tells you which hyperparameters actually matter")
print(f"✓ This search explored {len(study.trials)} configs vs {5*4==20} for grid search")
print("\nGrid search would need 4×5×4×4×5×4×3 = 19,200 trials for same space!")
print("Optuna found good solution in just 50 trials.")