import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

device = 'cpu'
data = fetch_california_housing()
X,y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42
                                                    )


scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train).to(device)
X_test_t  = torch.FloatTensor(X_test).to(device)
y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
y_test_t  = torch.FloatTensor(y_test).view(-1, 1).to(device)


# Create Loaders
train_dataset = TensorDataset(X_train_t, y_train_t)

train_loader = DataLoader(train_dataset, 
                          shuffle=True,
                          batch_size=256)

# Architecture
class HousingNET(nn.Module):
    def __init__(self, input_dim):
        super(HousingNET, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = HousingNET(X_train.shape[1]).to(device=device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS       = 50
train_losses = []
test_losses  = []

print("Training California Housing Price Prediction Model...")
print(f"{'Epoch':<10}{'Train Loss':<15}{'Test Loss':<15}{'RMSE':<15}{'R²':<10}")
print("-" * 65)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()

        # Proper accumulation
        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_train_loss = total_loss / total_samples
    train_losses.append(avg_train_loss)

    # ---- Evaluation ----
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_loss = loss_fn(test_preds, y_test_t).item()
        test_losses.append(test_loss)

        rmse = np.sqrt(test_loss)
        ss_res = torch.sum((y_test_t - test_preds) ** 2).item()
        ss_tot = torch.sum((y_test_t - torch.mean(y_test_t)) ** 2).item()
        r2_score = 1 - (ss_res / ss_tot)

    if epoch % 1 == 0 or epoch == 1:
        print(f"{epoch:<10}{avg_train_loss:<15.4f}{test_loss:<15.4f}{rmse:<15.4f}{r2_score:<10.4f}")

        
# Final Evaluation
model.eval()
with torch.no_grad():
    final_preds = model(X_test_t)
    final_rmse = np.sqrt(loss_fn(final_preds, y_test_t)).item()
    ss_res = torch.sum((y_test_t - final_preds) ** 2).item()
    ss_tot = torch.sum((y_test_t - torch.mean(y_test_t)) ** 2).item()
    final_r2 = 1 - (ss_res / ss_tot)
    
    print("\n" + "="*65)
    print("FINAL METRICS:")
    print(f"  RMSE: {final_rmse:.4f}")
    print(f"  R² Score: {final_r2:.4f}")
    print(f"  Mean Absolute Error: {torch.mean(torch.abs(y_test_t - final_preds)).item():.4f}")
    print("="*65)


fig, axes = plt.subplots(1, 2, figsize= (14, 5))
axes[0].plot(train_losses, label = 'Train loss', alpha=0.8)
axes[0].plot(train_losses, label='Train Loss', alpha=0.8)
axes[0].plot(test_losses, label='Test Loss', alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training and Test Loss Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Predictions vs Actual
axes[1].scatter(y_test_t.numpy(), final_preds.numpy(), alpha=0.5)
axes[1].plot([y_test_t.min(), y_test_t.max()], 
             [y_test_t.min(), y_test_t.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')
axes[1].set_title(f'Predictions vs Actual (R²={final_r2:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('housing_price_results.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'housing_price_results.png'")
plt.show()

# Save the model
torch.save(model.state_dict(), 'housing_price_model.pth')
print("Model saved as 'housing_price_model.pth'")