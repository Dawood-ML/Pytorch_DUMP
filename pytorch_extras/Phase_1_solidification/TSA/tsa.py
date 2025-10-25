import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Download stock data (Apple as example)
print("Downloading stock data...")
ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # 5 years of data

stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
prices = stock_data['Close'].values.reshape(-1, 1)

print(f"Downloaded {len(prices)} days of {ticker} stock data")

# Create sequences for time series prediction
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Normalize data (CRITICAL for stock prices with trends)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Create sequences (use past 60 days to predict next day)
sequence_length = 60
X, y = create_sequences(prices_scaled, sequence_length)

# Train/test split (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the Time Series ANN
class StockPriceNet(nn.Module):
    def __init__(self, seq_length):
        super(StockPriceNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(seq_length, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Flatten the sequence
        x = x.view(x.size(0), -1)
        return self.network(x)

# Initialize model
model = StockPriceNet(seq_length=sequence_length)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 150
train_losses = []
test_losses = []

print(f"\nTraining {ticker} Stock Price Prediction Model...")
print(f"{'Epoch':<10}{'Train Loss':<15}{'Test Loss':<15}{'RMSE':<15}{'R²':<10}")
print("-" * 65)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_t)
        test_loss = criterion(test_predictions, y_test_t).item()
        test_losses.append(test_loss)
        
        # RMSE (in normalized space)
        rmse = np.sqrt(test_loss)
        
        # R² score
        ss_res = torch.sum((y_test_t - test_predictions) ** 2).item()
        ss_tot = torch.sum((y_test_t - torch.mean(y_test_t)) ** 2).item()
        r2_score = 1 - (ss_res / ss_tot)
    
    if (epoch + 1) % 15 == 0 or epoch == 0:
        print(f"{epoch+1:<10}{avg_train_loss:<15.6f}{test_loss:<15.6f}{rmse:<15.6f}{r2_score:<10.4f}")

# Final evaluation with denormalized prices
model.eval()
with torch.no_grad():
    final_predictions = model(X_test_t)
    
    # Denormalize predictions and actual values
    predictions_actual = scaler.inverse_transform(final_predictions.numpy())
    y_test_actual = scaler.inverse_transform(y_test_t.numpy())
    
    # Calculate metrics in actual price space
    mse = np.mean((predictions_actual - y_test_actual) ** 2)
    rmse_actual = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_actual - y_test_actual))
    
    ss_res = np.sum((y_test_actual - predictions_actual) ** 2)
    ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
    r2_actual = 1 - (ss_res / ss_tot)
    
    # Calculate percentage error
    mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100
    
    print("\n" + "="*65)
    print("FINAL METRICS (Actual Price Scale):")
    print(f"  RMSE: ${rmse_actual:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R² Score: {r2_actual:.4f}")
    print("="*65)
    print("\nBRUTAL TRUTH: This model is NOT ready for real trading.")
    print(f"MAPE of {mape:.1f}% means you're off by ~${mae:.2f} per prediction.")
    print("Markets are complex; this is a learning exercise, not a money printer.")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Loss curves
axes[0, 0].plot(train_losses, label='Train Loss', alpha=0.8)
axes[0, 0].plot(test_losses, label='Test Loss', alpha=0.8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss (Normalized)')
axes[0, 0].set_title('Training and Test Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Predictions vs Actual (scatter)
axes[0, 1].scatter(y_test_actual, predictions_actual, alpha=0.5)
axes[0, 1].plot([y_test_actual.min(), y_test_actual.max()], 
                [y_test_actual.min(), y_test_actual.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Price ($)')
axes[0, 1].set_ylabel('Predicted Price ($)')
axes[0, 1].set_title(f'Predictions vs Actual (R²={r2_actual:.3f})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Time series plot (last 200 test points)
plot_points = min(200, len(y_test_actual))
test_dates = range(plot_points)
axes[1, 0].plot(test_dates, y_test_actual[:plot_points], 
                label='Actual Price', linewidth=2, alpha=0.7)
axes[1, 0].plot(test_dates, predictions_actual[:plot_points], 
                label='Predicted Price', linewidth=2, alpha=0.7)
axes[1, 0].set_xlabel('Test Sample')
axes[1, 0].set_ylabel('Price ($)')
axes[1, 0].set_title(f'{ticker} Stock Price: Actual vs Predicted (Last {plot_points} Days)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Prediction errors distribution
errors = predictions_actual - y_test_actual
axes[1, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
axes[1, 1].set_xlabel('Prediction Error ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Prediction Error Distribution (MAE=${mae:.2f})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stock_forecast_results.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'stock_forecast_results.png'")
plt.show()

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'sequence_length': sequence_length
}, 'stock_price_model.pth')
print("Model and scaler saved as 'stock_price_model.pth'")