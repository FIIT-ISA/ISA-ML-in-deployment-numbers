import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Matplotlib settings
plt.rcParams["figure.figsize"] = (12, 2)
plt.style.use('ggplot')

def decide_device():
  if (torch.cuda.is_available()): return "cuda"
  #if (torch.backends.mps.is_available()): return "mps"
  return "cpu"

# Data creation and preprocessing
n = 500
x = np.arange(0, n, 1) 
y = np.sin(16*np.pi*x/n) + np.cos(32*np.pi*x/n) + np.random.rand(n)
data_org = y.reshape(-1, 1)

print(data_org.shape)
plt.plot(data_org)

# Stationarity test
dftest = adfuller(data_org, autolag='AIC')
print(f"\t1. ADF: {dftest[0]}")
print(f"\t2. P-Value: {dftest[1]}")
print(f"\t3. Num Of Lags: {dftest[2]}")

# Seasonal decomposition
result = STL(data_org, period=6, robust=True).fit()
result.plot()
plt.show()

data_cleaned = result.trend.reshape(-1, 1)

# Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
data_trans = scaler.fit_transform(data_cleaned)

# Data splitting
train_size = int(len(data_trans) * 0.80)
test_size = len(data_trans) - train_size
train, test = data_trans[0:train_size, :], data_trans[train_size:len(data_trans), :]

# PyTorch Dataset
look_back = 10

def create_dataset(data, look_back):
    sequences = []
    targets = []

    for i in range(len(data) - look_back):
        sequence = data[i:i + look_back]
        target = data[i + look_back]
        sequences.append(sequence)
        targets.append(target)

    sequences, targets = np.array(sequences), np.array(targets)
    return torch.tensor(sequences).float(), torch.tensor(targets).float()

train_sequences, train_targets = create_dataset(train, look_back)
test_sequences, test_targets = create_dataset(test, look_back)

train_dataset = TensorDataset(train_sequences, train_targets)
test_dataset = TensorDataset(test_sequences, test_targets)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# PyTorch model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.linear(h_n.squeeze(0))
        return x

model = LSTMModel(1, 5, 1)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
device = torch.device(decide_device())
best_model_state = None
# Training loop
def train(model, train_loader, criterion, optimizer):
    global best_model_state
    model.train()
    curr_loss = 200
    for epoch in range(20):
        for sequences, targets in train_loader:
            optimizer.zero_grad()
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if loss.item() < curr_loss:
            print("Found best model at epoch: ",epoch)
            curr_loss = loss.item()
            best_model_state = model.state_dict().copy()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

train(model, train_loader, criterion, optimizer)

# Evaluation function
def evaluate(model, loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for sequences, targets in loader:
            outputs = model(sequences)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
    return np.array(predictions), np.array(actuals)

trainPredict, trainY = evaluate(model, train_loader)
testPredict, testY = evaluate(model, test_loader)

# Rescaling predictions
trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
testY = scaler.inverse_transform(testY.reshape(-1, 1))

# Calculate RMSE
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
torch.save(best_model_state, 'lstm_model.pt')
