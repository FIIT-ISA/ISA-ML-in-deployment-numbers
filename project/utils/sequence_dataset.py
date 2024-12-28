import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
import torch
from torch.utils.data import DataLoader, TensorDataset

scaler = MinMaxScaler(feature_range=(0, 1))


def create_test_dataset(look_back):
    n = 500
    x = np.arange(0, n, 1) 
    y = np.sin(16*np.pi*x/n) + np.cos(32*np.pi*x/n) + np.random.rand(n)
    data_org = y.reshape(-1, 1)
    result = STL(data_org, period=6, robust=True).fit()
    data_cleaned = result.trend.reshape(-1, 1)
    data_trans = scale_sequence(data_cleaned)
    train_size = int(len(data_trans) * 0.80)
    test = data_trans[train_size:len(data_trans), :]
    test_clean = data_cleaned[train_size:len(data_cleaned), :]
    return test, test_clean


def create_test_loader(test, look_back):
    test_sequences, test_targets = new_dataset(test, look_back)
    test_dataset = TensorDataset(test_sequences, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader



def new_dataset(data, look_back):
    sequences = []
    targets = []
    
    for i in range(len(data) - look_back):
        sequence = data[i:i + look_back]
        target = data[i + look_back]
        sequences.append(sequence)
        targets.append(target)

    sequences, targets = np.array(sequences), np.array(targets)
    return torch.tensor(sequences).float(), torch.tensor(targets).float()

def get_sequences(data, look_back):
    return [data[i:i + look_back].tolist() for i in range(len(data) - look_back)]


def scale_sequence(sequencce):
    return scaler.fit_transform(sequencce)


def inverse_scale(data):
    return scaler.inverse_transform(data)