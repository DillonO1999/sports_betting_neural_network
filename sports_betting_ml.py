#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

# 1. NEW: Defining a simple but powerful Neural Network
class NBAPredictor(nn.Module):
    def __init__(self, input_dim):
        super(NBAPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents overfitting (very common in sports ML)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Outputs a probability between 0 and 1
        )
        
    def forward(self, x):
        return self.network(x)

def fetch_and_preprocess():
    # Fetching data using your existing logic
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2024-25') # Updated season
    df = gamefinder.get_data_frames()[0]
    
    # Preprocessing
    df['target'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    numeric_df = df.select_dtypes(include=['float64', 'int64']).dropna()
    
    X = numeric_df.drop(columns=['target']).values
    y = numeric_df['target'].values
    
    # Always scale for Neural Networks (essential for convergence)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch Tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)
    
    return X_train, X_test, y_train, y_test

def train_nn_model(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    model = NBAPredictor(input_dim)
    
    # Loss & Optimizer
    criterion = nn.BCELoss() # Binary Cross Entropy for Win/Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        rounded_preds = predictions.round()
        accuracy = (rounded_preds.eq(y_test).sum() / float(y_test.shape[0]))
        print(f"\nFinal Test Accuracy: {accuracy:.2%}")

    return model

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = fetch_and_preprocess()
    nba_model = train_nn_model(X_tr, y_tr, X_te, y_te)
