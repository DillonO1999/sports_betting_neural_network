#!/usr/bin/env python3

from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Lightweight replacement
from sklearn.model_selection import train_test_split

def fetch_nba_data():
    # Grabs all games from the current season
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2023-24')
    games = gamefinder.get_data_frames()[0]
    return games

def preprocess_data(df, target_column='WL'):
    # 1. Convert Target (W/L) to binary
    if 'WL' in df.columns:
        df['target'] = df[target_column].apply(lambda x: 1 if x == 'W' else 0)
    
    # 2. Select numerical features only
    # Note: We drop 'target' from the features list before scaling
    numeric_df = df.select_dtypes(include=['float64', 'int64']).dropna()
    
    y = numeric_df['target']
    X = numeric_df.drop(columns=['target'])
    
    # 3. Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def build_and_train_model(X, y):
    # Random Forest is great for "small" data and low storage
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2%}")
    
    return model