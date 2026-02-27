#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import statistics
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2
from nba_api.stats.static import teams
from sklearn.preprocessing import StandardScaler

# --- STABILITY ANCHOR ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- LIVE UPDATES ---
CRITICAL_INJURIES = {
    "76ers": ["Embiid", "George"], 
    "Kings": ["Sabonis", "Murray"],
    "Suns": ["Booker"],
    "Magic": ["F. Wagner"]
}

VEGAS_ODDS = {
    "Suns": -136, "Magic": +116,
    "76ers": -168, "Pelicans": +142,
    "Bulls": +420, "Pistons": -559,
    "Knicks": -170, "Rockets": +142,
    "Heat": -140, "Grizzlies": +118,
    "Spurs": -1200, "Kings": +750
}

def calculate_kelly_units(win_prob, ml, fraction=0.25):
    if ml == 0: return 0.0
    # b is net odds
    b = (ml / 100) if ml > 0 else (100 / abs(ml))
    q = 1 - win_prob 
    kelly_f = (b * win_prob - q) / b
    units = kelly_f * fraction * 100 
    return max(0, units)

class NBAHeadToHeadNet(nn.Module):
    def __init__(self, input_dim):
        super(NBAHeadToHeadNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.4),          
            nn.Linear(32, 1)          
        )
    def forward(self, x):
        return self.net(x)

def get_h2h_data():
    print("Loading historical matchups...")
    finder = leaguegamefinder.LeagueGameFinder(season_nullable=['2021-22', '2022-23', '2023-24', '2024-25', '2025-26'])
    df = finder.get_data_frames()[0]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    
    stats_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    rolling = df.groupby('TEAM_ID')[stats_cols].rolling(window=7, closed='left').mean().reset_index(0, drop=True)
    rolling.columns = [f'ROLL_{c}' for c in rolling.columns]
    df = pd.concat([df, rolling], axis=1).dropna(subset=['ROLL_PTS'])
    
    h2h_data = []
    for game_id, group in df.groupby('GAME_ID'):
        if len(group) == 2:
            try:
                home = group[group['MATCHUP'].str.contains('vs.')]
                away = group[group['MATCHUP'].str.contains('@')]
                if not home.empty and not away.empty:
                    features = np.concatenate([home.iloc[0][rolling.columns].values, away.iloc[0][rolling.columns].values])
                    h2h_data.append({'features': features, 'target': 1 if home.iloc[0]['WL'] == 'W' else 0})
            except: continue
    return np.array([d['features'] for d in h2h_data]), np.array([d['target'] for d in h2h_data]), df, rolling.columns

def run_betting_tool():
    set_seed(42) 
    X, y, full_df, roll_names = get_h2h_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Setup Team Lookup
    team_lookup = {team['id']: team['nickname'] for team in teams.get_teams()}
    
    NUM_MODELS = 7 
    ensemble_results = {}
    fold_size = len(X_scaled) // NUM_MODELS

    # 2. Get LIVE games for TODAY
    sb = scoreboardv2.ScoreboardV2() 
    tonight = sb.get_data_frames()[0]

    if tonight.empty:
        print("No games found for today.")
        return []

    print(f"Training a Jury of {NUM_MODELS} models...")
    for i in range(NUM_MODELS):
        start, end = i * fold_size, (i + 1) * fold_size
        X_train = np.delete(X_scaled, slice(start, end), axis=0)
        y_train = np.delete(y, slice(start, end), axis=0)
        
        model = NBAHeadToHeadNet(X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(150): # Reduced epochs slightly for speed
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(torch.FloatTensor(X_train)), torch.FloatTensor(y_train).view(-1, 1))
            loss.backward()
            optimizer.step()
        
        for _, game in tonight.iterrows():
            g_id = game['GAME_ID']
            h_feat = full_df[full_df['TEAM_ID'] == game['HOME_TEAM_ID']].tail(1)
            a_feat = full_df[full_df['TEAM_ID'] == game['VISITOR_TEAM_ID']].tail(1)
            if not h_feat.empty and not a_feat.empty:
                pred_input = np.concatenate([h_feat[roll_names].values[0], a_feat[roll_names].values[0]])
                pred_scaled = scaler.transform(pred_input.reshape(1, -1))
                with torch.no_grad():
                    p = torch.sigmoid(model(torch.FloatTensor(pred_scaled))).item()
                if g_id not in ensemble_results: ensemble_results[g_id] = []
                ensemble_results[g_id].append(p)

    report_data = []

    for g_id, probs in ensemble_results.items():
        # --- THE FIX FOR DIVISION BY ZERO ---
        if len(probs) > 1:
            avg_home_p = statistics.mean(probs)
            stdev = statistics.stdev(probs)
        elif len(probs) == 1:
            avg_home_p = probs[0]
            stdev = 0.0
        else: continue

        game_row = tonight[tonight['GAME_ID'] == g_id].iloc[0]
        h_name = team_lookup.get(game_row['HOME_TEAM_ID'], "Home")
        a_name = team_lookup.get(game_row['VISITOR_TEAM_ID'], "Away")

        # Injury Logic
        for team_key in CRITICAL_INJURIES:
            if team_key in h_name: avg_home_p -= 0.12  
            if team_key in a_name: avg_home_p += 0.12  
        
        avg_home_p = max(0.01, min(0.99, avg_home_p))
        winner = h_name if avg_home_p > 0.5 else a_name
        winner_p = avg_home_p if avg_home_p > 0.5 else (1 - avg_home_p)

        # Betting Odds Logic
        ml = 0
        for team_key, odds_value in VEGAS_ODDS.items():
            if team_key in winner:
                ml = odds_value
                break
        
        implied = (abs(ml)/(abs(ml)+100)) if ml < 0 else (100/(ml+100)) if ml != 0 else 0.50
        edge = winner_p - implied

        # Verdict Formatting
        if stdev > 0.10: status = "<span class='noise'>⚠️ HIGH NOISE</span>"
        elif edge > 0.08: status = "<span class='value'>💰 VALUE FOUND</span>"
        elif edge < -0.02: status = "<span class='avoid'>❌ OVERPRICED</span>"
        else: status = "⚖️ FAIR PRICE"

        report_data.append({
            "Matchup": f"{a_name} @ {h_name}",
            "ML Choice": f"<b>{winner}</b> ({winner_p:.1%})",
            "Vegas Implied": f"{implied:.1%}",
            "Edge": f"{edge:.1%}" if stdev <= 0.10 else "N/A",
            "Kelly Units": f"{calculate_kelly_units(winner_p, ml):.2f}",
            "Consistency Error": f"{stdev:.1%}",
            "Verdict": status
        })

    return report_data

if __name__ == "__main__":
    results = run_betting_tool()
    for res in results:
        print(f"{res['Matchup']} | Prediction: {res['ML Choice']} | Edge: {res['Edge']}")