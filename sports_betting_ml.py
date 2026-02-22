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
from sklearn.preprocessing import StandardScaler

# --- STABILITY ANCHOR ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- LIVE UPDATES (Feb 21, 2026) ---
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

# --- UNIT CALCULATION (KELLY CRITERION) ---
def calculate_kelly_units(win_prob, ml, fraction=0.25):
    """
    Calculates bet size in units using the Kelly Criterion.
    win_prob: AI predicted probability (0.0 to 1.0)
    ml: American Moneyline (+142 or -136)
    fraction: 'Fractional Kelly' multiplier to reduce volatility (0.25 is standard)
    """
    # Convert American Odds to Decimal Odds (b)
    # b is the "net odds" (profit per $1 wagered)
    if ml > 0:
        b = ml / 100
    else:
        b = 100 / abs(ml)
    
    q = 1 - win_prob # Probability of losing
    
    # Kelly Formula: f* = (bp - q) / b
    kelly_f = (b * win_prob - q) / b
    
    # Apply fractional Kelly for safety and convert to "Units"
    # Assuming 1 Unit = 1% of bankroll for this calculation
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
    print("Loading historical matchups (Multi-Season Deep Dive)...")
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
    
    NUM_MODELS = 7 
    ensemble_results = {}
    fold_size = len(X_scaled) // NUM_MODELS

    print(f"Training a Jury of {NUM_MODELS} models (Stability Tuning Active)...")
    for i in range(NUM_MODELS):
        start, end = i * fold_size, (i + 1) * fold_size
        X_train = np.delete(X_scaled, slice(start, end), axis=0)
        y_train = np.delete(y, slice(start, end), axis=0)
        
        model = NBAHeadToHeadNet(X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(torch.FloatTensor(X_train)), torch.FloatTensor(y_train).view(-1, 1))
            loss.backward()
            optimizer.step()
        
        sb = scoreboardv2.ScoreboardV2(game_date='2026-02-21')
        tonight = sb.get_data_frames()[0]
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
        print(f"   Brain {i+1} locked.")

    print("\n--- FINAL VALUE ANALYSIS (Feb 21, 2026) ---")
    bets_to_consider = []

    report_data = [] # We will fill this for the dashboard

    for g_id, probs in ensemble_results.items():
        avg_home_p = statistics.mean(probs)
        stdev = statistics.stdev(probs)
        game_row = tonight[tonight['GAME_ID'] == g_id].iloc[0]
        h_name = full_df[full_df['TEAM_ID'] == game_row['HOME_TEAM_ID']]['TEAM_NAME'].iloc[0]
        a_name = full_df[full_df['TEAM_ID'] == game_row['VISITOR_TEAM_ID']]['TEAM_NAME'].iloc[0]

        for team_key in CRITICAL_INJURIES:
            if team_key in h_name: avg_home_p -= 0.12  
            if team_key in a_name: avg_home_p += 0.12  
        
        avg_home_p = max(0.01, min(0.99, avg_home_p))
        winner = h_name if avg_home_p > 0.5 else a_name
        winner_p = avg_home_p if avg_home_p > 0.5 else (1 - avg_home_p)

        ml = 0
        for team_key, odds_value in VEGAS_ODDS.items():
            if team_key in winner:
                ml = odds_value
                break
        
        if ml == 0:
            implied = 0.50 
        else:
            implied = (abs(ml)/(abs(ml)+100)) if ml < 0 else (100/(ml+100))

        edge = winner_p - implied

        # Formatting for HTML
        if stdev > 0.08: 
            status_html = "<span class='noise'>⚠️ HIGH NOISE</span>"
            display_edge = "N/A"
            units = 0.00
        elif edge > 0.08: 
            status_html = "<span class='value'>💰 VALUE FOUND</span>"
            display_edge = f"{edge:.1%}"
            units = calculate_kelly_units(winner_p, ml)
        elif edge < -0.02: 
            status_html = "<span class='avoid'>❌ OVERPRICED</span>"
            display_edge = f"{edge:.1%}"
            units = 0.00
        else: 
            status_html = "⚖️ FAIR PRICE"
            display_edge = f"{edge:.1%}"
            units = 0.00

        # Append data for the Dashboard
        report_data.append({
            "Matchup": f"{a_name} @ {h_name}",
            "ML Choice": f"<b>{winner}</b> ({winner_p:.1%})",
            "Vegas Implied": f"{implied:.1%}",
            "Edge": display_edge,
            "Kelly Units": f"{units:.2f}",
            "Consistency Error": f"{stdev:.1%}",
            "Verdict": status_html
        })

    return report_data # IMPORTANT: Returning the data to dashboard.py

if __name__ == "__main__":
    run_betting_tool()