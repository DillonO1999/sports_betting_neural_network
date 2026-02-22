#!/usr/bin/env python3

import pandas as pd
import os
import webbrowser
from datetime import datetime

# Import your model logic from the other file
try:
    import sports_betting_ml 
except ImportError:
    print("Error: Make sure your main code is saved as 'sports_betting_ml.py' in this folder.")

def generate_html_report(game_results):
    # Convert results to a DataFrame for easy HTML export
    df = pd.DataFrame(game_results)
    
    # Modern Dark Mode CSS
    style = """
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background-color: #121212; 
            color: #e0e0e0; 
            padding: 40px;
            line-height: 1.6;
        }
        .container { max-width: 1000px; margin: auto; }
        h1 { color: #bb86fc; text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 30px; 
            background-color: #1e1e1e; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        }
        th { background-color: #333; color: #03dac6; padding: 18px; text-align: left; text-transform: uppercase; letter-spacing: 1px; }
        td { padding: 15px; border-bottom: 1px solid #333; }
        tr:hover { background-color: #2c2c2c; }
        
        /* Status Colors */
        .value { color: #4caf50; font-weight: bold; background: rgba(76, 175, 80, 0.1); padding: 5px 10px; border-radius: 4px; }
        .noise { color: #ff9800; font-weight: bold; }
        .avoid { color: #f44336; opacity: 0.8; }
        
        .footer { margin-top: 30px; text-align: center; font-size: 0.8em; color: #777; }
    </style>
    """
    
    today_str = datetime.now().strftime("%B %d, %Y")
    
    html_content = f"""
    <html>
    <head>
        <title>Gemini NBA Edge Dashboard</title>
        {style}
    </head>
    <body>
        <div class="container">
            <h1>🏀 NBA Value Dashboard: {today_str}</h1>
            {df.to_html(index=False, classes='dataframe', escape=False)}
            <div class="footer">
                <p>Strategy: 0.25 Fractional Kelly | Data Source: nba_api | ML: Ensemble Head-to-Head Net</p>
                <p><i>Disclaimer: Predictions are based on historical rolling averages and do not guarantee future results.</i></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save and launch
    file_path = os.path.abspath("nba_report.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Report generated: {file_path}")
    webbrowser.open(f"file://{file_path}")

if __name__ == "__main__":
    print("🚀 Initializing NBA Model Engine...")
    
    # Run the model logic from your engine file
    # This assumes run_betting_tool() returns the results list
    try:
        results = sports_betting_ml.run_betting_tool() 
        
        if results:
            generate_html_report(results)
        else:
            print("No data returned from the engine.")
    except Exception as e:
        print(f"An error occurred: {e}")