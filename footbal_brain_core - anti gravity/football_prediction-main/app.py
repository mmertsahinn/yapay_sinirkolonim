from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from functools import lru_cache
from datetime import datetime

app = Flask(__name__)

# Load the trained ensemble model, label encoder, and goal models
ensemble_model = joblib.load('football_prediction_ensemble.joblib')
le = joblib.load('label_encoder.joblib')
home_goals_model = joblib.load('home_goals_model.joblib')
away_goals_model = joblib.load('away_goals_model.joblib')

# Load team strength data and other necessary data
df = pd.read_csv('football_match_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Calculate team strengths and defenses
@lru_cache(maxsize=None)
def get_team_stats():
    window = 10
    home_strength = df.groupby('home_team')['home_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean()).to_dict()
    away_strength = df.groupby('away_team')['away_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean()).to_dict()
    home_defense = df.groupby('home_team')['away_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean()).to_dict()
    away_defense = df.groupby('away_team')['home_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean()).to_dict()
    return home_strength, away_strength, home_defense, away_defense

# Calculate head-to-head statistics
@lru_cache(maxsize=None)
def get_h2h_stats():
    def calculate_h2h(group):
        home_wins = (group['home_goals'] > group['away_goals']).sum()
        away_wins = (group['home_goals'] < group['away_goals']).sum()
        draws = (group['home_goals'] == group['away_goals']).sum()
        total_matches = len(group)
        return {
            'h2h_home_win_rate': home_wins / total_matches if total_matches > 0 else 0.5,
            'h2h_away_win_rate': away_wins / total_matches if total_matches > 0 else 0.5,
            'h2h_draw_rate': draws / total_matches if total_matches > 0 else 0.5
        }
    return df.groupby(['home_team', 'away_team']).apply(calculate_h2h).to_dict()

def get_h2h_history(home_team, away_team, n=5):
    matches = df[(df['home_team'] == home_team) & (df['away_team'] == away_team) |
                 (df['home_team'] == away_team) & (df['away_team'] == home_team)]
    
    matches = matches.sort_values('date', ascending=False).drop_duplicates(subset=['date', 'home_team', 'away_team'])
    matches = matches.head(n)
    
    history = []
    for _, match in matches.iterrows():
        date_obj = match['date']
        formatted_date = date_obj.strftime('%d %b %Y')
        
        if match['home_team'] == home_team:
            result = 'home_win' if match['home_goals'] > match['away_goals'] else 'away_win' if match['home_goals'] < match['away_goals'] else 'draw'
            history.append({
                'date': formatted_date,
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'score': f"{match['home_goals']}-{match['away_goals']}",
                'result': result
            })
        else:
            result = 'away_win' if match['home_goals'] > match['away_goals'] else 'home_win' if match['home_goals'] < match['away_goals'] else 'draw'
            history.append({
                'date': formatted_date,
                'home_team': match['away_team'],
                'away_team': match['home_team'],
                'score': f"{match['away_goals']}-{match['home_goals']}",
                'result': result
            })
    return history

def get_team_last_matches(team, n=5):
    current_date = datetime.now()
    home_matches = df[(df['home_team'] == team) & (df['date'] < current_date)].sort_values('date', ascending=False).head(n)
    away_matches = df[(df['away_team'] == team) & (df['date'] < current_date)].sort_values('date', ascending=False).head(n)
    
    all_matches = pd.concat([home_matches, away_matches]).sort_values('date', ascending=False).head(n)
    
    history = []
    for _, match in all_matches.iterrows():
        date_obj = match['date']
        formatted_date = date_obj.strftime('%d %b %Y')
        
        if match['home_team'] == team:
            result = 'win' if match['home_goals'] > match['away_goals'] else 'loss' if match['home_goals'] < match['away_goals'] else 'draw'
            history.append({
                'date': formatted_date,
                'opponent': match['away_team'],
                'score': f"{match['home_goals']}-{match['away_goals']}",
                'result': result
            })
        else:
            result = 'win' if match['away_goals'] > match['home_goals'] else 'loss' if match['away_goals'] < match['home_goals'] else 'draw'
            history.append({
                'date': formatted_date,
                'opponent': match['home_team'],
                'score': f"{match['away_goals']}-{match['home_goals']}",
                'result': result
            })
    return history

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    
    home_strength, away_strength, home_defense, away_defense = get_team_stats()
    h2h_stats = get_h2h_stats()
    
    try:
        # Calculate features
        home_team_strength = home_strength.get(home_team, np.mean(list(home_strength.values())))
        away_team_strength = away_strength.get(away_team, np.mean(list(away_strength.values())))
        home_team_defense = home_defense.get(home_team, np.mean(list(home_defense.values())))
        away_team_defense = away_defense.get(away_team, np.mean(list(away_defense.values())))
        
        avg_home_xG = df[df['home_team'] == home_team]['home_xG'].mean()
        avg_away_xG = df[df['away_team'] == away_team]['away_xG'].mean()
        
        xG_difference = avg_home_xG - avg_away_xG
        
        h2h = h2h_stats.get((home_team, away_team), {
            'h2h_home_win_rate': 0.5,
            'h2h_away_win_rate': 0.5,
            'h2h_draw_rate': 0.5
        })
        
        home_matches = df[df['home_team'] == home_team].tail(5)
        away_matches = df[df['away_team'] == away_team].tail(5)
        
        home_form = (home_matches['home_goals'] - home_matches['away_goals']).mean()
        away_form = (away_matches['away_goals'] - away_matches['home_goals']).mean()
        
        # New features
        goal_ratio = home_team_strength / (away_team_strength + 1e-5)
        xG_ratio = avg_home_xG / (avg_away_xG + 1e-5)
        current_date = datetime.now()
        day_of_week = current_date.weekday()
        month = current_date.month
        
        # Social media and hype features (from football_brain data)
        # Get latest available hype data for these teams, or use defaults
        home_matches_with_hype = df[(df['home_team'] == home_team) & (df['home_support'].notna())].sort_values('date', ascending=False)
        away_matches_with_hype = df[(df['away_team'] == away_team) & (df['away_support'].notna())].sort_values('date', ascending=False)
        
        # Use most recent hype data if available, otherwise defaults
        home_support = home_matches_with_hype['home_support'].iloc[0] if len(home_matches_with_hype) > 0 else 0.5
        away_support = away_matches_with_hype['away_support'].iloc[0] if len(away_matches_with_hype) > 0 else 0.5
        
        # Get sentiment and tweet data from most recent match between these teams
        h2h_matches_with_hype = df[((df['home_team'] == home_team) & (df['away_team'] == away_team) |
                                     (df['home_team'] == away_team) & (df['away_team'] == home_team)) &
                                    (df['sentiment_score'].notna())].sort_values('date', ascending=False)
        
        sentiment_score = h2h_matches_with_hype['sentiment_score'].iloc[0] if len(h2h_matches_with_hype) > 0 else 0.0
        total_tweets = h2h_matches_with_hype['total_tweets'].iloc[0] if len(h2h_matches_with_hype) > 0 else 0
        
        # Calculate derived features
        support_difference = home_support - away_support
        support_ratio = home_support / (away_support + 1e-5)
        sentiment_positive = 1 if sentiment_score > 0.2 else 0
        sentiment_negative = 1 if sentiment_score < -0.2 else 0
        log_total_tweets = np.log1p(total_tweets)
        # Calculate high_hype threshold from available data
        if 'total_tweets' in df.columns and df['total_tweets'].notna().sum() > 0:
            tweets_75th = df['total_tweets'].quantile(0.75)
            high_hype = 1 if total_tweets > tweets_75th else 0
        else:
            high_hype = 0
        hype_score = ((home_support + away_support) * 0.4 + 
                     (sentiment_score + 1) * 0.3 + 
                     np.clip(log_total_tweets / 10, 0, 1) * 0.3)
        
        # Varsayılan odds değerleri (veriler yoksa)
        implied_prob_home = 0.33
        implied_prob_draw = 0.33
        implied_prob_away = 0.34
        
        input_data = pd.DataFrame({
            'home_team_strength': [home_team_strength],
            'away_team_strength': [away_team_strength],
            'home_team_defense': [home_team_defense],
            'away_team_defense': [away_team_defense],
            'home_xG': [avg_home_xG],
            'away_xG': [avg_away_xG],
            'xG_difference': [xG_difference],
            'home_form': [home_form],
            'away_form': [away_form],
            'h2h_home_win_rate': [h2h['h2h_home_win_rate']],
            'h2h_away_win_rate': [h2h['h2h_away_win_rate']],
            'h2h_draw_rate': [h2h['h2h_draw_rate']],
            'goal_ratio': [goal_ratio],
            'xG_ratio': [xG_ratio],
            'day_of_week': [day_of_week],
            'month': [month],
            # Social media and hype features
            'home_support': [home_support],
            'away_support': [away_support],
            'support_difference': [support_difference],
            'support_ratio': [support_ratio],
            'sentiment_score': [sentiment_score],
            'sentiment_positive': [sentiment_positive],
            'sentiment_negative': [sentiment_negative],
            'total_tweets': [total_tweets],
            'log_total_tweets': [log_total_tweets],
            'high_hype': [high_hype],
            'hype_score': [hype_score],
            # Odds özellikleri (varsayılan değerler)
            'implied_prob_home': [implied_prob_home],
            'implied_prob_draw': [implied_prob_draw],
            'implied_prob_away': [implied_prob_away],
            'hype_favored_team': [0],
            'odds_favored_team': [0],
            'hype_odds_alignment': [0],
            'home_hype_inflation': [0],
            'away_hype_inflation': [0],
            'hype_inflation_score': [0],
            'high_engagement_match': [0],
            'hype_odds_discrepancy': [0],
            'tweets_odds_ratio': [0],
            'odds_market_efficiency': [0],
            'odds_entropy': [1.0],
            'hype_odds_consistency_home': [0],
            'hype_odds_consistency_away': [0],
            'hype_odds_consistency_draw': [0],
            'hype_odds_diff_home': [0],
            'hype_odds_diff_away': [0],
            'hype_odds_diff_draw': [0],
            'home_hype_inflation_score': [0],
            'away_hype_inflation_score': [0],
            'total_hype_inflation': [0],
            'tweets_odds_variance': [0],
            'tweets_per_odds_prob': [0],
            'implied_prob_over_25': [0.5],
            'implied_prob_under_25': [0.5],
            'tweets_over_odds': [0],
            'ah_home_implied_prob': [0.5],
            'ah_away_implied_prob': [0.5],
            'hype_ah_consistency_home': [0],
            'hype_ah_consistency_away': [0],
            'market_consensus': [0.5]
        })
        
        prediction_encoded = ensemble_model.predict(input_data)
        prediction = le.inverse_transform(prediction_encoded)[0]
        
        probabilities = ensemble_model.predict_proba(input_data)[0]
        probability_dict = {le.classes_[i]: float(prob) * 100 for i, prob in enumerate(probabilities)}

        # ============================================================
        # SKOR TAHMİNİ - 2 YÖNTEM
        # ============================================================
        
        # YÖNTEM 1: ORİJİNAL (Gol Modelleri - Bağımsız)
        original_home = int(round(home_goals_model.predict(input_data)[0]))
        original_away = int(round(away_goals_model.predict(input_data)[0]))
        original_home = max(0, min(5, original_home))
        original_away = max(0, min(5, original_away))
        original_score = f"{original_home}-{original_away}"
        
        # YÖNTEM 2: YENİ (Ensemble Olasılıklarına Göre - Tutarlı)
        home_win_prob = probability_dict.get('home_win', 0) / 100
        draw_prob = probability_dict.get('draw', 0) / 100
        away_win_prob = probability_dict.get('away_win', 0) / 100
        
        base_home = home_team_strength
        base_away = away_team_strength
        
        if home_win_prob > 0.6:  # Açık ev galibiyeti
            new_home = round(base_home + 0.5)
            new_away = round(base_away - 0.5)
        elif away_win_prob > 0.6:  # Açık deplasman galibiyeti
            new_home = round(base_home - 0.5)
            new_away = round(base_away + 0.5)
        elif draw_prob > 0.4:  # Beraberlik muhtemel
            avg_goals = (base_home + base_away) / 2
            new_home = round(avg_goals)
            new_away = round(avg_goals)
        else:  # Dengeli maç
            new_home = round(base_home)
            new_away = round(base_away)
        
        new_home = max(0, min(5, int(new_home)))
        new_away = max(0, min(5, int(new_away)))
        new_score = f"{new_home}-{new_away}"

        h2h_history = get_h2h_history(home_team, away_team)
        home_team_last_matches = get_team_last_matches(home_team)
        away_team_last_matches = get_team_last_matches(away_team)
        
        return jsonify({
            'prediction': prediction,
            'probabilities': probability_dict,
            'predicted_score_original': original_score,  # Orijinal gol modelleri
            'predicted_score_new': new_score,            # Yeni ensemble tabanlı
            'h2h_history': h2h_history,
            'home_team_last_matches': home_team_last_matches,
            'away_team_last_matches': away_team_last_matches
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/teams', methods=['GET'])
def get_teams():
    # Tüm takımları al ve normalize et
    all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    # Boşlukları temizle ve unique yap
    teams = sorted(set([str(t).strip() for t in all_teams if pd.notna(t)]))
    return jsonify(teams)

if __name__ == '__main__':
    app.run(debug=True)
