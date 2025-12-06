"""
Basit test: Sadece ensemble
"""
import pandas as pd
import numpy as np
import joblib

# Model yükle
ensemble = joblib.load('football_prediction_ensemble.joblib')
le = joblib.load('label_encoder.joblib')

# CSV yükle
df = pd.read_csv('football_match_data.csv', low_memory=False)

# İlk maç
match = df.iloc[0]

# Feature listesi (train_enhance_v2.py'den)
feature_names = ['home_team_strength', 'away_team_strength', 'home_team_defense', 'away_team_defense',
    'home_xG', 'away_xG', 'xG_difference', 'home_form', 'away_form',
    'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate',
    'goal_ratio', 'xG_ratio', 'day_of_week', 'month',
    'home_support', 'away_support', 'support_difference', 'support_ratio',
    'sentiment_score', 'sentiment_positive', 'sentiment_negative',
    'total_tweets', 'log_total_tweets', 'high_hype', 'hype_score',
    'implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
    'hype_favored_team', 'odds_favored_team', 'hype_odds_alignment',
    'home_hype_inflation', 'away_hype_inflation', 'hype_inflation_score',
    'high_engagement_match', 'hype_odds_discrepancy', 'tweets_odds_ratio',
    'odds_market_efficiency',
    'odds_entropy',
    'hype_odds_consistency_home', 'hype_odds_consistency_away', 'hype_odds_consistency_draw',
    'hype_odds_diff_home', 'hype_odds_diff_away', 'hype_odds_diff_draw',
    'home_hype_inflation_score', 'away_hype_inflation_score', 'total_hype_inflation',
    'tweets_odds_variance', 'tweets_per_odds_prob',
    'implied_prob_over_25', 'implied_prob_under_25', 'tweets_over_odds',
    'ah_home_implied_prob', 'ah_away_implied_prob',
    'hype_ah_consistency_home', 'hype_ah_consistency_away',
    'market_consensus']

print(f"Feature sayısı: {len(feature_names)}")

# Features çıkar
features = []
for feat in feature_names:
    features.append(match.get(feat, 0.0))

features = np.array(features).reshape(1, -1)

print(f"Features shape: {features.shape}")
print(f"Features: {features[0][:5]}...")

# Tahmin
proba = ensemble.predict_proba(features)[0]
pred = le.inverse_transform([np.argmax(proba)])[0]

print(f"\n✅ TAHMİN BAŞARILI!")
print(f"Proba: {proba}")
print(f"Tahmin: {pred}")




