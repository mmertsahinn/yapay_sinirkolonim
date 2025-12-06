import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("=" * 60)
print("FOOTBALL PREDICTION MODEL - EGITIM BASLADI")
print("=" * 60)
print("\n[1/7] CSV dosyasi yukleniyor...")
try:
    df = pd.read_csv('football_match_data.csv', low_memory=False)
    print(f"   ✓ {len(df)} mac verisi yuklendi")
    print(f"   ✓ {len(df.columns)} ozellik bulundu")
except Exception as e:
    print(f"   ✗ HATA: CSV yuklenemedi - {e}")
    raise

# Feature engineering
def engineer_features(df):
    df['goal_difference'] = df['home_goals'] - df['away_goals']
    df['xG_difference'] = df['home_xG'] - df['away_xG']
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort the dataframe by date
    df = df.sort_values('date')
    
    # Calculate team strengths based on rolling average
    window = 10
    df['home_team_strength'] = df.groupby('home_team')['home_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['away_team_strength'] = df.groupby('away_team')['away_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['home_team_defense'] = df.groupby('home_team')['away_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['away_team_defense'] = df.groupby('away_team')['home_goals'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
    # Calculate recent form (last 5 matches)
    df['home_form'] = df.groupby('home_team')['goal_difference'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['away_form'] = df.groupby('away_team')['goal_difference'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Head-to-head performance
    def get_h2h_stats(group):
        home_wins = (group['home_goals'] > group['away_goals']).sum()
        away_wins = (group['home_goals'] < group['away_goals']).sum()
        draws = (group['home_goals'] == group['away_goals']).sum()
        total_matches = len(group)
        return pd.Series({
            'h2h_home_win_rate': home_wins / total_matches if total_matches > 0 else 0.5,
            'h2h_away_win_rate': away_wins / total_matches if total_matches > 0 else 0.5,
            'h2h_draw_rate': draws / total_matches if total_matches > 0 else 0.5
        })

    try:
        # Pandas 2.1+ için
        h2h_stats = df.groupby(['home_team', 'away_team']).apply(get_h2h_stats, include_groups=False).reset_index()
    except TypeError:
        # Eski pandas versiyonları için
        h2h_stats = df.groupby(['home_team', 'away_team']).apply(get_h2h_stats).reset_index()
    df = pd.merge(df, h2h_stats, on=['home_team', 'away_team'], how='left')
    
    # Create 'result' column
    df['result'] = np.select(
        [df['goal_difference'] > 0, df['goal_difference'] < 0, df['goal_difference'] == 0],
        ['home_win', 'away_win', 'draw'],
        default='draw'
    ).astype(str)
    
    # Additional features
    df['goal_ratio'] = df['home_goals'] / (df['away_goals'] + 1e-5)  # Avoid division by zero
    df['xG_ratio'] = df['home_xG'] / (df['away_xG'] + 1e-5)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Social media and hype features (from football_brain_export.json)
    # Fill missing values with mean or forward fill for time-series continuity
    if 'home_support' in df.columns:
        # Support difference (higher support for home team = advantage)
        df['support_difference'] = df['home_support'].fillna(0.5) - df['away_support'].fillna(0.5)
        # Support ratio
        df['support_ratio'] = df['home_support'].fillna(0.5) / (df['away_support'].fillna(0.5) + 1e-5)
        
        # Forward fill for time-series data (use previous match's hype data if available)
        df['home_support'] = df.groupby('home_team')['home_support'].ffill().fillna(0.5)
        df['away_support'] = df.groupby('away_team')['away_support'].ffill().fillna(0.5)
    else:
        # Create columns with default values if they don't exist
        df['home_support'] = 0.5
        df['away_support'] = 0.5
        df['support_difference'] = 0.0
        df['support_ratio'] = 1.0
    
    if 'sentiment_score' in df.columns:
        # Sentiment score indicates overall match sentiment
        # Fill missing values with forward fill or mean
        df['sentiment_score'] = df.groupby(['home_team', 'away_team'])['sentiment_score'].ffill().fillna(0.0)
        # Create sentiment categories
        df['sentiment_positive'] = (df['sentiment_score'] > 0.2).astype(int)
        df['sentiment_negative'] = (df['sentiment_score'] < -0.2).astype(int)
    else:
        df['sentiment_score'] = 0.0
        df['sentiment_positive'] = 0
        df['sentiment_negative'] = 0
    
    if 'total_tweets' in df.columns:
        # Log transform to handle outliers (social media data often skewed)
        df['total_tweets'] = df['total_tweets'].fillna(0)
        df['log_total_tweets'] = np.log1p(df['total_tweets'])  # log(1+x) to handle zeros
        # Hype level categories
        tweets_75th = df['total_tweets'].quantile(0.75) if df['total_tweets'].notna().sum() > 0 else 0
        df['high_hype'] = (df['total_tweets'] > tweets_75th).astype(int)
    else:
        df['total_tweets'] = 0
        df['log_total_tweets'] = 0.0
        df['high_hype'] = 0
    
    # Combined hype metric (combining support, sentiment, and volume)
    df['hype_score'] = (
        (df['home_support'] + df['away_support']) * 0.4 +
        (df['sentiment_score'] + 1) * 0.3 +  # normalize sentiment to 0-1
        np.clip(df['log_total_tweets'] / 10, 0, 1) * 0.3  # normalize log tweets
    )
    
    # ODDŞ-HYPE İLİŞKİ ÖZELLİKLERİ (YENİ!)
    # Odds kolonlarını kontrol et ve yoksa oluştur
    if 'odds_b365_h' not in df.columns:
        df['odds_b365_h'] = np.nan
    if 'odds_b365_d' not in df.columns:
        df['odds_b365_d'] = np.nan
    if 'odds_b365_a' not in df.columns:
        df['odds_b365_a'] = np.nan
    if 'odds_max_h' not in df.columns:
        df['odds_max_h'] = np.nan
    if 'odds_max_d' not in df.columns:
        df['odds_max_d'] = np.nan
    if 'odds_max_a' not in df.columns:
        df['odds_max_a'] = np.nan
    if 'odds_avg_h' not in df.columns:
        df['odds_avg_h'] = np.nan
    if 'odds_avg_d' not in df.columns:
        df['odds_avg_d'] = np.nan
    if 'odds_avg_a' not in df.columns:
        df['odds_avg_a'] = np.nan
    
    if df['odds_b365_h'].notna().sum() > 0:
        # Odds'ları implied probability'ye çevir
        df['implied_prob_home'] = 1.0 / (df['odds_b365_h'].fillna(2.0) + 1e-5)
        df['implied_prob_draw'] = 1.0 / (df['odds_b365_d'].fillna(3.5) + 1e-5)
        df['implied_prob_away'] = 1.0 / (df['odds_b365_a'].fillna(2.0) + 1e-5)
        
        # Normalize probabilities (toplam 1 olsun)
        prob_sum = df['implied_prob_home'] + df['implied_prob_draw'] + df['implied_prob_away']
        df['implied_prob_home'] = df['implied_prob_home'] / (prob_sum + 1e-5)
        df['implied_prob_draw'] = df['implied_prob_draw'] / (prob_sum + 1e-5)
        df['implied_prob_away'] = df['implied_prob_away'] / (prob_sum + 1e-5)
        
        # Hype'den beklenen sonuç (hangi takım daha popüler)
        df['hype_favored_team'] = np.where(df['home_support'] > df['away_support'], 1, 0)  # 1=home, 0=away
        
        # Odds'tan beklenen sonuç (hangi takım daha favori)
        df['odds_favored_team'] = np.where(
            df['implied_prob_home'] > df['implied_prob_away'], 1, 0
        )
        
        # Hype-Odds uyumu (aynı takım favori mi?)
        df['hype_odds_alignment'] = (df['hype_favored_team'] == df['odds_favored_team']).astype(float)
        
        # Hype şişirme: Hype çok yüksek ama odds düşük mü? (aşırı hype)
        # Ev sahibi için
        high_home_hype = (df['home_support'] > 0.7).astype(float)
        low_home_odds = (df['implied_prob_home'] < 0.4).astype(float)  # Düşük şans
        df['home_hype_inflation'] = high_home_hype * low_home_odds
        
        # Deplasman için
        high_away_hype = (df['away_support'] > 0.7).astype(float)
        low_away_odds = (df['implied_prob_away'] < 0.4).astype(float)
        df['away_hype_inflation'] = high_away_hype * low_away_odds
        
        # Genel hype şişirme metriği
        df['hype_inflation_score'] = df['home_hype_inflation'] + df['away_hype_inflation']
        
        # Odds-Twitter ilişkisi
        # Yüksek tweet sayısı + yüksek odds = popüler maç
        high_tweets = (df['total_tweets'] > df['total_tweets'].quantile(0.75)).astype(float) if df['total_tweets'].notna().sum() > 0 else 0
        high_odds_variance = (df['implied_prob_home'] - df['implied_prob_away']).abs() < 0.2  # Yakın maç
        df['high_engagement_match'] = (high_tweets * high_odds_variance).astype(float)
        
        # Hype-odds farkı (ne kadar uyumsuz?)
        # Hype çok yüksek ama odds çok düşük = şişirme
        home_hype_odds_diff = df['home_support'] - df['implied_prob_home']
        away_hype_odds_diff = df['away_support'] - df['implied_prob_away']
        df['hype_odds_discrepancy'] = (home_hype_odds_diff.abs() + away_hype_odds_diff.abs()) / 2
        
        # Odds market efficiency: Market ortalama ile tek bahis şirketi farkı
        if 'odds_max_h' in df.columns and 'odds_avg_h' in df.columns:
            max_avg_diff = (df['odds_max_h'] - df['odds_avg_h']).abs() / (df['odds_avg_h'] + 1e-5)
            df['odds_market_efficiency'] = 1.0 - np.clip(max_avg_diff, 0, 1)  # 0-1 arası normalize
        else:
            df['odds_market_efficiency'] = 0.8  # Default
        
        # Tweet sayısı ile odds değişimi ilişkisi
        # Daha fazla tweet = daha fazla ilgi = odds değişebilir
        df['tweets_odds_ratio'] = df['log_total_tweets'] / (df['implied_prob_home'] + 1e-5)
        
        # ODDS ENTROPY - Belirsizlik ölçüsü (ne kadar belirsiz maç?)
        # Yüksek entropy = belirsiz maç, düşük entropy = net favori var
        df['odds_entropy'] = -(
            df['implied_prob_home'] * np.log(df['implied_prob_home'] + 1e-9) +
            df['implied_prob_draw'] * np.log(df['implied_prob_draw'] + 1e-9) +
            df['implied_prob_away'] * np.log(df['implied_prob_away'] + 1e-9)
        )
        
        # HYPE-ODDS CONSISTENCY (Tutarlılık) - Hype ve odds ne kadar uyumlu?
        # Home için: Hype yüksek ve odds da yüksek = tutarlı
        df['hype_odds_consistency_home'] = df['home_support'] * df['implied_prob_home']
        df['hype_odds_consistency_away'] = df['away_support'] * df['implied_prob_away']
        
        # Draw için: Hype'ta belirsizlik varsa ve odds'ta da draw yüksekse tutarlı
        draw_hype_estimate = 1 - df['home_support'] - df['away_support']
        draw_hype_estimate = np.clip(draw_hype_estimate, 0, 1)  # 0-1 arası
        df['hype_odds_consistency_draw'] = draw_hype_estimate * df['implied_prob_draw']
        
        # HYPE-ODDS DIFFERENCE - Fark ne kadar?
        df['hype_odds_diff_home'] = df['home_support'] - df['implied_prob_home']
        df['hype_odds_diff_away'] = df['away_support'] - df['implied_prob_away']
        df['hype_odds_diff_draw'] = draw_hype_estimate - df['implied_prob_draw']
        
        # HYPE ŞİŞİRME ANALİZİ - Daha detaylı
        # Hype yüksek ama odds düşük = şişirme var
        # Home için şişirme
        df['home_hype_inflation_score'] = np.where(
            (df['home_support'] > 0.6) & (df['implied_prob_home'] < 0.4),
            df['home_support'] - df['implied_prob_home'],  # Fark ne kadar büyük?
            0.0
        )
        
        # Away için şişirme
        df['away_hype_inflation_score'] = np.where(
            (df['away_support'] > 0.6) & (df['implied_prob_away'] < 0.4),
            df['away_support'] - df['implied_prob_away'],
            0.0
        )
        
        # TOPLAM HYPE ŞİŞİRME
        df['total_hype_inflation'] = df['home_hype_inflation_score'] + df['away_hype_inflation_score']
        
        # TWEET SAYISI ve ODDS İLİŞKİSİ - Daha detaylı
        # Yüksek tweet + düşük odds variance = popüler ve belirsiz maç
        odds_variance = df[['implied_prob_home', 'implied_prob_draw', 'implied_prob_away']].std(axis=1)
        df['tweets_odds_variance'] = df['total_tweets'] * odds_variance
        
        # Tweet başına odds değişimi (normalize)
        df['tweets_per_odds_prob'] = df['total_tweets'] / (df['implied_prob_home'] + df['implied_prob_away'] + 1e-5)
        
        # OVER/UNDER ODDS İLE HYPE İLİŞKİSİ
        if 'odds_b365_over_25' in df.columns and df['odds_b365_over_25'].notna().sum() > 0:
            # Over/Under implied probability
            df['implied_prob_over_25'] = 1.0 / (df['odds_b365_over_25'].fillna(1.9) + 1e-5)
            df['implied_prob_under_25'] = 1.0 / (df['odds_b365_under_25'].fillna(1.9) + 1e-5)
            
            # Normalize
            ou_sum = df['implied_prob_over_25'] + df['implied_prob_under_25']
            df['implied_prob_over_25'] = df['implied_prob_over_25'] / (ou_sum + 1e-5)
            df['implied_prob_under_25'] = df['implied_prob_under_25'] / (ou_sum + 1e-5)
            
            # Tweet sayısı ile over/under odds ilişkisi
            # Yüksek tweet = yüksek gol beklentisi mi?
            df['tweets_over_odds'] = df['total_tweets'] * df['implied_prob_over_25']
        else:
            df['implied_prob_over_25'] = 0.5
            df['implied_prob_under_25'] = 0.5
            df['tweets_over_odds'] = 0.0
        
        # ASIAN HANDICAP İLE HYPE İLİŞKİSİ (eğer varsa)
        if 'odds_b365_ah_h' in df.columns and df['odds_b365_ah_h'].notna().sum() > 0:
            df['ah_home_implied_prob'] = 1.0 / (df['odds_b365_ah_h'].fillna(2.0) + 1e-5)
            df['ah_away_implied_prob'] = 1.0 / (df['odds_b365_ah_a'].fillna(2.0) + 1e-5)
            ah_sum = df['ah_home_implied_prob'] + df['ah_away_implied_prob']
            df['ah_home_implied_prob'] = df['ah_home_implied_prob'] / (ah_sum + 1e-5)
            df['ah_away_implied_prob'] = df['ah_away_implied_prob'] / (ah_sum + 1e-5)
            
            # Asian handicap ile hype tutarlılığı
            df['hype_ah_consistency_home'] = df['home_support'] * df['ah_home_implied_prob']
            df['hype_ah_consistency_away'] = df['away_support'] * df['ah_away_implied_prob']
        else:
            df['ah_home_implied_prob'] = 0.5
            df['ah_away_implied_prob'] = 0.5
            df['hype_ah_consistency_home'] = 0.0
            df['hype_ah_consistency_away'] = 0.0
        
        # MARKET EFFICIENCY - Daha detaylı
        if 'odds_max_h' in df.columns and 'odds_avg_h' in df.columns:
            # Max ve avg arasındaki fark ne kadar? (küçük fark = verimli market)
            max_avg_diff_h = (df['odds_max_h'] - df['odds_avg_h']).abs() / (df['odds_avg_h'] + 1e-5)
            max_avg_diff_a = (df['odds_max_a'] - df['odds_avg_a']).abs() / (df['odds_avg_a'] + 1e-5) if 'odds_max_a' in df.columns else 0
            max_avg_diff = (max_avg_diff_h + max_avg_diff_a) / 2
            df['odds_market_efficiency'] = 1.0 - np.clip(max_avg_diff, 0, 1)
            
            # Market consensus (tüm bookmaker'lar ne kadar uyumlu?)
            avg_cols = [col for col in ['odds_avg_h', 'odds_avg_d', 'odds_avg_a'] if col in df.columns]
            if len(avg_cols) >= 2:
                market_variance = df[avg_cols].std(axis=1)
                df['market_consensus'] = 1.0 / (market_variance + 1e-5)  # Düşük variance = yüksek consensus
            else:
                df['market_consensus'] = 0.5
        else:
            df['odds_market_efficiency'] = 0.8
            df['market_consensus'] = 0.5
        
        # Missing değerleri doldur
        odds_hype_cols = ['implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
                         'hype_favored_team', 'odds_favored_team', 'hype_odds_alignment',
                         'home_hype_inflation', 'away_hype_inflation', 'hype_inflation_score',
                         'high_engagement_match', 'hype_odds_discrepancy', 'tweets_odds_ratio',
                         'odds_entropy', 'hype_odds_consistency_home', 'hype_odds_consistency_away', 
                         'hype_odds_consistency_draw', 'hype_odds_diff_home', 'hype_odds_diff_away', 
                         'hype_odds_diff_draw', 'home_hype_inflation_score', 'away_hype_inflation_score',
                         'total_hype_inflation', 'tweets_odds_variance', 'tweets_per_odds_prob',
                         'implied_prob_over_25', 'implied_prob_under_25', 'tweets_over_odds',
                         'ah_home_implied_prob', 'ah_away_implied_prob', 'hype_ah_consistency_home',
                         'hype_ah_consistency_away', 'odds_market_efficiency', 'market_consensus']
        for col in odds_hype_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
    else:
        # Odds verisi yoksa default değerler
        df['implied_prob_home'] = 0.33
        df['implied_prob_draw'] = 0.33
        df['implied_prob_away'] = 0.33
        df['hype_favored_team'] = 0.0
        df['odds_favored_team'] = 0.0
        df['hype_odds_alignment'] = 0.5
        df['home_hype_inflation'] = 0.0
        df['away_hype_inflation'] = 0.0
        df['hype_inflation_score'] = 0.0
        df['high_engagement_match'] = 0.0
        df['hype_odds_discrepancy'] = 0.0
        df['odds_market_efficiency'] = 0.8
        df['tweets_odds_ratio'] = 0.0
        # Yeni eklenen özellikler için default değerler
        df['odds_entropy'] = 1.1  # Maksimum entropy (belirsizlik)
        df['hype_odds_consistency_home'] = 0.0
        df['hype_odds_consistency_away'] = 0.0
        df['hype_odds_consistency_draw'] = 0.0
        df['hype_odds_diff_home'] = 0.0
        df['hype_odds_diff_away'] = 0.0
        df['hype_odds_diff_draw'] = 0.0
        df['home_hype_inflation_score'] = 0.0
        df['away_hype_inflation_score'] = 0.0
        df['total_hype_inflation'] = 0.0
        df['tweets_odds_variance'] = 0.0
        df['tweets_per_odds_prob'] = 0.0
        df['implied_prob_over_25'] = 0.5
        df['implied_prob_under_25'] = 0.5
        df['tweets_over_odds'] = 0.0
        df['ah_home_implied_prob'] = 0.5
        df['ah_away_implied_prob'] = 0.5
        df['hype_ah_consistency_home'] = 0.0
        df['hype_ah_consistency_away'] = 0.0
        df['market_consensus'] = 0.5
    
    return df

print("\n[2/7] Ozellikler olusturuluyor...")
df = engineer_features(df)
print(f"   ✓ Feature engineering tamamlandi")

# Handle NaN values in the target variables
df = df.dropna(subset=['home_goals', 'away_goals'])

# Prepare features and target
features = ['home_team_strength', 'away_team_strength', 'home_team_defense', 'away_team_defense',
            'home_xG', 'away_xG', 'xG_difference', 'home_form', 'away_form',
            'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate',
            'goal_ratio', 'xG_ratio', 'day_of_week', 'month',
            # Social media and hype features
            'home_support', 'away_support', 'support_difference', 'support_ratio',
            'sentiment_score', 'sentiment_positive', 'sentiment_negative',
            'total_tweets', 'log_total_tweets', 'high_hype', 'hype_score',
            # Odds-Hype ilişki özellikleri (DETAYLI!)
            'implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
            'hype_favored_team', 'odds_favored_team', 'hype_odds_alignment',
            'home_hype_inflation', 'away_hype_inflation', 'hype_inflation_score',
            'high_engagement_match', 'hype_odds_discrepancy', 'tweets_odds_ratio',
            'odds_market_efficiency',
            # Odds-Hype Detaylı İlişkiler (YENİ!)
            'odds_entropy',  # Belirsizlik ölçüsü
            'hype_odds_consistency_home', 'hype_odds_consistency_away', 'hype_odds_consistency_draw',  # Tutarlılık
            'hype_odds_diff_home', 'hype_odds_diff_away', 'hype_odds_diff_draw',  # Fark metrikleri
            'home_hype_inflation_score', 'away_hype_inflation_score', 'total_hype_inflation',  # Şişirme skorları
            'tweets_odds_variance', 'tweets_per_odds_prob',  # Tweet-Odds ilişkileri
            'implied_prob_over_25', 'implied_prob_under_25', 'tweets_over_odds',  # Over/Under ilişkileri
            'ah_home_implied_prob', 'ah_away_implied_prob',  # Asian Handicap
            'hype_ah_consistency_home', 'hype_ah_consistency_away',  # Asian Handicap-Hype uyumu
            'market_consensus']  # Piyasa konsensüsü
X = df[features]
y_result = df['result']
y_home_goals = df['home_goals']
y_away_goals = df['away_goals']

# Encode the target variable for result prediction
le = LabelEncoder()
y_result_encoded = le.fit_transform(y_result)

print("\n[3/7] Veri egitim/test olarak ayrilıyor...")
# Time-based split
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_result_train, y_result_test = y_result_encoded[:train_size], y_result_encoded[train_size:]
y_home_goals_train, y_home_goals_test = y_home_goals[:train_size], y_home_goals[train_size:]
y_away_goals_train, y_away_goals_test = y_away_goals[:train_size], y_away_goals[train_size:]
print(f"   ✓ Egitim seti: {len(X_train)} mac")
print(f"   ✓ Test seti: {len(X_test)} mac")

# Create a pipeline with imputer, scaler, feature selection, and model
def create_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42))),
        ('model', model)
    ])

# Define models with hyperparameter distributions for random search
rf_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': [None] + list(randint(10, 50).rvs(4)),
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10),
    'feature_selection__max_features': randint(2, len(features))
}

xgb_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__subsample': uniform(0.5, 0.5),
    'model__colsample_bytree': uniform(0.5, 0.5),
    'feature_selection__max_features': randint(2, len(features))
}

gb_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__subsample': uniform(0.5, 0.5),
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10),
    'feature_selection__max_features': randint(2, len(features))
}

models = {
    'RandomForest': (create_pipeline(RandomForestClassifier(random_state=42)), rf_params),
    'XGBoost': (create_pipeline(XGBClassifier(random_state=42)), xgb_params),
    'GradientBoosting': (create_pipeline(GradientBoostingClassifier(random_state=42)), gb_params),
}

# Function to print feature importance
def print_feature_importance(pipeline, feature_names):
    # Get the final estimator (model) from the pipeline
    model = pipeline.named_steps['model']
    
    if hasattr(model, 'feature_importances_'):
        # Get the feature selector from the pipeline
        feature_selector = pipeline.named_steps['feature_selection']
        # Get the mask of selected features
        feature_mask = feature_selector.get_support()
        # Filter the feature names
        selected_features = [f for f, selected in zip(feature_names, feature_mask) if selected]
        
        importances = model.feature_importances_
        
        # Sort features by importance
        feature_importance = sorted(zip(importances, selected_features), reverse=True)
        
        print("Top 10 most important features:")
        for i, (importance, feature) in enumerate(feature_importance[:10], 1):
            print(f"{i}. {feature} ({importance:.6f})")
    else:
        print("This model doesn't have feature importances.")

# Load the previously tuned SVC model from the existing ensemble (if exists)
svc_model = None
try:
    print("Loading previously tuned SVC model...")
    ensemble_model = joblib.load('football_prediction_ensemble.joblib')
    svc_model = [model for name, model in ensemble_model.named_estimators_.items() if name == 'SVC'][0]
    print("SVC model loaded successfully!")
except FileNotFoundError:
    print("⚠️  No existing ensemble model found. SVC will be trained from scratch.")
except Exception as e:
    print(f"⚠️  Could not load SVC model: {e}. SVC will be trained from scratch.")

print("\n[4/7] Model hiperparametreleri optimize ediliyor (Bu islem uzun surebilir)...")
# Perform random search for each model (except SVC)
best_models = {}
model_count = len(models)
for idx, (name, (pipeline, params)) in enumerate(models.items(), 1):
    print(f"\n   [{idx}/{model_count}] {name} modeli egitiliyor...")
    print(f"      (20 parametre kombinasyonu, 3-fold CV ile test ediliyor...)")
    random_search = RandomizedSearchCV(pipeline, params, n_iter=20, cv=3, n_jobs=-1, verbose=0, random_state=42)
    random_search.fit(X_train, y_result_train)
    best_models[name] = random_search.best_estimator_
    print(f"   ✓ {name} tamamlandi - En iyi skor: {random_search.best_score_:.4f}")
    # print(f"Best parameters for {name}: {random_search.best_params_}")
    # print(f"\nFeature importance for {name}:")
    # print_feature_importance(random_search.best_estimator_, features)

# Add the pre-tuned SVC model to the best_models dictionary (if available)
if svc_model is not None:
    best_models['SVC'] = svc_model
else:
    # Train SVC from scratch if no existing model
    print("\nTraining SVC model from scratch...")
    svc_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42))),
        ('model', SVC(probability=True, random_state=42))
    ])
    svc_pipeline.fit(X_train, y_result_train)
    best_models['SVC'] = svc_pipeline
    print("SVC model trained successfully!")

# Create ensemble model for result prediction
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    voting='soft'
)

print("\n[5/7] Ensemble model egitiliyor...")
# Train and evaluate the ensemble model for result prediction
ensemble.fit(X_train, y_result_train)
y_result_pred = ensemble.predict(X_test)
result_accuracy = accuracy_score(y_result_test, y_result_pred)
print(f"   ✓ Ensemble model tamamlandi")
print(f"\n{'='*60}")
print("ENSEMBLE MODEL SONUCLARI:")
print(f"{'='*60}")
print(f"Dogruluk (Accuracy): {result_accuracy:.4f} ({result_accuracy*100:.2f}%)")
print("\nSiniflandirma Raporu:")
print(classification_report(y_result_test, y_result_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_result_test, y_result_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Cross-validation
cv_scores = cross_val_score(ensemble, X, y_result_encoded, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n[6/7] Gol tahmin modelleri egitiliyor...")
# Train models for score prediction
score_models = {
    'home_goals': RandomForestClassifier(n_estimators=100, random_state=42),
    'away_goals': RandomForestClassifier(n_estimators=100, random_state=42)
}

score_models['home_goals'].fit(X_train, y_home_goals_train)
print("   ✓ Ev sahibi gol modeli tamamlandi")
score_models['away_goals'].fit(X_train, y_away_goals_train)
print("   ✓ Deplasman gol modeli tamamlandi")

# Evaluate score prediction models
y_home_goals_pred = score_models['home_goals'].predict(X_test)
y_away_goals_pred = score_models['away_goals'].predict(X_test)
home_goals_mae = np.mean(np.abs(y_home_goals_test - y_home_goals_pred))
away_goals_mae = np.mean(np.abs(y_away_goals_test - y_away_goals_pred))
print(f"\nGol Tahmin Modeli Sonuclari:")
print(f"   Ev Sahibi Gol MAE: {home_goals_mae:.4f}")
print(f"   Deplasman Gol MAE: {away_goals_mae:.4f}")

print("\n[7/7] Modeller kaydediliyor...")
# Save the ensemble model, label encoder, and score models
try:
    joblib.dump(ensemble, 'football_prediction_ensemble.joblib')
    print("   ✓ Ensemble model kaydedildi (football_prediction_ensemble.joblib)")
    
    joblib.dump(le, 'label_encoder.joblib')
    print("   ✓ Label encoder kaydedildi (label_encoder.joblib)")
    
    joblib.dump(score_models['home_goals'], 'home_goals_model.joblib')
    print("   ✓ Ev sahibi gol modeli kaydedildi (home_goals_model.joblib)")
    
    joblib.dump(score_models['away_goals'], 'away_goals_model.joblib')
    print("   ✓ Deplasman gol modeli kaydedildi (away_goals_model.joblib)")
    
    print(f"\n{'='*60}")
    print("EGITIM BASARIYLA TAMAMLANDI!")
    print(f"{'='*60}")
    print(f"\nToplam {len(df)} mac verisiyle egitim yapildi")
    print(f"Model dogrulugu: {result_accuracy:.4f} ({result_accuracy*100:.2f}%)")
    print(f"\nKaydedilen dosyalar:")
    print("  - football_prediction_ensemble.joblib")
    print("  - label_encoder.joblib")
    print("  - home_goals_model.joblib")
    print("  - away_goals_model.joblib")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print(f"{'='*60}")
except Exception as e:
    print(f"\n✗ HATA: Modeller kaydedilemedi - {e}")
    raise