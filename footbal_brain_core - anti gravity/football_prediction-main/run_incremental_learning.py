"""
INCREMENTAL LEARNING - TAM SİSTEM
7 Temmuz sonrası maçlarla öğrenme
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from advanced_incremental_system import AdvancedIncrementalLearner
import os

print("=" * 80)
print("🧠 INCREMENTAL LEARNING - TAM SİSTEM")
print("=" * 80)

# ============================================================================
# 1. MODELLERİ YÜKLE
# ============================================================================

print("\n[1/6] Modeller yukleniyor...")
try:
    ensemble = joblib.load('football_prediction_ensemble.joblib')
    le = joblib.load('label_encoder.joblib')
    print("   ✓ Ensemble model yuklendi")
except:
    print("   ✗ Model bulunamadi! Önce train_enhance_v2.py calistirin!")
    exit(1)

# ============================================================================
# 2. INCREMENTAL LEARNER BAŞLAT
# ============================================================================

print("\n[2/6] Incremental learning sistemi baslatiliyor...")
learner = AdvancedIncrementalLearner(n_features=58)

# Önceki öğrenmeyi yükle (varsa)
learner.load('incremental_learning_state.joblib')

# ============================================================================
# 3. TEST VERİSİNİ YÜKLE (2 CSV)
# ============================================================================

print("\n[3/6] Test verisi yukleniyor...")

# CSV 1: Takvim (Sadece tarih + takımlar - tahmin için)
takvim_file = '2025_temmuz_sonrasi_TAKVIM.csv'
if not os.path.exists(takvim_file):
    print(f"   ✗ Takvim dosyasi bulunamadi: {takvim_file}")
    print("   Önce 'python prepare_incremental_simple.py' calistirin!")
    exit(1)

df_takvim = pd.read_csv(takvim_file)
df_takvim['date'] = pd.to_datetime(df_takvim['date'])
print(f"   ✓ TAKVIM: {len(df_takvim)} mac (kronolojik)")

# CSV 2: Sonuçlar (Tüm veriler - öğrenme için)
sonuc_file = '2025_temmuz_sonrasi_SONUCLAR.csv'
if not os.path.exists(sonuc_file):
    print(f"   ✗ Sonuc dosyasi bulunamadi: {sonuc_file}")
    print("   Önce 'python prepare_incremental_simple.py' calistirin!")
    exit(1)

df_sonuclar = pd.read_csv(sonuc_file, low_memory=False)
df_sonuclar['date'] = pd.to_datetime(df_sonuclar['date'])
print(f"   ✓ SONUCLAR: {len(df_sonuclar)} mac (tum veriler)")

print(f"\n   Tarih araligi: {df_takvim['date'].min()} - {df_takvim['date'].max()}")

# ============================================================================
# 4. ÖĞRENME DÖNGÜSÜ
# ============================================================================

print("\n[4/6] Incremental learning basladi...")
print("=" * 80)

results = []
correct_count = 0
total_count = 0

# Feature names (train_enhance_v2.py ile aynı sırada)
feature_cols = [
    'home_team_strength', 'away_team_strength', 'home_team_defense', 'away_team_defense',
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
    'odds_market_efficiency', 'odds_entropy',
    'hype_odds_consistency_home', 'hype_odds_consistency_away', 'hype_odds_consistency_draw',
    'hype_odds_diff_home', 'hype_odds_diff_away', 'hype_odds_diff_draw',
    'home_hype_inflation_score', 'away_hype_inflation_score', 'total_hype_inflation',
    'tweets_odds_variance', 'tweets_per_odds_prob',
    'implied_prob_over_25', 'implied_prob_under_25', 'tweets_over_odds',
    'ah_home_implied_prob', 'ah_away_implied_prob',
    'hype_ah_consistency_home', 'hype_ah_consistency_away', 'market_consensus'
]

# Her maç için (TAKVIM'den kronolojik)
for idx, takvim_row in df_takvim.iterrows():
    total_count += 1
    
    match_date = takvim_row['date']
    home_team = takvim_row['home_team']
    away_team = takvim_row['away_team']
    
    # SONUCLAR CSV'sinden bu maçı bul
    sonuc_row = df_sonuclar[
        (df_sonuclar['date'] == match_date) &
        (df_sonuclar['home_team'] == home_team) &
        (df_sonuclar['away_team'] == away_team)
    ]
    
    if len(sonuc_row) == 0:
        print(f"\n⚠️  [{total_count}] Sonuç bulunamadı: {home_team} vs {away_team}")
        continue
    
    sonuc_row = sonuc_row.iloc[0]
    
    # Özellikleri çıkar
    features = []
    for col in feature_cols:
        val = sonuc_row.get(col, 0.0)
        features.append(val if pd.notna(val) else 0.0)
    
    features = np.array(features[:58])
    
    # Gerçek sonuç
    home_goals = sonuc_row.get('home_goals')
    away_goals = sonuc_row.get('away_goals')
    
    if pd.isna(home_goals) or pd.isna(away_goals):
        continue
    
    if home_goals > away_goals:
        actual_result = 'home_win'
    elif home_goals < away_goals:
        actual_result = 'away_win'
    else:
        actual_result = 'draw'
    
    # ========================================
    # TAHMIN YAP (Ensemble)
    # ========================================
    try:
        X = features.reshape(1, -1)
        base_proba = ensemble.predict_proba(X)[0]
        base_prediction = le.inverse_transform([ensemble.predict(X)[0]])[0]
        
        # ========================================
        # INCREMENTAL LEARNING İLE AYARLA
        # ========================================
        adjusted_proba = learner.adjust_prediction(features, base_proba)
        adjusted_prediction = le.classes_[np.argmax(adjusted_proba)]
        
        # ========================================
        # SONUCU GÖSTER (Her 10 maçta bir)
        # ========================================
        base_correct = (base_prediction == actual_result)
        adjusted_correct = (adjusted_prediction == actual_result)
        
        if base_correct:
            correct_count += 1
        
        if total_count % 10 == 0 or total_count <= 5:
            print(f"\n[Maç {total_count}] {match_date.strftime('%Y-%m-%d')} | {home_team} vs {away_team}")
            print(f"   Gerçek: {actual_result} ({int(home_goals)}-{int(away_goals)})")
            print(f"   Base Tahmin: {base_prediction} ({base_proba.max()*100:.1f}%) {'✓' if base_correct else '✗'}")
            print(f"   Adjusted: {adjusted_prediction} ({adjusted_proba.max()*100:.1f}%) {'✓' if adjusted_correct else '✗'}")
        
        # ========================================
        # ÖĞREN!
        # ========================================
        learning_result = learner.learn_from_match(
            features=features,
            predicted_proba=base_proba,
            predicted_class=base_prediction,
            actual_class=actual_result
        )
        
        results.append({
            'date': match_date,
            'home_team': home_team,
            'away_team': away_team,
            'actual': actual_result,
            'score': f"{int(home_goals)}-{int(away_goals)}",
            'base_pred': base_prediction,
            'adjusted_pred': adjusted_prediction,
            'base_correct': base_correct,
            'adjusted_correct': adjusted_correct,
            'loss': learning_result['loss'],
            'learning_rate': learning_result['learning_rate']
        })
        
    except Exception as e:
        print(f"   ✗ Hata: {e}")
        continue

# ============================================================================
# 5. SONUÇLARI ANALİZ ET
# ============================================================================

print("\n[5/6] Sonuclar analiz ediliyor...")
print("=" * 80)

df_results = pd.DataFrame(results)

base_accuracy = df_results['base_correct'].mean()
adjusted_accuracy = df_results['adjusted_correct'].mean()

print(f"\n📊 PERFORMANS KARŞILAŞTIRMASI:")
print(f"   Base Ensemble:       {base_accuracy*100:.2f}%")
print(f"   + Incremental:       {adjusted_accuracy*100:.2f}%")
print(f"   İyileşme:            {(adjusted_accuracy - base_accuracy)*100:+.2f}%")

# Zaman içinde iyileşme
if len(df_results) >= 50:
    first_half = df_results[:len(df_results)//2]['adjusted_correct'].mean()
    second_half = df_results[len(df_results)//2:]['adjusted_correct'].mean()
    print(f"\n📈 ÖĞRENME TRENDİ:")
    print(f"   İlk yarı:  {first_half*100:.2f}%")
    print(f"   İkinci yarı: {second_half*100:.2f}%")
    print(f"   Gelişme:   {(second_half - first_half)*100:+.2f}%")

# ============================================================================
# 6. ÖĞRENME STATE'İNİ KAYDET
# ============================================================================

print("\n[6/6] Ogrenme state'i kaydediliyor...")
learner.save('incremental_learning_state.joblib')

# Diagnostics
learner.print_diagnostics()

# Sonuçları kaydet
df_results.to_csv('incremental_learning_results.csv', index=False)
print(f"\n💾 Detaylı sonuçlar: incremental_learning_results.csv")

print("\n" + "=" * 80)
print("✅ INCREMENTAL LEARNING TAMAMLANDI!")
print("=" * 80)
print(f"\nToplam {total_count} maç üzerinde öğrenme yapıldı")
print(f"Sistem artık gelecek tahminlerde bu bilgileri kullanacak!")
print("\nBir sonraki maç için:")
print("   python app.py  (Flask'ta incremental learning aktif olacak)")
print("=" * 80)

