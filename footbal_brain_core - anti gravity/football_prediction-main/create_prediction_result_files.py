"""
7 Temmuz verilerini 2 dosyaya böl:
1. prediction_matches.csv -> Sadece maç bilgileri (sonuç yok)
2. results_matches.csv -> Aynı maçlar + gerçek sonuçlar
"""
import pandas as pd

print("="*60)
print("7 TEMMUZ VERİSİ BÖLÜNÜYOR")
print("="*60)

# Ana veriyi oku
df = pd.read_csv('son_4_ay_tum_maclarin_verisi/7_temmuz_ve_sonrasi_TUM_VERI.csv')
print(f"\n✅ {len(df)} maç yüklendi")

# Gerçek sonucu hesapla
def calculate_result(row):
    h = row.get('home_goals')
    a = row.get('away_goals')
    
    if pd.isna(h) or pd.isna(a):
        return None
    
    if h > a:
        return 'HOME'
    elif a > h:
        return 'AWAY'
    else:
        return 'DRAW'

df['result'] = df.apply(calculate_result, axis=1)

# Sonuçlu maçları filtrele
df_with_results = df[df['result'].notna()].copy()
print(f"✅ {len(df_with_results)} maçta sonuç var")

# ============================================================================
# DOSYA 1: TAHMİN DOSYASI (Sonuçsuz - sadece maç bilgileri)
# ============================================================================
# Sonuç sütunlarını çıkar
prediction_df = df_with_results.drop(columns=['result', 'home_goals', 'away_goals'], errors='ignore')

prediction_df.to_csv('prediction_matches.csv', index=False)
print(f"\n📋 DOSYA 1 OLUŞTURULDU: prediction_matches.csv")
print(f"   {len(prediction_df)} maç (SONUÇSUZ)")
print(f"   Sütunlar: {len(prediction_df.columns)}")

# ============================================================================
# DOSYA 2: SONUÇ DOSYASI (Sadece maç ID + gerçek sonuç)
# ============================================================================
# Minimal bilgi: home, away, date, sonuç
result_df = df_with_results[['home_team', 'away_team', 'date', 'home_goals', 'away_goals', 'result']].copy()

result_df.to_csv('results_matches.csv', index=False)
print(f"\n✅ DOSYA 2 OLUŞTURULDU: results_matches.csv")
print(f"   {len(result_df)} maç (SONUÇLU)")
print(f"   Sütunlar: home_team, away_team, date, home_goals, away_goals, result")

# Test: İlk 3 maç
print("\n" + "="*60)
print("TEST: İLK 3 MAÇ")
print("="*60)

for idx in range(min(3, len(df_with_results))):
    pred_row = prediction_df.iloc[idx]
    result_row = result_df.iloc[idx]
    
    print(f"\n{idx+1}. MAÇ:")
    print(f"   Prediction dosyası: {pred_row.get('home_team', 'N/A')} vs {pred_row.get('away_team', 'N/A')}")
    print(f"   Result sütunu var mı? {'result' in prediction_df.columns}")
    print(f"   Result dosyası: {result_row['home_team']} vs {result_row['away_team']} -> {result_row['result']}")

print("\n" + "="*60)
print("✅ HAZIR! Artık sistem:")
print("   1) prediction_matches.csv'den maç okur (sonuç yok)")
print("   2) Tahmin yapar")
print("   3) results_matches.csv'den gerçek sonucu alır")
print("   4) Karşılaştırır ve öğrenir!")
print("="*60)



