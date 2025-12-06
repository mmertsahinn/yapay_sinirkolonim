"""
INCREMENTAL LEARNING İÇİN VERİ HAZIRLAMA
2025-07-07 sonrası maçları 2 CSV'ye ayır
"""

import pandas as pd

print("=" * 80)
print("INCREMENTAL LEARNING VERİ HAZIRLAMA")
print("=" * 80)

# CSV'yi yükle
print("\n[1/4] CSV yukleniyor...")
df = pd.read_csv('football_match_data.csv', low_memory=False)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df = df.sort_values('date')
print(f"   ✓ {len(df)} mac yuklendi")

# 7 Temmuz 2025 sonrası
split_date = pd.to_datetime('2025-07-07')

df_before = df[df['date'] < split_date]
df_after = df[df['date'] >= split_date].copy()

print(f"\n[2/4] Veri ayrildi:")
print(f"   7 Temmuz 2025 oncesi: {len(df_before)} mac")
print(f"   7 Temmuz 2025 sonrasi: {len(df_after)} mac")

if len(df_after) == 0:
    print("\n   ⚠️ 7 Temmuz 2025 sonrasi mac yok!")
    exit(0)

# Kronolojik sırala
df_after = df_after.sort_values('date')

# CSV 1: Sadece tarih + takımlar (Tahmin için)
print(f"\n[3/4] Tahmin CSV'si olusturuluyor...")
df_schedule = df_after[['date', 'home_team', 'away_team']].copy()
df_schedule['date'] = df_schedule['date'].dt.strftime('%Y-%m-%d %H:%M')
df_schedule.to_csv('2025_temmuz_sonrasi_TAKVIM.csv', index=False)
print(f"   ✓ 2025_temmuz_sonrasi_TAKVIM.csv ({len(df_schedule)} mac)")
print(f"   İçerik: Sadece tarih, ev, deplasman")

# CSV 2: Tam veriler (Öğrenme için - sonuçlar dahil)
print(f"\n[4/4] Sonuc CSV'si olusturuluyor...")
df_after.to_csv('2025_temmuz_sonrasi_SONUCLAR.csv', index=False)
print(f"   ✓ 2025_temmuz_sonrasi_SONUCLAR.csv ({len(df_after)} mac)")
print(f"   İçerik: Tüm veriler (skor, xG, hype, odds)")

# Eğitim CSV'si (7 Temmuz öncesi)
print(f"\n[5/5] Egitim CSV'si olusturuluyor...")
df_before.to_csv('football_match_data_EGITIM.csv', index=False)
print(f"   ✓ football_match_data_EGITIM.csv ({len(df_before)} mac)")

# Önizleme
print("\n" + "=" * 80)
print("ÖNİZLEME - İLK 10 MAÇ (Kronolojik):")
print("=" * 80)
for idx, row in df_schedule.head(10).iterrows():
    print(f"{row['date']} | {row['home_team']:30s} vs {row['away_team']}")

print("\n" + "=" * 80)
print("HAZIR!")
print("=" * 80)
print(f"\n📅 TAKVIM CSV (Tahmin için):")
print(f"   2025_temmuz_sonrasi_TAKVIM.csv")
print(f"   {len(df_schedule)} mac kronolojik sırada")
print(f"\n📊 SONUÇ CSV (Öğrenme için):")
print(f"   2025_temmuz_sonrasi_SONUCLAR.csv")
print(f"   Tüm veriler (skor, xG, hype, odds)")
print(f"\n🎓 EĞİTİM CSV:")
print(f"   football_match_data_EGITIM.csv")
print(f"   7 Temmuz 2025 öncesi tüm maçlar")

print(f"\nSONRAKİ ADIMLAR:")
print(f"   1. football_match_data.csv'yi değiştir:")
print(f"      copy football_match_data_EGITIM.csv football_match_data.csv")
print(f"   2. Modeli eğit:")
print(f"      python train_enhance_v2.py")
print(f"   3. Incremental learning başlat:")
print(f"      python run_incremental_learning.py")
print("=" * 80)





