"""
7 Temmuz 2025 sonrası veri kontrolü
"""
import pandas as pd

# CSV'yi yükle
df = pd.read_csv('football_match_data.csv', low_memory=False)

print(f"Toplam maç: {len(df)}")

# Tarihleri parse et
df['date'] = pd.to_datetime(df['date'], errors='coerce')

print(f"En eski maç: {df['date'].min()}")
print(f"En yeni maç: {df['date'].max()}")

# 7 Temmuz 2025 sonrası
july_2025 = df[df['date'] >= '2025-07-07']

print(f"\n7 Temmuz 2025 ve sonrası: {len(july_2025)} maç")

if len(july_2025) > 0:
    print(f"\nİlk 5 maç:")
    print(july_2025[['date', 'home_team', 'away_team']].head(5))
    print(f"\nSon 5 maç:")
    print(july_2025[['date', 'home_team', 'away_team']].tail(5))
    
    print(f"\n⚠️ 7 Temmuz 2025 sonrası {len(july_2025)} maç VAR!")
    print("Bu maçları SİLMEK için prepare_incremental_simple.py çalıştır")
else:
    print("\n✅ 7 Temmuz 2025 sonrası veri YOK (temiz)")

