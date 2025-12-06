"""
7 Temmuz ayırma işlemini tamamla:
1. Ana CSV'den 7 Temmuz sonrasını sil
2. Backup'ı son_4_ay klasörüne taşı
"""
import pandas as pd
import shutil
import os

print("="*80)
print("🔧 7 TEMMUZ AYIRMA İŞLEMİ TAMAMLANIYOR")
print("="*80)

# 1) Ana CSV'yi yükle
print("\n1️⃣ Ana CSV yükleniyor...")
df = pd.read_csv('football_match_data.csv', low_memory=False)
print(f"   Şu an: {len(df)} maç")

# 2) Tarihleri parse et
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 3) 7 Temmuz öncesini al
july_7_2025 = pd.to_datetime('2025-07-07')
df_before = df[df['date'] < july_7_2025].copy()
df_after = df[df['date'] >= july_7_2025].copy()

print(f"\n2️⃣ Ayırma:")
print(f"   7 Temmuz ÖNCESİ: {len(df_before)} maç → Ana CSV'de KALACAK")
print(f"   7 Temmuz SONRASI: {len(df_after)} maç → SILINECEK")

# 4) Backup al
backup_path = 'football_match_data_BACKUP.csv'
print(f"\n3️⃣ Yedek alınıyor...")
df.to_csv(backup_path, index=False, encoding='utf-8')
print(f"   ✅ {backup_path}")

# 5) Ana CSV'yi güncelle
print(f"\n4️⃣ Ana CSV güncelleniyor...")
df_before.to_csv('football_match_data.csv', index=False, encoding='utf-8')
print(f"   ✅ football_match_data.csv")
print(f"   Yeni toplam: {len(df_before)} maç")

# 6) Backup'ı son_4_ay klasörüne taşı
output_dir = "son_4_ay_tum_maclarin_verisi"
backup_new_path = os.path.join(output_dir, "BACKUP_79163_mac_tum_veri.csv")

print(f"\n5️⃣ Backup taşınıyor...")
shutil.move(backup_path, backup_new_path)
print(f"   ✅ {backup_new_path}")

print(f"\n{'='*80}")
print("✅ İŞLEM TAMAMLANDI!")
print(f"{'='*80}")
print(f"\n📂 Ana dizin:")
print(f"   • football_match_data.csv → {len(df_before)} maç (7 Temmuz öncesi)")
print(f"\n📂 son_4_ay_tum_maclarin_verisi/")
print(f"   • 7_temmuz_ve_sonrasi_TUM_VERI.csv → {len(df_after)} maç")
print(f"   • 7_temmuz_ve_sonrasi_TAKVIM.csv → {len(df_after)} maç")
print(f"   • BACKUP_79163_mac_tum_veri.csv → {len(df)} maç (orijinal)")
print(f"\n{'='*80}\n")




