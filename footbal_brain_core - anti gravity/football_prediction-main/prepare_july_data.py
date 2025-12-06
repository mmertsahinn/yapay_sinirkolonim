"""
7 Temmuz 2025 ve sonrası veriyi ayır
2 CSV oluştur:
1. TÜM VERİ (tüm sütunlar)
2. TAKVİM (sadece tarih + takımlar)
"""
import pandas as pd
import os

print("="*80)
print("📅 7 TEMMUZ 2025 VE SONRASI VERİ HAZIRLANIYOR")
print("="*80)

# Ana CSV'yi yükle
print("\n1️⃣ Ana CSV yükleniyor...")
df = pd.read_csv('football_match_data.csv', low_memory=False)
print(f"   ✅ Toplam: {len(df)} maç")

# Tarihleri parse et
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 7 Temmuz 2025 öncesi ve sonrası ayır
print("\n2️⃣ 7 Temmuz 2025'e göre ayırıyorum...")
july_7_2025 = pd.to_datetime('2025-07-07')

df_before = df[df['date'] < july_7_2025].copy()
df_after = df[df['date'] >= july_7_2025].copy()

print(f"   ✅ 7 Temmuz ÖNCESİ: {len(df_before)} maç")
print(f"   ✅ 7 Temmuz SONRASI: {len(df_after)} maç")

# Kronolojik sırala (7 Temmuz sonrası)
print("\n3️⃣ 7 Temmuz sonrası kronolojik sıralanıyor...")
df_after = df_after.sort_values('date').reset_index(drop=True)
print(f"   ✅ İlk maç: {df_after['date'].min()}")
print(f"   ✅ Son maç: {df_after['date'].max()}")

# Klasör oluştur
output_dir = "son_4_ay_tum_maclarin_verisi"
os.makedirs(output_dir, exist_ok=True)
print(f"\n4️⃣ Klasör oluşturuldu: {output_dir}/")

# 1) TÜM VERİ (tüm sütunlar)
output_full = os.path.join(output_dir, "7_temmuz_ve_sonrasi_TUM_VERI.csv")
df_after.to_csv(output_full, index=False, encoding='utf-8')
print(f"\n✅ TÜM VERİ kaydedildi: {output_full}")
print(f"   • Satır: {len(df_after)}")
print(f"   • Sütun: {len(df_after.columns)}")

# 2) TAKVİM (sadece tarih + takımlar)
print(f"\n5️⃣ Takvim CSV'si oluşturuluyor...")
df_takvim = df_after[['date', 'home_team', 'away_team']].copy()

output_takvim = os.path.join(output_dir, "7_temmuz_ve_sonrasi_TAKVIM.csv")
df_takvim.to_csv(output_takvim, index=False, encoding='utf-8')
print(f"\n✅ TAKVİM kaydedildi: {output_takvim}")
print(f"   • Satır: {len(df_takvim)}")
print(f"   • Sütun: {len(df_takvim.columns)} (date, home_team, away_team)")

# İlk ve son 5 maçı göster
print(f"\n{'='*80}")
print("📊 İLK 5 MAÇ:")
print(f"{'='*80}")
print(df_takvim.head(5).to_string(index=False))

print(f"\n{'='*80}")
print("📊 SON 5 MAÇ:")
print(f"{'='*80}")
print(df_takvim.tail(5).to_string(index=False))

# Ana CSV'yi güncelle (7 Temmuz öncesini kaydet)
print(f"\n{'='*80}")
print("⚠️ ANA CSV'Yİ GÜNCELLEYELİM Mİ?")
print(f"{'='*80}")
print(f"   Şu an: {len(df)} maç")
print(f"   Yeni: {len(df_before)} maç (7 Temmuz öncesi)")
print(f"   Silinecek: {len(df_after)} maç (7 Temmuz sonrası)")

cevap = input("\n❓ Ana CSV'den 7 Temmuz sonrasını sil? (evet/hayir): ").strip().lower()

if cevap in ['evet', 'e', 'yes', 'y']:
    # Yedek al
    backup_path = 'football_match_data_BACKUP.csv'
    df.to_csv(backup_path, index=False, encoding='utf-8')
    print(f"\n💾 Yedek alındı: {backup_path}")
    
    # Güncelle
    df_before.to_csv('football_match_data.csv', index=False, encoding='utf-8')
    print(f"✅ Ana CSV güncellendi!")
    print(f"   Yeni toplam: {len(df_before)} maç")
else:
    print(f"\n❌ Ana CSV değiştirilmedi")

print(f"\n{'='*80}")
print("✅ İŞLEM TAMAMLANDI!")
print(f"{'='*80}")
print(f"\n📂 Oluşturulan dosyalar:")
print(f"   1. {output_full}")
print(f"   2. {output_takvim}")
if cevap in ['evet', 'e', 'yes', 'y']:
    print(f"   3. {backup_path} (yedek)")
print(f"\n{'='*80}\n")




