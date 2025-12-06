"""
Takım İsimlerini Normalize Eden Script
Aynı takımın farklı isimlerini birleştirir
"""

import pandas as pd
import re

print("=" * 80)
print("TAKIM İSİMLERİ NORMALİZASYONU")
print("=" * 80)

# CSV'yi yükle
print("\n[1/5] CSV yukleniyor...")
df = pd.read_csv('football_match_data.csv', low_memory=False)
print(f"   Toplam {len(df)} mac")

# Orijinal dosyayı yedekle
print("\n[2/5] Yedek olusturuluyor...")
df.to_csv('football_match_data_YEDEK.csv', index=False)
print("   ✓ Yedek: football_match_data_YEDEK.csv")

def normalize_team_name(name):
    """Takım ismini normalize et"""
    if pd.isna(name):
        return name
    
    name = str(name).strip()
    original_name = name
    
    # ADIM 1: Tam eşleşmeler (Manuel mapping)
    exact_replacements = {
        'Juventus FC': 'Juventus',
        'Inter Milan': 'Inter',
        'AC Milan': 'Milan',
        'Bayern Munich': 'Bayern München',
        'Manchester United': 'Manchester Utd',
        'Paris Saint Germain': 'Paris Saint-Germain',
        'PSG': 'Paris Saint-Germain',
        'Atlético Madrid': 'Atletico Madrid',
        'Atletico de Madrid': 'Atletico Madrid',
        'Real Madrid CF': 'Real Madrid',
        'FC Barcelona': 'Barcelona',
        'Borussia Dortmund': 'Dortmund',
        'Borussia Mönchengladbach': 'Mönchengladbach',
        'RB Leipzig': 'Leipzig',
        'Bayer Leverkusen': 'Leverkusen',
        'Bayer 04 Leverkusen': 'Leverkusen',
        'Eintracht Frankfurt': 'Frankfurt',
        'VfB Stuttgart': 'Stuttgart',
        'Werder Bremen': 'Bremen',
        'Schalke 04': 'Schalke',
        'Tottenham Hotspur': 'Tottenham',
        'Wolverhampton Wanderers': 'Wolves',
        'Brighton & Hove Albion': 'Brighton',
        'West Ham United': 'West Ham',
        'Newcastle United': 'Newcastle',
        'Leicester City': 'Leicester',
        'İstanbul Başakşehir': 'Basaksehir',
        'Istanbul Basaksehir': 'Basaksehir',
        'Fatih Karagümrük': 'Karagumruk',
        'Fatih Karagumruk': 'Karagumruk',
        'Adana Demirspor': 'Adana Demir',
        '1461 Trabzon': 'Trabzon',
        'Trabzonspor': 'Trabzon',
        'Trabzon FK': 'Trabzon',
        'Galatasaray SK': 'Galatasaray',
        'Galatasaray AS': 'Galatasaray',
        'Fenerbahce SK': 'Fenerbahce',
        'Fenerbahçe': 'Fenerbahce',
        'Besiktas JK': 'Besiktas',
        'Beşiktaş': 'Besiktas',
        'Konyaspor': 'Konya',
        'Kasimpasa': 'Kasimpasa',
        'Kasımpaşa': 'Kasimpasa',
        'Antalyaspor': 'Antalya',
        'Alanyaspor': 'Alanya',
        'Samsunspor': 'Samsun',
        'Kayserispor': 'Kayseri',
        'Gaziantep FK': 'Gaziantep',
        'Sivasspor': 'Sivas',
        'Eyupspor': 'Eyup',
        'Eyüpspor': 'Eyup',
        'Rizespor': 'Rize',
        'Çaykur Rizespor': 'Rize',
        'Hatayspor': 'Hatay',
        'Bodrum FK': 'Bodrum',
        'Bodrumspor': 'Bodrum',
    }
    
    if name in exact_replacements:
        return exact_replacements[name]
    
    # ADIM 2: Regex temizleme (FC, SK, vs. eklerini kaldır)
    regex_patterns = [
        (r'\s+FC$', ''),
        (r'\s+SK$', ''),
        (r'\s+CF$', ''),
        (r'\s+AC$', ''),
        (r'\s+AS$', ''),
        (r'\s+JK$', ''),
        (r'\s+FK$', ''),
        (r'^FC\s+', ''),
        (r'^AC\s+', ''),
        (r'^AS\s+', ''),
        (r'\s+Kulübü$', ''),
        (r'\s+Spor$', ''),
    ]
    
    for pattern, replacement in regex_patterns:
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    
    # ADIM 3: Özel karakterleri temizle
    char_replacements = {
        'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I',
        'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C',
        'é': 'e', 'á': 'a', 'í': 'i', 'ó': 'o', 'ú': 'u',
    }
    
    for old_char, new_char in char_replacements.items():
        name = name.replace(old_char, new_char)
    
    # ADIM 4: Ekstra boşlukları temizle
    name = ' '.join(name.split())
    
    # ADIM 5: Sayıları baştan kaldır (1461 Trabzon → Trabzon)
    name = re.sub(r'^\d+\s+', '', name)
    
    return name

print("\n[3/5] Takim isimleri normalize ediliyor...")

# Önceki durumu kaydet
before_home = df['home_team'].nunique()
before_away = df['away_team'].nunique()

# Normalize et
df['home_team'] = df['home_team'].apply(normalize_team_name)
df['away_team'] = df['away_team'].apply(normalize_team_name)

# Sonraki durumu kontrol et
after_home = df['home_team'].nunique()
after_away = df['away_team'].nunique()

print(f"   Onceki unique ev takimi: {before_home}")
print(f"   Sonraki unique ev takimi: {after_home}")
print(f"   Birlesme: {before_home - after_home} takim")

print("\n[4/5] Değişiklikler kontrol ediliyor...")

# En çok etkilenen takımlar
original_df = pd.read_csv('football_match_data_YEDEK.csv', low_memory=False)
changes = {}  # {old_name: new_name}
change_count = {}  # {old_name: count}

# Tüm satırları kontrol et
for idx in range(len(df)):
    old_home = str(original_df.loc[idx, 'home_team'])
    new_home = str(df.loc[idx, 'home_team'])
    old_away = str(original_df.loc[idx, 'away_team'])
    new_away = str(df.loc[idx, 'away_team'])
    
    if old_home != new_home:
        changes[old_home] = new_home
        change_count[old_home] = change_count.get(old_home, 0) + 1
    
    if old_away != new_away:
        changes[old_away] = new_away
        change_count[old_away] = change_count.get(old_away, 0) + 1

if changes:
    print(f"\n   ✓ Toplam {len(changes)} farkli takim degisti")
    print("\n   DEGISIKLIK DETAYLARI:")
    print("   " + "=" * 70)
    
    # Değişiklik sayısına göre sırala
    sorted_changes = sorted(change_count.items(), key=lambda x: x[1], reverse=True)
    
    for old_name, count in sorted_changes:
        new_name = changes[old_name]
        print(f"   [{count:4d} mac] {old_name:35s} → {new_name}")
    
    print("   " + "=" * 70)
else:
    print("   ℹ Hicbir degisiklik bulunamadi!")

print("\n[5/5] CSV kaydediliyor...")
df.to_csv('football_match_data.csv', index=False)
print("   ✓ Kaydedildi: football_match_data.csv")

print("\n" + "=" * 80)
print("NORMALİZASYON TAMAMLANDI!")
print("=" * 80)
print(f"\nToplam {before_home + before_away - after_home - after_away} takim birlesti")
print("\nŞimdi modeli yeniden eğitmelisiniz:")
print("   python train_enhance_v2.py")
print("\nYedek dosya: football_match_data_YEDEK.csv")
print("=" * 80)

