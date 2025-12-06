"""
JSON Veri İmport Scripti
Kullanım: python import_json.py yeni_maclar.json
"""

import pandas as pd
import json
import sys
from datetime import datetime

def import_json_to_csv(json_file):
    """
    JSON dosyasından maçları okur ve CSV'ye ekler
    
    JSON Formatı:
    [
        {
            "date": "2024-12-15",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_goals": 2,
            "away_goals": 1,
            "home_xG": 1.8,
            "away_xG": 1.2,
            "league": "EPL"
        },
        ...
    ]
    """
    
    print("=" * 80)
    print("JSON VERİ İMPORT SİSTEMİ")
    print("=" * 80)
    
    # JSON'u yükle
    print(f"\n[1/5] JSON dosyası okunuyor: {json_file}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            new_matches = json.load(f)
        print(f"   ✓ {len(new_matches)} yeni mac bulundu")
    except FileNotFoundError:
        print(f"   ✗ Dosya bulunamadi: {json_file}")
        return
    except json.JSONDecodeError as e:
        print(f"   ✗ JSON formatı hatali: {e}")
        return
    
    # Mevcut CSV'yi yükle
    print("\n[2/5] Mevcut CSV yukleniyor...")
    try:
        df_existing = pd.read_csv('football_match_data.csv', low_memory=False)
        print(f"   ✓ Mevcut {len(df_existing)} mac yuklendi")
    except FileNotFoundError:
        print("   ℹ CSV bulunamadi, yeni olusturulacak")
        df_existing = pd.DataFrame()
    
    # Yedek oluştur
    print("\n[3/5] Yedek olusturuluyor...")
    backup_file = f'football_match_data_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    if not df_existing.empty:
        df_existing.to_csv(backup_file, index=False)
        print(f"   ✓ Yedek: {backup_file}")
    
    # JSON'u DataFrame'e çevir
    print("\n[4/5] Yeni maclar isleniyor...")
    df_new = pd.DataFrame(new_matches)
    
    # Tarih formatını düzelt
    if 'date' in df_new.columns:
        df_new['date'] = pd.to_datetime(df_new['date'])
    
    # Eksik kolonları varsayılan değerlerle doldur
    default_values = {
        'home_support': 0.5,
        'away_support': 0.5,
        'sentiment_score': 0.0,
        'total_tweets': 0,
        'odds_b365_h': None,
        'odds_b365_d': None,
        'odds_b365_a': None,
    }
    
    for col, default in default_values.items():
        if col not in df_new.columns:
            df_new[col] = default
    
    # Duplikasyon kontrolü
    if not df_existing.empty and 'date' in df_existing.columns:
        df_existing['date'] = pd.to_datetime(df_existing['date'], errors='coerce')
        
        # Aynı tarih, ev takımı ve deplasman takımı olan maçları bul
        merge_cols = ['date', 'home_team', 'away_team']
        existing_matches = set(
            tuple(x) for x in df_existing[merge_cols].dropna().values
        )
        
        duplicates = []
        for idx, row in df_new.iterrows():
            match_key = (row['date'], row['home_team'], row['away_team'])
            if match_key in existing_matches:
                duplicates.append(idx)
        
        if duplicates:
            print(f"   ⚠ {len(duplicates)} duplikasyon bulundu, guncellenecek")
            # Duplikasyonları güncelle (eski verileri sil, yenileri ekle)
            for idx in duplicates:
                row = df_new.loc[idx]
                mask = (
                    (df_existing['date'] == row['date']) &
                    (df_existing['home_team'] == row['home_team']) &
                    (df_existing['away_team'] == row['away_team'])
                )
                df_existing = df_existing[~mask]
            print(f"   ✓ Eski veriler silindi, yenileri eklenecek")
    
    # Birleştir
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Tarihe göre sırala
    if 'date' in df_combined.columns:
        df_combined = df_combined.sort_values('date')
    
    print(f"   ✓ Toplam {len(df_combined)} mac (Eski: {len(df_existing)}, Yeni: {len(df_new)})")
    
    # Kaydet
    print("\n[5/5] CSV kaydediliyor...")
    df_combined.to_csv('football_match_data.csv', index=False)
    print("   ✓ Kaydedildi: football_match_data.csv")
    
    # Özet
    print("\n" + "=" * 80)
    print("İMPORT TAMAMLANDI!")
    print("=" * 80)
    print(f"\nYeni eklenen maclar: {len(df_new)}")
    if 'date' in df_new.columns:
        print(f"Tarih araligi: {df_new['date'].min()} - {df_new['date'].max()}")
    
    print(f"\nToplam mac sayisi: {len(df_combined)}")
    print(f"Yedek dosya: {backup_file if not df_existing.empty else 'Yok'}")
    
    print("\n⚠️  ÖNEMLİ: Modeli yeniden eğitmeyi unutmayin!")
    print("   python train_enhance_v2.py")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python import_json.py <json_dosyasi>")
        print("\nJSON Formatı Örneği:")
        print("""
[
  {
    "date": "2024-12-15",
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "home_goals": 2,
    "away_goals": 1,
    "home_xG": 1.8,
    "away_xG": 1.2,
    "league": "EPL"
  }
]
        """)
        sys.exit(1)
    
    json_file = sys.argv[1]
    import_json_to_csv(json_file)





