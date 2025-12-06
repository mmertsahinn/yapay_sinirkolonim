"""
JSON'dan CSV'ye Komple Entegrasyon
Mevcut scrape verilerini bozmadan, JSON'daki tüm verileri CSV'ye ekler/günceller.
Eşleştirme yapmadan direkt JSON formatından CSV formatına çevirir ve birleştirir.
"""
import pandas as pd
import json
import numpy as np
from datetime import datetime
from pathlib import Path

def load_json_data(json_path):
    """JSON export dosyasını yükle"""
    print(f"Loading JSON data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matches = data.get('data', {}).get('matches', [])
    print(f"Loaded {len(matches)} matches from JSON")
    return matches

def json_to_csv_format(json_matches):
    """
    JSON'daki tüm maçları CSV formatına çevir
    """
    print(f"Converting {len(json_matches)} JSON matches to CSV format...")
    
    csv_rows = []
    
    for idx, match in enumerate(json_matches):
        if (idx + 1) % 5000 == 0:
            print(f"  Processing: {idx + 1}/{len(json_matches)} matches...")
        
        # Tarih parse et
        match_date = match.get('match_date', '')
        date_obj = None
        date_str = None
        time_str = None
        
        if match_date:
            try:
                if 'T' in match_date:
                    date_obj = pd.to_datetime(match_date)
                    date_str = date_obj.strftime('%Y-%m-%d')
                    time_str = date_obj.strftime('%H:%M:%S')
                else:
                    date_obj = pd.to_datetime(match_date)
                    date_str = date_obj.strftime('%Y-%m-%d')
                    time_str = '00:00:00'
            except:
                date_str = match_date[:10] if len(match_date) >= 10 else None
                time_str = '00:00:00'
        
        # Odds verilerini al
        odds = match.get('odds', {})
        if not odds:
            odds = {}
        
        # Liga ismini normalize et (CSV formatına uygun)
        league_name = match.get('league_name', '')
        league_mapping = {
            'Premier League': 'english_premier_league',
            'La Liga': 'spanish_la_liga',
            'Serie A': 'italian_serie_a',
            'Bundesliga': 'german_bundesliga',
            'Ligue 1': 'french_ligue_1',
            'Liga Portugal': 'portuguese_primeira_liga',
            'Turkey': 'turkish_super_lig',
            'Süper Lig': 'turkish_super_lig'
        }
        league_normalized = league_mapping.get(league_name, league_name.lower().replace(' ', '_') if league_name else None)
        
        # Sezon hesapla
        season = None
        if date_obj:
            year = date_obj.year
            month = date_obj.month
            if month >= 8:
                season = f"{year}-{year+1}"
            else:
                season = f"{year-1}-{year}"
        
        # CSV satırı oluştur - TÜM JSON verilerini içerir
        csv_row = {
            # Temel bilgiler (scrape'den gelenler)
            'date': date_str,
            'time': time_str,
            'league': league_normalized,
            'season': season,
            'home_team': match.get('home_team_name'),
            'away_team': match.get('away_team_name'),
            'home_goals': match.get('home_score'),
            'away_goals': match.get('away_score'),
            'home_xG': None,  # JSON'da yoksa None (scrape'den gelirse korunur)
            'away_xG': None,  # JSON'da yoksa None (scrape'den gelirse korunur)
            
            # Hype verileri (JSON'dan)
            'home_support': match.get('home_support'),
            'away_support': match.get('away_support'),
            'sentiment_score': match.get('sentiment_score'),
            'total_tweets': match.get('total_tweets'),
            
            # Odds verileri - TÜM OLANLAR (JSON'dan)
            'odds_b365_h': odds.get('b365_h'),
            'odds_b365_d': odds.get('b365_d'),
            'odds_b365_a': odds.get('b365_a'),
            'odds_b365_ch': odds.get('b365_ch'),
            'odds_b365_cd': odds.get('b365_cd'),
            'odds_b365_ca': odds.get('b365_ca'),
            'odds_max_h': odds.get('max_h'),
            'odds_max_d': odds.get('max_d'),
            'odds_max_a': odds.get('max_a'),
            'odds_avg_h': odds.get('avg_h'),
            'odds_avg_d': odds.get('avg_d'),
            'odds_avg_a': odds.get('avg_a'),
            'odds_b365_over_25': odds.get('b365_over_25'),
            'odds_b365_under_25': odds.get('b365_under_25'),
            'odds_max_over_25': odds.get('max_over_25'),
            'odds_max_under_25': odds.get('max_under_25'),
            'odds_avg_over_25': odds.get('avg_over_25'),
            'odds_avg_under_25': odds.get('avg_under_25'),
            'odds_ah_h': odds.get('ah_h'),
            'odds_b365_ah_h': odds.get('b365_ah_h'),
            'odds_b365_ah_a': odds.get('b365_ah_a'),
            'odds_bf_h': odds.get('bf_h'),
            'odds_bf_d': odds.get('bf_d'),
            'odds_bf_a': odds.get('bf_a'),
            'odds_p_h': odds.get('p_h'),
            'odds_p_d': odds.get('p_d'),
            'odds_p_a': odds.get('p_a'),
            'odds_json': json.dumps(odds) if odds else None
        }
        
        csv_rows.append(csv_row)
    
    return pd.DataFrame(csv_rows)

def integrate_json_to_csv(json_path, csv_path, output_path=None):
    """
    JSON'daki tüm verileri CSV'ye entegre et
    Mevcut scrape verilerini korur, JSON verilerini ekler/günceller
    """
    print("="*60)
    print("JSON'DAN CSV'YE KOMPLE ENTEGRASYON")
    print("="*60)
    
    # Mevcut CSV'yi yükle (eğer varsa)
    if Path(csv_path).exists():
        print(f"\n1. Loading existing CSV: {csv_path}")
        df_csv_existing = pd.read_csv(csv_path, low_memory=False)
        print(f"   Existing CSV: {len(df_csv_existing)} rows, {len(df_csv_existing.columns)} columns")
    else:
        print(f"\n1. CSV not found, creating new one: {csv_path}")
        df_csv_existing = pd.DataFrame()
    
    # JSON'u yükle ve CSV formatına çevir
    print(f"\n2. Loading and converting JSON: {json_path}")
    json_matches = load_json_data(json_path)
    df_json_csv = json_to_csv_format(json_matches)
    
    print(f"   JSON converted: {len(df_json_csv)} rows, {len(df_json_csv.columns)} columns")
    
    # Mevcut CSV boşsa direkt JSON'u kullan
    if df_csv_existing.empty:
        print("\n3. Existing CSV is empty, using JSON data directly")
        df_final = df_json_csv.copy()
    else:
        # Mevcut CSV ile JSON'u birleştir
        print("\n3. Merging existing CSV with JSON data...")
        
        # Tüm kolonları birleştir
        all_columns = sorted(set(df_csv_existing.columns.tolist() + df_json_csv.columns.tolist()))
        print(f"   Total unique columns: {len(all_columns)}")
        
        # Eksik kolonları ekle
        for col in all_columns:
            if col not in df_csv_existing.columns:
                df_csv_existing[col] = None
            if col not in df_json_csv.columns:
                df_json_csv[col] = None
        
        # Aynı kolon sırasına getir
        df_csv_existing = df_csv_existing[all_columns]
        df_json_csv = df_json_csv[all_columns]
        
        # Birleştirme stratejisi: Scrape verilerini koru, JSON verilerini ekle/güncelle
        print("   Merging strategy: Preserve scrape data, add/update JSON data...")
        
        # Mevcut CSV'deki verileri koru (scrape verileri)
        # JSON'daki verileri ekle veya mevcut olanları güncelle
        
        # Duplicate kontrolü için key oluştur
        def create_match_key(row):
            date = str(row.get('date', ''))[:10] if pd.notna(row.get('date')) else ''
            home = str(row.get('home_team', '')).strip() if pd.notna(row.get('home_team')) else ''
            away = str(row.get('away_team', '')).strip() if pd.notna(row.get('away_team')) else ''
            return f"{date}_{home}_{away}".lower()
        
        # Mevcut CSV'den match key'leri
        df_csv_existing['_match_key'] = df_csv_existing.apply(create_match_key, axis=1)
        df_json_csv['_match_key'] = df_json_csv.apply(create_match_key, axis=1)
        
        # JSON'daki key'ler
        json_keys = set(df_json_csv['_match_key'].unique())
        existing_keys = set(df_csv_existing['_match_key'].unique())
        
        # Yeni maçlar (sadece JSON'da var)
        new_matches = df_json_csv[~df_json_csv['_match_key'].isin(existing_keys)].copy()
        
        # Mevcut maçlar (hem CSV'de hem JSON'da var) - JSON ile güncelle
        existing_matches_in_json = df_json_csv[df_json_csv['_match_key'].isin(existing_keys)].copy()
        
        # CSV'deki mevcut maçları güncelle (JSON verileri ekle)
        # Önce JSON olmayan maçları al (scrape verileri korunur)
        df_csv_only = df_csv_existing[~df_csv_existing['_match_key'].isin(json_keys)].copy()
        
        # JSON verileri ile mevcut maçları güncelle
        # ÖNEMLİ: Scrape verilerini koru (home_xG, away_xG, vb.), JSON verilerini ekle (hype, odds)
        df_existing_updated = df_csv_existing[df_csv_existing['_match_key'].isin(json_keys)].copy()
        
        # JSON verilerini CSV verilerine ekle (scrape verilerini bozmadan)
        if not df_existing_updated.empty and not existing_matches_in_json.empty:
            # JSON'da olan ama scrape'de olmayan kolonlar (hype, odds)
            json_only_columns = [
                'home_support', 'away_support', 'sentiment_score', 'total_tweets',
                'odds_b365_h', 'odds_b365_d', 'odds_b365_a', 'odds_b365_ch', 'odds_b365_cd', 'odds_b365_ca',
                'odds_max_h', 'odds_max_d', 'odds_max_a', 'odds_avg_h', 'odds_avg_d', 'odds_avg_a',
                'odds_b365_over_25', 'odds_b365_under_25', 'odds_max_over_25', 'odds_max_under_25',
                'odds_avg_over_25', 'odds_avg_under_25', 'odds_ah_h', 'odds_b365_ah_h', 'odds_b365_ah_a',
                'odds_bf_h', 'odds_bf_d', 'odds_bf_a', 'odds_p_h', 'odds_p_d', 'odds_p_a', 'odds_json'
            ]
            
            for key in json_keys & existing_keys:
                csv_row_idx = df_existing_updated[df_existing_updated['_match_key'] == key].index[0]
                json_row = existing_matches_in_json[existing_matches_in_json['_match_key'] == key].iloc[0]
                
                # Sadece JSON kolonlarını ekle (scrape verilerini koru)
                for col in json_only_columns:
                    if col in json_row.index and col in df_existing_updated.columns:
                        if pd.notna(json_row[col]):
                            df_existing_updated.at[csv_row_idx, col] = json_row[col]
        
        # Birleştir: Scrape-only + Updated + JSON-only
        df_final = pd.concat([df_csv_only, df_existing_updated, new_matches], ignore_index=True)
        
        # Match key kolonunu kaldır
        if '_match_key' in df_final.columns:
            df_final = df_final.drop(columns=['_match_key'])
        
        print(f"   Scrape-only matches (preserved): {len(df_csv_only)}")
        print(f"   Matches updated with JSON data: {len(df_existing_updated)}")
        print(f"   New matches from JSON: {len(new_matches)}")
        print(f"   Total after merge: {len(df_final)} rows")
    
    # Tarihe göre sırala
    if 'date' in df_final.columns:
        df_final['date'] = pd.to_datetime(df_final['date'], errors='coerce')
        df_final = df_final.sort_values('date')
        df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')
    
    # İstatistikler
    print("\n" + "="*60)
    print("ENTEGRASYON ISTATISTIKLERI")
    print("="*60)
    print(f"Toplam mac: {len(df_final):,}")
    print(f"Hype verisi olan: {df_final['home_support'].notna().sum():,} ({df_final['home_support'].notna().sum()/len(df_final)*100:.1f}%)")
    print(f"Odds verisi olan: {df_final['odds_b365_h'].notna().sum():,} ({df_final['odds_b365_h'].notna().sum()/len(df_final)*100:.1f}%)")
    print(f"Hem hype hem odds: {(df_final['home_support'].notna() & df_final['odds_b365_h'].notna()).sum():,}")
    print("="*60)
    
    # Kaydet
    output_file = output_path or csv_path
    print(f"\n4. Saving to: {output_file}")
    df_final.to_csv(output_file, index=False)
    print(f"[OK] CSV dosyasi kaydedildi: {output_file}")
    
    return df_final

if __name__ == "__main__":
    # Dosya yolları
    script_dir = Path(__file__).parent
    json_path = script_dir.parent / 'football_brain_core' / 'football_brain_export.json'
    csv_path = script_dir / 'football_match_data.csv'
    
    # Entegrasyon yap
    df = integrate_json_to_csv(json_path, csv_path)
    
    print("\n" + "="*60)
    print("TAMAMLANDI!")
    print("="*60)
    print(f"Final CSV: {csv_path}")
    print(f"Toplam satır: {len(df):,}")
    print(f"Toplam kolon: {len(df.columns)}")
    print("\n[OK] JSON'daki tum veriler CSV'ye aktarildi!")
    print("[OK] Mevcut scrape verileri korundu!")

