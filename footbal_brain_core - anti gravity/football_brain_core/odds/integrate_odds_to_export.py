"""
Odds CSV dosyalarını okuyup football_brain_export.json dosyasına entegre eden script.
Tüm liglerden odds verilerini toplayıp JSON formatına çevirir ve mevcut export dosyasına ekler.
"""
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict

# Lig klasör isimlerinden lig isimlerine mapping
LEAGUE_MAPPING = {
    'england': 'England',
    'espana': 'Spain', 
    'italy': 'Italy',
    'bundesliga': 'Germany',
    'france': 'France',
    'portugal': 'Portugal',
    'turkey': 'Turkey'
}

# CSV kolonlarından JSON alanlarına mapping
ODDS_COLUMN_MAPPING = {
    # Bet365
    'B365H': 'b365_h',
    'B365D': 'b365_d', 
    'B365A': 'b365_a',
    'B365CH': 'b365_ch',
    'B365CD': 'b365_cd',
    'B365CA': 'b365_ca',
    
    # Betfair
    'BFH': 'bf_h',
    'BFD': 'bf_d',
    'BFA': 'bf_a',
    
    # Betfred
    'BFDH': 'bfd_h',
    'BFDD': 'bfd_d',
    'BFDA': 'bfd_a',
    'BFDCH': 'bfd_ch',
    'BFDCD': 'bfd_cd',
    'BFDCA': 'bfd_ca',
    
    # BetMGM
    'BMGMH': 'bmgm_h',
    'BMGMD': 'bmgm_d',
    'BMGMA': 'bmgm_a',
    'BMGMCH': 'bmgm_ch',
    'BMGMCD': 'bmgm_cd',
    'BMGMCA': 'bmgm_ca',
    
    # Betvictor
    'BVH': 'bv_h',
    'BVD': 'bv_d',
    'BVA': 'bv_a',
    'BVCH': 'bv_ch',
    'BVCD': 'bv_cd',
    'BVCA': 'bv_ca',
    
    # Coral
    'CLH': 'cl_h',
    'CLD': 'cl_d',
    'CLA': 'cl_a',
    'CLCH': 'cl_ch',
    'CLCD': 'cl_cd',
    'CLCA': 'cl_ca',
    
    # Ladbrokes
    'LBH': 'lb_h',
    'LBD': 'lb_d',
    'LBA': 'lb_a',
    'LBCH': 'lb_ch',
    'LBCD': 'lb_cd',
    'LBCA': 'lb_ca',
    
    # Pinnacle
    'PSH': 'p_h',
    'PSD': 'p_d',
    'PSA': 'p_a',
    'PSCH': 'p_ch',
    'PSCD': 'p_cd',
    'PSCA': 'p_ca',
    
    # Market averages
    'MaxH': 'max_h',
    'MaxD': 'max_d',
    'MaxA': 'max_a',
    'AvgH': 'avg_h',
    'AvgD': 'avg_d',
    'AvgA': 'avg_a',
    
    # Over/Under
    'B365>2.5': 'b365_over_25',
    'B365<2.5': 'b365_under_25',
    'P>2.5': 'p_over_25',
    'P<2.5': 'p_under_25',
    'Max>2.5': 'max_over_25',
    'Max<2.5': 'max_under_25',
    'Avg>2.5': 'avg_over_25',
    'Avg<2.5': 'avg_under_25',
    
    # Asian Handicap
    'AHh': 'ah_h',
    'B365AHH': 'b365_ah_h',
    'B365AHA': 'b365_ah_a',
    'PAHH': 'p_ah_h',
    'PAHA': 'p_ah_a',
}

def parse_date(date_str, time_str=None):
    """CSV'deki tarih formatını parse et"""
    try:
        # Format: dd/mm/yy
        if pd.isna(date_str) or date_str == '':
            return None
        
        date_str = str(date_str).strip()
        parts = date_str.split('/')
        if len(parts) == 3:
            day, month, year = parts
            # Yıl iki haneli ise 2000-2099 arası varsay
            year = int(year)
            if year < 50:
                year += 2000
            elif year < 100:
                year += 1900
            
            date_obj = datetime(int(year), int(month), int(day))
            
            # Saat bilgisi varsa ekle
            if time_str and pd.notna(time_str):
                try:
                    time_parts = str(time_str).split(':')
                    if len(time_parts) >= 2:
                        hour = int(time_parts[0])
                        minute = int(time_parts[1])
                        date_obj = date_obj.replace(hour=hour, minute=minute)
                except:
                    pass
            
            return date_obj.isoformat()
    except Exception as e:
        print(f"Tarih parse hatasi: {date_str}, Hata: {e}")
        return None
    return None

def normalize_team_name(name):
    """Takım ismini normalize et"""
    if pd.isna(name) or name == '':
        return None
    return str(name).strip()

def convert_odds_value(value):
    """Odds değerini float'a çevir"""
    if pd.isna(value) or value == '':
        return None
    try:
        return float(value)
    except:
        return None

def read_all_csv_files(odds_dir):
    """Tüm CSV dosyalarını oku ve birleştir"""
    all_matches = []
    failed_files = []
    processed_files = []
    
    league_dirs = [d for d in os.listdir(odds_dir) if os.path.isdir(os.path.join(odds_dir, d)) and d in LEAGUE_MAPPING]
    
    print(f"Bulunan lig klasorleri: {len(league_dirs)}")
    
    for league_dir in league_dirs:
        league_path = os.path.join(odds_dir, league_dir)
        league_name = LEAGUE_MAPPING[league_dir]
        
        print(f"\n{league_name} isleniyor...")
        
        # Klasördeki tüm CSV dosyalarını bul
        csv_files = [f for f in os.listdir(league_path) if f.endswith('.csv') and f != 'notes.TXT']
        
        for csv_file in csv_files:
            csv_path = os.path.join(league_path, csv_file)
            
            try:
                # Farklı encoding'leri dene
                df = None
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig', 'windows-1252']
                encoding_used = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(csv_path, encoding=encoding, low_memory=False, on_bad_lines='skip')
                        if df is not None and len(df) > 0:
                            encoding_used = encoding
                            break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        # Diğer hataları görmezden gel, sonraki encoding'i dene
                        continue
                
                if df is None or len(df) == 0:
                    print(f"  UYARI: {csv_file} okunamadi veya bos (tum encoding'ler denendi)")
                    continue
                
                if encoding_used and encoding_used != 'utf-8':
                    print(f"    ({encoding_used} encoding kullanildi)")
                
                # Div kolonu varsa lig seviyesini al
                division = None
                if 'Div' in df.columns:
                    division = df['Div'].iloc[0] if len(df) > 0 else None
                
                print(f"  {csv_file}: {len(df)} maç")
                
                # Her satırı işle
                processed_count = 0
                for idx, row in df.iterrows():
                    match_data = process_match_row(row, league_name, division)
                    if match_data:
                        all_matches.append(match_data)
                        processed_count += 1
                
                if processed_count == 0:
                    print(f"    UYARI: {csv_file} icinden hic maç islenemedi!")
                    failed_files.append(f"{league_name}/{csv_file} (0 maç islendi)")
                else:
                    processed_files.append(f"{league_name}/{csv_file} ({processed_count} maç)")
                        
            except UnicodeDecodeError as e:
                error_msg = f"{league_name}/{csv_file} - Encoding hatasi: {str(e)[:100]}"
                print(f"  HATA: {error_msg}")
                failed_files.append(error_msg)
                continue
            except Exception as e:
                error_msg = f"{league_name}/{csv_file} - {type(e).__name__}: {str(e)[:100]}"
                print(f"  HATA: {error_msg}")
                failed_files.append(error_msg)
                continue
    
    # Özet rapor
    print("\n" + "="*60)
    print("OZET RAPOR")
    print("="*60)
    print(f"Basarili: {len(processed_files)} dosya")
    print(f"Basarisiz: {len(failed_files)} dosya")
    
    if failed_files:
        print(f"\nOkunamayan dosyalar:")
        for failed in failed_files[:10]:  # İlk 10'unu göster
            print(f"  - {failed}")
        if len(failed_files) > 10:
            print(f"  ... ve {len(failed_files) - 10} dosya daha")
    
    return all_matches

def process_match_row(row, league_name, division):
    """Bir maç satırını işle ve JSON formatına çevir"""
    try:
        # Temel maç bilgileri
        date = parse_date(row.get('Date'), row.get('Time'))
        home_team = normalize_team_name(row.get('HomeTeam'))
        away_team = normalize_team_name(row.get('AwayTeam'))
        
        if not date or not home_team or not away_team:
            return None
        
        # Odds verilerini topla
        odds_data = {}
        
        # Tüm odds kolonlarını map et
        for csv_col, json_key in ODDS_COLUMN_MAPPING.items():
            # Özel karakterli kolonlar için esnek eşleştirme
            matching_col = None
            if csv_col in row.index:
                matching_col = csv_col
            else:
                # Benzer kolon ara (büyük/küçük harf farkı, boşluk vs)
                for col in row.index:
                    if col.upper().replace(' ', '').replace('_', '') == csv_col.upper().replace(' ', '').replace('_', ''):
                        matching_col = col
                        break
            
            if matching_col:
                value = convert_odds_value(row[matching_col])
                if value is not None and 0.1 <= value <= 100:  # Mantıklı odds değeri kontrolü
                    odds_data[json_key] = value
        
        # Tüm diğer odds kolonlarını all_odds'a ekle (özellikle özel karakterli olanlar)
        all_odds_dict = {}
        
        # İstatistik kolonlarını hariç tut
        exclude_cols = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                       'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
                       'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'HG', 'AG', 'Res', 'Attendance']
        
        for col in row.index:
            if col in exclude_cols or col in ODDS_COLUMN_MAPPING:
                continue
                
            if pd.notna(row[col]) and row[col] != '':
                try:
                    value = float(row[col])
                    # Sadece mantıklı odds değerleri (0.1 ile 100 arası)
                    if 0.1 <= value <= 100:
                        all_odds_dict[col] = value
                except:
                    pass
        
        if all_odds_dict:
            odds_data['all_odds'] = all_odds_dict
        
        # Maç sonuçları
        home_goals = None
        away_goals = None
        if 'FTHG' in row.index and pd.notna(row['FTHG']):
            try:
                home_goals = int(float(row['FTHG']))
            except:
                pass
        elif 'HG' in row.index and pd.notna(row['HG']):
            try:
                home_goals = int(float(row['HG']))
            except:
                pass
        
        if 'FTAG' in row.index and pd.notna(row['FTAG']):
            try:
                away_goals = int(float(row['FTAG']))
            except:
                pass
        elif 'AG' in row.index and pd.notna(row['AG']):
            try:
                away_goals = int(float(row['AG']))
            except:
                pass
        
        # Match ID oluştur (league_date_home_away formatında)
        match_id = f"{league_name}_{date[:10]}_{home_team}_{away_team}".replace(' ', '_')
        
        # Odds data'yı düzenle - all_odds içindeki değerleri de ana seviyeye çıkar
        # Sadece önemli olanları ana seviyede tut, diğerleri all_odds içinde kalsın
        return {
            'match_id': match_id,
            'league': league_name,
            'division': division,
            'date': date,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'odds': odds_data  # odds_data içinde tüm odds değerleri var
        }
        
    except Exception as e:
        print(f"  Satir isleme hatasi: {str(e)}")
        return None

def normalize_team_name_for_match(team_name):
    """Takım ismini match_id formatına uygun hale getir - GELİŞTİRİLMİŞ"""
    if not team_name:
        return ""
    
    name = str(team_name).strip()
    
    # Yaygın varyasyonları düzelt
    replacements = {
        " FC": "",
        " AFC": "",
        " United": " Utd",
        " City": "",
        " Town": "",
        " Rovers": "",
        " Wanderers": "",
        " Athletic": "",
        " Albion": "",
        " Hotspur": "",
        " & ": " ",
        " and ": " ",
        "Olympique ": "Olympique ",
        "Real ": "Real ",
        "Atletico ": "Atletico ",
        "Atlético ": "Atletico ",
        "Deportivo ": "Deportivo ",
        "Sporting ": "Sporting ",
        "Sport ": "Sport ",
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Özel karakterleri temizle
    name = name.replace("'", "").replace("-", " ").replace(".", "")
    
    # Boşlukları normalize et
    name = " ".join(name.split())
    
    return name

def normalize_team_name_fuzzy(team_name):
    """Fuzzy matching için daha agresif normalizasyon"""
    if not team_name:
        return ""
    
    name = str(team_name).strip().lower()
    
    # Tüm özel karakterleri kaldır
    name = ''.join(c for c in name if c.isalnum() or c.isspace())
    
    # Yaygın kelimeleri kaldır
    stop_words = ['fc', 'afc', 'cf', 'cfc', 'united', 'city', 'town', 'rovers', 
                  'wanderers', 'athletic', 'albion', 'hotspur', 'club', 'team']
    words = name.split()
    words = [w for w in words if w not in stop_words]
    name = ' '.join(words)
    
    return name.strip()

def team_name_similarity(name1, name2):
    """İki takım ismi arasındaki benzerlik skoru (0-1)"""
    if not name1 or not name2:
        return 0.0
    
    # Normalize et
    n1 = normalize_team_name_fuzzy(name1)
    n2 = normalize_team_name_fuzzy(name2)
    
    if not n1 or not n2:
        return 0.0
    
    # SequenceMatcher kullan
    similarity = SequenceMatcher(None, n1, n2).ratio()
    
    # Ek kontrol: bir isim diğerinin içinde mi?
    if n1 in n2 or n2 in n1:
        similarity = max(similarity, 0.8)
    
    return similarity

def create_match_lookup_key(match):
    """Match objesinden lookup key oluştur"""
    match_id = match.get('match_id', '')
    if match_id:
        return match_id
    
    # match_id yoksa oluştur
    league_id = match.get('league_id')
    home_team_id = match.get('home_team_id')
    away_team_id = match.get('away_team_id')
    match_date = match.get('match_date', '')
    
    if league_id and home_team_id and away_team_id and match_date:
        # Format: league_id_year_home_id_away_id_date
        date_part = match_date[:10].replace('-', '')
        year = date_part[:4]
        return f"{league_id}_{year}_{home_team_id}_{away_team_id}_{date_part}"
    
    return None

def create_odds_lookup_key(odds_item):
    """Odds item'ından lookup key oluştur - matches ile eşleştirmek için"""
    # Önce match_id varsa onu kullan
    match_id = odds_item.get('match_id')
    if match_id:
        return match_id
    
    # Yoksa maç bilgilerinden oluştur
    league_name = odds_item.get('league', '')
    date = odds_item.get('date', '')
    home_team = normalize_team_name_for_match(odds_item.get('home_team', ''))
    away_team = normalize_team_name_for_match(odds_item.get('away_team', ''))
    
    if date and home_team and away_team:
        date_part = date[:10]
        return f"{league_name}_{date_part}_{home_team}_{away_team}".replace(' ', '_')
    
    return None

def integrate_to_export(odds_data, export_path):
    """Odds verilerini export JSON'a entegre et - HER MAÇ İÇİNE DİREKT EKLENİR"""
    
    # Path objesi ise string'e çevir
    export_path_str = str(export_path)
    print(f"\nExport dosyasi yukleniyor: {export_path_str}")
    
    # Mevcut export dosyasını yükle
    try:
        with open(export_path_str, 'r', encoding='utf-8') as f:
            export_data = json.load(f)
    except FileNotFoundError:
        print(f"Export dosyasi bulunamadi! Yeni dosya olusturuluyor...")
        export_data = {
            'export_date': datetime.now().isoformat(),
            'summary': {},
            'data': {
                'matches': []
            }
        }
    
    # Matches array'ini kontrol et
    if 'data' not in export_data:
        export_data['data'] = {}
    
    if 'matches' not in export_data['data']:
        print("UYARI: Export dosyasinda 'matches' array'i bulunamadi!")
        export_data['data']['matches'] = []
    
    matches = export_data['data']['matches']
    print(f"Mevcut maç sayisi: {len(matches)}")
    
    # Odds verilerini lookup dictionary'ye çevir - GELİŞTİRİLMİŞ
    odds_lookup = {}  # Ana lookup
    odds_by_date = defaultdict(list)  # Tarih bazlı index (fuzzy matching için hızlı erişim)
    
    for odds_item in odds_data:
        key = create_odds_lookup_key(odds_item)
        if key:
            # Birden fazla odds aynı maç için olabilir - birleştir
            if key in odds_lookup:
                odds_lookup[key]['odds'].update(odds_item.get('odds', {}))
            else:
                odds_lookup[key] = odds_item
            
            # Tarih bazlı index ekle (fuzzy matching için)
            odds_date = odds_item.get('date', '')
            if odds_date:
                try:
                    date_obj = datetime.fromisoformat(odds_date.replace('Z', '+00:00'))
                    date_key = date_obj.date().isoformat()
                    odds_by_date[date_key].append(odds_item)
                    # ±3 gün için de ekle (fuzzy matching için)
                    for day_offset in [-3, -2, -1, 1, 2, 3]:
                        alt_date = date_obj + timedelta(days=day_offset)
                        alt_date_key = alt_date.date().isoformat()
                        odds_by_date[alt_date_key].append(odds_item)
                except:
                    # Tarih parse edilemezse sadece ana lookup'a ekle
                    pass
    
    print(f"Odds verisi: {len(odds_lookup)} benzersiz maç")
    print(f"Tarih bazlı index: {len(odds_by_date)} farklı tarih")
    
    # Her maça odds verilerini ekle
    matches_with_odds = 0
    matches_updated = 0
    matches_skipped = 0  # Zaten odds'u olan maçlar (atlandı)
    matches_not_found = 0  # Odds bulunamayan maçlar
    
    total_matches = len(matches)
    
    print(f"\n{'='*60}")
    print(f"ODDS ENTEGRASYONU BASLIYOR")
    print(f"{'='*60}")
    print(f"Toplam maç sayısı: {total_matches}")
    print(f"Toplam odds verisi: {len(odds_lookup)}")
    print(f"{'='*60}\n")
    
    for idx, match in enumerate(matches):
        if (idx + 1) % 500 == 0:
            progress = (idx + 1) / total_matches * 100
            print(f"  İşleniyor: {idx + 1}/{total_matches} maç ({progress:.1f}%)...")
            print(f"    - Yeni odds eklenen: {matches_with_odds}")
            print(f"    - Güncellenen: {matches_updated}")
            print(f"    - Zaten odds'u var (atlandı): {matches_skipped}")
            print(f"    - Odds bulunamayan: {matches_not_found}")
        # Match için lookup key oluştur
        match_key = create_match_lookup_key(match)
        
        # Takım isimlerinden de eşleştirme yap
        home_team_name = normalize_team_name_for_match(match.get('home_team_name', ''))
        away_team_name = normalize_team_name_for_match(match.get('away_team_name', ''))
        match_date = match.get('match_date', '')
        league_name = match.get('league_name', '')
        
        # GELİŞTİRİLMİŞ EŞLEŞTİRME ALGORİTMASI
        matched_odds = None
        match_confidence = 0.0  # Eşleşme güven skoru
        
        # Tarih parse et (tolerans için)
        match_date_obj = None
        match_date_only = None
        if match_date:
            try:
                match_date_obj = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                match_date_only = match_date[:10]
            except:
                try:
                    match_date_obj = datetime.strptime(match_date[:10], '%Y-%m-%d')
                    match_date_only = match_date[:10]
                except:
                    match_date_only = match_date[:10] if len(match_date) >= 10 else None
        
        # Alternatif key'ler oluştur (daha fazla format)
        alt_keys = []
        if match_date_only and home_team_name and away_team_name:
            # Format 1: league_date_home_away
            alt_keys.append(f"{league_name}_{match_date_only}_{home_team_name}_{away_team_name}".replace(' ', '_'))
            # Format 2: league_date_away_home (ters sıra)
            alt_keys.append(f"{league_name}_{match_date_only}_{away_team_name}_{home_team_name}".replace(' ', '_'))
            # Format 3: Sadece tarih ve takımlar (lig olmadan)
            alt_keys.append(f"{match_date_only}_{home_team_name}_{away_team_name}".replace(' ', '_'))
            alt_keys.append(f"{match_date_only}_{away_team_name}_{home_team_name}".replace(' ', '_'))
            
            # League ID ile de dene
            league_id = match.get('league_id')
            if league_id:
                year = match_date_only[:4] if len(match_date_only) >= 4 else ''
                home_id = match.get('home_team_id')
                away_id = match.get('away_team_id')
                if year and home_id and away_id:
                    date_no_dash = match_date_only.replace('-', '')
                    alt_keys.append(f"{league_id}_{year}_{home_id}_{away_id}_{date_no_dash}")
                    alt_keys.append(f"{league_id}_{year}_{away_id}_{home_id}_{date_no_dash}")  # Ters sıra
        
        # 1. EXACT KEY MATCH (En güvenilir)
        match_key = create_match_lookup_key(match)
        if match_key and match_key in odds_lookup:
            matched_odds = odds_lookup[match_key]
            match_confidence = 1.0
        
        # 2. ALTERNATİF KEY'LERLE EŞLEŞTİRME
        if not matched_odds:
            for alt_key in alt_keys:
                if alt_key in odds_lookup:
                    matched_odds = odds_lookup[alt_key]
                    match_confidence = 0.9
                    break
        
        # 3. FUZZY MATCHING - Tarih toleransı ile (±1 gün) - OPTİMİZE EDİLMİŞ
        if not matched_odds and match_date_obj and home_team_name and away_team_name:
            best_match = None
            best_score = 0.0
            
            # Sadece ilgili tarihlerdeki odds'ları kontrol et (çok daha hızlı!)
            date_key = match_date_obj.date().isoformat()
            candidate_odds = []
            
            # İlgili tarihlerdeki tüm odds'ları topla
            for check_date in [date_key]:
                if check_date in odds_by_date:
                    candidate_odds.extend(odds_by_date[check_date])
            
            # ±2 gün kontrolü (daha geniş tarih aralığı)
            for day_offset in [-2, -1, 0, 1, 2]:
                check_date_obj = match_date_obj + timedelta(days=day_offset)
                check_date_key = check_date_obj.date().isoformat()
                if check_date_key in odds_by_date:
                    candidate_odds.extend(odds_by_date[check_date_key])
            
            # Candidate'ları unique yap
            seen = set()
            unique_candidates = []
            for item in candidate_odds:
                item_id = id(item)
                if item_id not in seen:
                    seen.add(item_id)
                    unique_candidates.append(item)
            
            # Her candidate için skor hesapla
            for odds_item in unique_candidates:
                odds_date_str = odds_item.get('date', '')
                if not odds_date_str:
                    continue
                
                # Tarih parse et
                try:
                    odds_date_obj = datetime.fromisoformat(odds_date_str.replace('Z', '+00:00'))
                except:
                    try:
                        odds_date_obj = datetime.strptime(odds_date_str[:10], '%Y-%m-%d')
                    except:
                        continue
                
                # Tarih toleransı: ±3 gün (maksimum esneklik)
                date_diff = abs((match_date_obj.date() - odds_date_obj.date()).days)
                if date_diff > 3:
                    continue
                
                # Takım isimleri benzerlik kontrolü
                odds_home = odds_item.get('home_team', '')
                odds_away = odds_item.get('away_team', '')
                odds_league = odds_item.get('league', '')
                
                # Takım ismi benzerlik skorları
                home_sim = team_name_similarity(home_team_name, odds_home)
                away_sim = team_name_similarity(away_team_name, odds_away)
                
                # Ters sıra kontrolü (home-away yerine away-home olabilir)
                home_sim_reverse = team_name_similarity(home_team_name, odds_away)
                away_sim_reverse = team_name_similarity(away_team_name, odds_home)
                
                # Normal sıra skoru
                normal_score = (home_sim + away_sim) / 2
                # Ters sıra skoru
                reverse_score = (home_sim_reverse + away_sim_reverse) / 2
                
                # En iyi skoru al
                team_score = max(normal_score, reverse_score)
                
                # Lig eşleşme kontrolü - Daha esnek (lig farkı çok kritik değil)
                league_score = 0.6  # Default daha yüksek
                if league_name and odds_league:
                    league_norm1 = normalize_team_name_fuzzy(league_name)
                    league_norm2 = normalize_team_name_fuzzy(odds_league)
                    if league_norm1 == league_norm2:
                        league_score = 1.0
                    elif league_norm1 in league_norm2 or league_norm2 in league_norm1:
                        league_score = 0.9  # Daha yüksek
                    else:
                        similarity = SequenceMatcher(None, league_norm1, league_norm2).ratio()
                        league_score = max(0.5, similarity)  # Minimum 0.5
                
                # Tarih skoru (aynı gün = 1.0, 1 gün = 0.9, 2 gün = 0.8, 3 gün = 0.7)
                if date_diff == 0:
                    date_score = 1.0
                elif date_diff == 1:
                    date_score = 0.9
                elif date_diff == 2:
                    date_score = 0.8
                else:
                    date_score = 0.7
                
                # Toplam skor - Takım isimlerine daha fazla ağırlık ver
                total_score = (team_score * 0.7 + league_score * 0.15 + date_score * 0.15)
                
                # Minimum eşik: 0.65 (daha da esnek - maksimum eşleşme için)
                # Özel durumlar için esnek kurallar
                if total_score > best_score:
                    if total_score >= 0.65:
                        best_score = total_score
                        best_match = odds_item
                    elif team_score >= 0.75 and date_score >= 0.9:  # Takım isimleri benzer ve tarih aynı
                        best_score = total_score
                        best_match = odds_item
                    elif team_score >= 0.85:  # Takım isimleri çok benzer, diğer faktörler önemli değil
                        best_score = total_score
                        best_match = odds_item
            
            if best_match:
                matched_odds = best_match
                match_confidence = best_score
        
        # Odds varsa ekle veya güncelle
        if matched_odds:
            odds_to_add = matched_odds.get('odds', {})
            if odds_to_add:
                # Eğer maçın zaten odds'u varsa kontrol et
                if 'odds' in match and match.get('odds'):
                    existing_odds = match.get('odds', {})
                    # Temel odds kontrolü - eğer b365_h varsa ve aynıysa atla
                    if 'b365_h' in existing_odds and 'b365_h' in odds_to_add:
                        if existing_odds.get('b365_h') == odds_to_add.get('b365_h'):
                            matches_skipped += 1
                            continue  # Aynı odds var, atla
                    
                    # Mevcut odds varsa birleştir (farklı odds ekleniyor)
                    match['odds'].update(odds_to_add)
                    matches_updated += 1
                else:
                    # Yeni odds ekle
                    match['odds'] = odds_to_add
                    matches_with_odds += 1
            else:
                matches_not_found += 1
        else:
            matches_not_found += 1
    
    # Summary güncelle
    if 'summary' not in export_data:
        export_data['summary'] = {}
    
    # Summary güncelle
    total_matches_with_odds = sum(1 for m in matches if 'odds' in m and m.get('odds'))
    
    if 'summary' not in export_data:
        export_data['summary'] = {}
    
    # Eşleşme istatistikleri
    total_odds_available = len(odds_lookup)
    match_rate = (total_matches_with_odds / total_matches * 100) if total_matches > 0 else 0
    odds_usage_rate = (total_matches_with_odds / total_odds_available * 100) if total_odds_available > 0 else 0
    
    export_data['summary']['matches_with_odds'] = total_matches_with_odds
    export_data['summary']['matches_skipped'] = matches_skipped
    export_data['summary']['matches_not_found'] = matches_not_found
    export_data['summary']['total_odds_available'] = total_odds_available
    export_data['summary']['match_rate'] = match_rate
    export_data['summary']['odds_usage_rate'] = odds_usage_rate
    export_data['export_date'] = datetime.now().isoformat()
    
    print(f"\n{'='*60}")
    print(f"ODDS ENTEGRASYONU TAMAMLANDI")
    print(f"{'='*60}")
    print(f"OZET:")
    print(f"  - Toplam mac sayisi: {total_matches}")
    print(f"  - Toplam odds verisi: {total_odds_available}")
    print(f"  - YENI odds eklenen maclar: {matches_with_odds}")
    print(f"  - Guncellenen maclar (farkli odds eklendi): {matches_updated}")
    print(f"  - Zaten odds'u var (atlandi): {matches_skipped}")
    print(f"  - Odds bulunamayan maclar: {matches_not_found}")
    print(f"  - Toplam odds'u olan mac sayisi: {total_matches_with_odds} / {total_matches}")
    if total_matches > 0:
        odds_percentage = (total_matches_with_odds / total_matches) * 100
        missing_odds = total_matches - total_matches_with_odds
        print(f"  - Eksik odds: {missing_odds} ({100-odds_percentage:.1f}%)")
        print(f"  - Odds orani: {odds_percentage:.1f}%")
    if total_odds_available > 0:
        unused_odds = total_odds_available - total_matches_with_odds
        usage_rate = (total_matches_with_odds / total_odds_available) * 100
        print(f"  - Kullanilmayan odds: {unused_odds} ({100-usage_rate:.1f}%)")
        print(f"  - Odds kullanim orani: {usage_rate:.1f}%")
    print(f"{'='*60}")
    
    # Yedek al (Path objesi ise string'e çevir)
    export_path_str = str(export_path)
    backup_path = export_path_str.replace('.json', '_backup.json')
    if os.path.exists(export_path_str):
        import shutil
        shutil.copy2(export_path_str, backup_path)
        print(f"\nYedek olusturuldu: {backup_path}")
    
    # Kaydet
    print(f"\nExport dosyasi guncelleniyor...")
    with open(export_path_str, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Tamamlandi! Export dosyasi guncellendi: {export_path_str}")
    
    return export_data

def main():
    """Ana fonksiyon"""
    # Dosya yolları
    script_dir = Path(__file__).parent
    odds_dir = script_dir  # odds klasörü
    export_path = script_dir.parent / 'football_brain_export.json'
    
    print("="*60)
    print("ODDS VERILERI JSON EXPORT'A ENTEGRE EDILIYOR")
    print("="*60)
    
    # Tüm CSV dosyalarını oku
    print("\n1. CSV dosyalari okunuyor...")
    odds_data = read_all_csv_files(odds_dir)
    
    print(f"\nToplam {len(odds_data)} maç odds verisi bulundu")
    
    if len(odds_data) == 0:
        print("HATA: Odds verisi bulunamadi!")
        return
    
    # Export dosyasına entegre et
    print("\n2. Export dosyasina entegre ediliyor...")
    integrate_to_export(odds_data, export_path)
    
    print("\n" + "="*60)
    print("ISLEM TAMAMLANDI!")
    print("="*60)

if __name__ == "__main__":
    main()

