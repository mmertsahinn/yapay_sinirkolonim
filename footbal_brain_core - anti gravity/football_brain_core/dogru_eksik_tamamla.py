"""
Doğru eksik maçları tamamla - 2021-2024
Beklenen toplam: 6,424 maç
Mevcut: 3,926 maç
Eksik: 2,498 maç
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime, date

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# YENİ API KEY (ÇALIŞAN)
os.environ["API_FOOTBALL_KEY"] = "5abc4531c6a98fedb6a657d7f439d1c0"

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from src.ingestion.historical_loader import HistoricalLoader
from src.config import Config
from src.db.connection import get_session
from src.db.schema import Match, League

print("=" * 80)
print("DOGRU EKSIK MACLARI TAMAMLA - 2021-2024")
print("=" * 80)
print(f"Beklenen toplam: 6,424 mac")
print(f"Mevcut: 3,926 mac")
print(f"Eksik: 2,498 mac")
print(f"Baslangic zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"API Key: {os.environ['API_FOOTBALL_KEY'][:20]}...")
print()

config = Config()
loader = HistoricalLoader()
session = get_session()

# Doğru beklenen maç sayıları (kullanıcının verdiği bilgilere göre)
expected_matches = {
    # Premier League: 3 sezon x 380 = 1140
    ("Premier League", 2021): 380,
    ("Premier League", 2022): 380,
    ("Premier League", 2023): 380,
    
    # La Liga: 3 sezon x 380 = 1140
    ("La Liga", 2021): 380,
    ("La Liga", 2022): 380,
    ("La Liga", 2023): 380,
    
    # Serie A: 3 sezon x 380 = 1140
    ("Serie A", 2021): 380,
    ("Serie A", 2022): 380,
    ("Serie A", 2023): 380,
    
    # Bundesliga: 3 sezon x 306 = 918
    ("Bundesliga", 2021): 306,
    ("Bundesliga", 2022): 306,
    ("Bundesliga", 2023): 306,
    
    # Ligue 1: 2x380 + 1x306 = 1066
    ("Ligue 1", 2021): 380,
    ("Ligue 1", 2022): 380,
    ("Ligue 1", 2023): 306,  # 18 takım
    
    # Liga Portugal: 3 sezon x 306 = 918
    ("Liga Portugal", 2021): 306,
    ("Liga Portugal", 2022): 306,
    ("Liga Portugal", 2023): 306,
    
    # Süper Lig: 380 + 342 + 380 = 1102
    ("Süper Lig", 2021): 380,
    ("Süper Lig", 2022): 342,  # 19 takım (deprem)
    ("Süper Lig", 2023): 380,
}

# Eksik sezonları bul
eksik_sezonlar = []

for (league_name, season), expected in expected_matches.items():
    league_db = session.query(League).filter(League.name == league_name).first()
    if league_db:
        season_start = datetime(season, 8, 1)
        season_end = datetime(season + 1, 7, 31)
        current_count = session.query(Match).filter(
            Match.league_id == league_db.id,
            Match.match_date >= season_start,
            Match.match_date <= season_end
        ).count()
        
        eksik = expected - current_count
        if eksik > 0:
            eksik_sezonlar.append((league_name, season, expected, current_count, eksik))
            print(f"{league_name} {season}: {current_count}/{expected} mac - {eksik} eksik")

session.close()

total_operations = len(eksik_sezonlar)
toplam_eksik = sum(eksik for _, _, _, _, eksik in eksik_sezonlar)

print()
print("=" * 80)
print(f"TOPLAM EKSIK: {toplam_eksik} mac")
print(f"Eksik sezon/lig kombinasyonu: {total_operations}")
print("=" * 80)
print()

operation_count = 0
total_new_matches = 0

try:
    for league_name, season, expected, current_count, eksik in eksik_sezonlar:
        operation_count += 1
        
        print(f"\n[{operation_count}/{total_operations}] {league_name} - Sezon {season}")
        print(f"  Mevcut: {current_count}/{expected} mac - {eksik} eksik")
        print("-" * 80)
        
        session = get_session()
        league_db = session.query(League).filter(League.name == league_name).first()
        if league_db:
            season_start = datetime(season, 8, 1)
            season_end = datetime(season + 1, 7, 31)
            before_count = session.query(Match).filter(
                Match.league_id == league_db.id,
                Match.match_date >= season_start,
                Match.match_date <= season_end
            ).count()
        else:
            before_count = 0
        session.close()
        
        try:
            # Ligleri yükle
            loader.load_leagues(season)
            
            # Takımları yükle
            print(f"  Takimlar yukleniyor...")
            loader.load_teams_for_league(league_name, season)
            print(f"  [OK] Takimlar yuklendi")
            
            # Maçları yükle (tam sezon)
            print(f"  Maclar yukleniyor...")
            loader.load_matches_for_league(league_name, season)
            print(f"  [OK] Maclar yuklendi")
            
            # Yeni maç sayısını kontrol et
            session = get_session()
            league_db = session.query(League).filter(League.name == league_name).first()
            if league_db:
                season_start = datetime(season, 8, 1)
                season_end = datetime(season + 1, 7, 31)
                after_count = session.query(Match).filter(
                    Match.league_id == league_db.id,
                    Match.match_date >= season_start,
                    Match.match_date <= season_end
                ).count()
                new_matches = after_count - before_count
                total_new_matches += new_matches
                print(f"  Yeni eklenen mac: {new_matches}")
                print(f"  Toplam mac: {after_count}/{expected}")
                
                if after_count >= expected:
                    print(f"  [TAM] Sezon tamamlandi!")
                else:
                    kalan_eksik = expected - after_count
                    print(f"  [UYARI] Hala {kalan_eksik} mac eksik!")
            session.close()
            
            # API limit bilgisi
            if hasattr(loader.api_client, 'requests_today'):
                remaining = loader.api_client.daily_limit - loader.api_client.requests_today
                if remaining <= 0:
                    print(f"  [UYARI] API LIMITI DOLDU! {remaining} kalan")
                    print(f"  [UYARI] Yeni API key gerekiyor!")
                    print(f"\n  Durduruldu! {operation_count}/{total_operations} tamamlandi.")
                    break
                elif remaining < 10:
                    print(f"  [UYARI] API limiti yaklasiyor: {remaining} kalan")
                else:
                    print(f"  [INFO] API requests kalan: {remaining}/{loader.api_client.daily_limit}")
            
        except Exception as e:
            print(f"  [HATA] {league_name} {season} yukleme hatasi: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Kısa bir bekleme (rate limit için)
        time.sleep(0.5)
    
    print("\n" + "=" * 80)
    print("EKSIK SEZONLAR TAMAMLANDI!")
    print("=" * 80)
    print(f"Bitis zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Toplam islem: {operation_count}/{total_operations}")
    print(f"Yeni eklenen mac: {total_new_matches}")
    
    # Son durumu göster
    session = get_session()
    final_total = session.query(Match).count()
    beklenen_toplam = 6424
    kalan_eksik = beklenen_toplam - final_total
    
    print(f"\nToplam mac sayisi: {final_total}/{beklenen_toplam}")
    if kalan_eksik > 0:
        print(f"Kalan eksik: {kalan_eksik} mac")
    else:
        print("[TAM] Tum maclar yuklendi!")
    
    session.close()
    
except KeyboardInterrupt:
    print("\n\n[UYARI] Kullanici tarafindan durduruldu!")
    print(f"Toplam islem: {operation_count}/{total_operations}")
    print(f"Yeni mac: {total_new_matches}")
except Exception as e:
    print(f"\n\n[HATA] Genel hata: {e}")
    import traceback
    traceback.print_exc()
    print(f"Toplam islem: {operation_count}/{total_operations}")

print("\n" + "=" * 80)

