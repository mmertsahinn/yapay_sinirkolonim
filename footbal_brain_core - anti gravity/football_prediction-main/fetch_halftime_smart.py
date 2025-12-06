"""
API-Football'dan ilk yarı verilerini akıllı çekme
Fuzzy matching + tarih bazlı arama
"""
import pandas as pd
import requests
import time
from difflib import SequenceMatcher
import json

API_KEY = "647f5de88a29d150a9d4e2c0c7b636fb"
BASE_URL = "https://v3.football.api-sports.io"

headers = {'x-apisports-key': API_KEY}

def similarity(a, b):
    """İki string arasında benzerlik oranı"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_match_by_date(date, home_team, away_team, max_results=50):
    """
    Tarih bazlı maç arama
    """
    url = f"{BASE_URL}/fixtures"
    params = {'date': date}
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    fixtures = data.get('response', [])
    
    # En benzer maçı bul
    best_match = None
    best_score = 0.0
    
    for fixture in fixtures[:max_results]:
        api_home = fixture['teams']['home']['name']
        api_away = fixture['teams']['away']['name']
        
        # Benzerlik skorları
        home_sim = similarity(home_team, api_home)
        away_sim = similarity(away_team, api_away)
        total_sim = (home_sim + away_sim) / 2
        
        if total_sim > best_score:
            best_score = total_sim
            best_match = {
                'fixture': fixture,
                'similarity': total_sim,
                'api_home': api_home,
                'api_away': api_away
            }
    
    return best_match if best_score > 0.5 else None

print("="*80)
print("🔍 AKILLI İLK YARI VERİSİ ÇEKİMİ")
print("="*80)

# Test: API durumu
test_url = f"{BASE_URL}/status"
response = requests.get(test_url, headers=headers)
data = response.json()
print(f"\n✅ API Aktif: {data['response']['requests']['current']}/{data['response']['requests']['limit_day']} request")

# Maçları yükle
df = pd.read_csv('son_4_ay_tum_maclarin_verisi/7_temmuz_ve_sonrasi_TUM_VERI.csv')
print(f"📂 {len(df)} maç yüklendi")

# İlk 5 maç için test
print("\n" + "="*80)
print("🧪 TEST: İlk 5 maç")
print("="*80)

success_count = 0
fail_count = 0

for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    home = row['home_team']
    away = row['away_team']
    date = row['date']
    
    print(f"\n{idx+1}. {home} vs {away}")
    print(f"   📅 Tarih: {date}")
    
    # API'dan ara
    result = find_match_by_date(date, home, away)
    
    if result:
        fixture = result['fixture']
        score = fixture.get('score', {})
        halftime = score.get('halftime', {})
        fulltime = score.get('fulltime', {})
        
        print(f"   ✅ BULUNDU! (Benzerlik: {result['similarity']:.0%})")
        print(f"      API: {result['api_home']} vs {result['api_away']}")
        print(f"      İlk Yarı: {halftime.get('home')}-{halftime.get('away')}")
        print(f"      Maç Sonu: {fulltime.get('home')}-{fulltime.get('away')}")
        print(f"      Lig: {fixture['league']['name']}")
        success_count += 1
    else:
        print(f"   ❌ Bulunamadı")
        fail_count += 1
    
    # Rate limit (100 request/gün)
    time.sleep(1)

print("\n" + "="*80)
print(f"SONUÇ: {success_count} başarılı, {fail_count} başarısız")
print("="*80)

if success_count > 0:
    print("\n✅ Sistem çalışıyor! Tüm 1626 maç için çekebiliriz!")
    print("⏳ Tahmini süre: ~30 dakika (rate limit nedeniyle)")
    print("\n📌 Devam etmek için: python fetch_all_halftime.py")
else:
    print("\n⚠️ Hiç maç bulunamadı. Stratejiyi gözden geçirmeliyiz.")



