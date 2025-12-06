"""
Büyük takımlarla test et
"""
import pandas as pd
import requests
from difflib import SequenceMatcher
import time

API_KEY = "647f5de88a29d150a9d4e2c0c7b636fb"
BASE_URL = "https://v3.football.api-sports.io"
headers = {'x-apisports-key': API_KEY}

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Maçları yükle
df = pd.read_csv('son_4_ay_tum_maclarin_verisi/7_temmuz_ve_sonrasi_TUM_VERI.csv')

# SON 10 maçı test et (büyük takımlar)
print("="*80)
print("🔍 BÜYÜK TAKIMLARLA TEST (SON 10 MAÇ)")
print("="*80)

for idx in range(len(df) - 10, len(df)):
    row = df.iloc[idx]
    home = row['home_team']
    away = row['away_team']
    date = row['date']
    
    print(f"\n{idx+1}. {home} vs {away} ({date})")
    
    # API sorgusu
    url = f"{BASE_URL}/fixtures"
    params = {'date': date}
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        fixtures = data.get('response', [])
        
        # En benzer maçı bul
        best_match = None
        best_score = 0.0
        
        for fixture in fixtures:
            api_home = fixture['teams']['home']['name']
            api_away = fixture['teams']['away']['name']
            
            home_sim = similarity(home, api_home)
            away_sim = similarity(away, api_away)
            total_sim = (home_sim + away_sim) / 2
            
            if total_sim > best_score:
                best_score = total_sim
                best_match = {
                    'fixture': fixture,
                    'similarity': total_sim,
                    'api_home': api_home,
                    'api_away': api_away
                }
        
        if best_match and best_score > 0.6:
            fixture = best_match['fixture']
            score = fixture.get('score', {})
            halftime = score.get('halftime', {})
            fulltime = score.get('fulltime', {})
            
            print(f"   ✅ BULUNDU! ({best_score:.0%})")
            print(f"      API: {best_match['api_home']} vs {best_match['api_away']}")
            print(f"      İlk Yarı: {halftime.get('home')}-{halftime.get('away')}")
            print(f"      Maç Sonu: {fulltime.get('home')}-{fulltime.get('away')}")
        else:
            print(f"   ❌ Bulunamadı (en iyi: {best_score:.0%})")
    
    time.sleep(1)



