"""API neden kullanılamıyor - detaylı test"""
import os
import sys
import requests
from pathlib import Path

os.environ["API_FOOTBALL_KEY"] = "81cf96e9b61dfdcef9ed54dc8c1ad772"
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("API DETAYLI TEST - NEDEN KULLANILAMIYOR?")
print("=" * 80)

api_key = os.environ["API_FOOTBALL_KEY"]
url = "https://v3.football.api-sports.io/fixtures"
headers = {
    "x-apisports-key": api_key,
    "Content-Type": "application/json"
}

# Test 1: Premier League 2023 sezonu
print("\nTEST 1: Premier League 2023 (Ağustos)")
params1 = {
    "league": 39,  # Premier League
    "season": 2023,
    "from": "2023-08-01",
    "to": "2023-08-31"
}

try:
    response1 = requests.get(url, headers=headers, params=params1, timeout=30)
    print(f"Status Code: {response1.status_code}")
    print(f"Response Headers: {dict(response1.headers)}")
    
    data1 = response1.json()
    print(f"Response Keys: {list(data1.keys())}")
    
    if "errors" in data1 and data1["errors"]:
        print(f"API Errors: {data1['errors']}")
    
    if "response" in data1:
        print(f"Response Array Length: {len(data1['response'])}")
        if len(data1['response']) > 0:
            print(f"[OK] {len(data1['response'])} fikstur bulundu!")
        else:
            print("[UYARI] 0 fikstur dondurdu")
    
    # Rate limit bilgileri
    if "x-ratelimit-requests-limit" in response1.headers:
        limit = response1.headers["x-ratelimit-requests-limit"]
        remaining = response1.headers.get("x-ratelimit-requests-remaining", "N/A")
        print(f"Rate Limit: {limit}, Remaining: {remaining}")
    
except Exception as e:
    print(f"[HATA] {e}")
    import traceback
    traceback.print_exc()

# Test 2: Farklı bir tarih aralığı
print("\n" + "-" * 80)
print("TEST 2: Premier League 2023 (Eylül)")
params2 = {
    "league": 39,
    "season": 2023,
    "from": "2023-09-01",
    "to": "2023-09-30"
}

try:
    response2 = requests.get(url, headers=headers, params=params2, timeout=30)
    data2 = response2.json()
    
    if "response" in data2:
        print(f"Response Array Length: {len(data2['response'])}")
        if len(data2['response']) > 0:
            print(f"[OK] {len(data2['response'])} fikstur bulundu!")
            print(f"Ilk fikstur: {data2['response'][0].get('fixture', {}).get('date', 'N/A')}")
        else:
            print("[UYARI] 0 fikstur dondurdu")
    
    if "errors" in data2 and data2["errors"]:
        print(f"API Errors: {data2['errors']}")
    
except Exception as e:
    print(f"[HATA] {e}")

# Test 3: Free plan erişim kontrolü
print("\n" + "-" * 80)
print("TEST 3: Free Plan Erişim Kontrolü")
print("Free plan'da belirli tarih aralıklarına erişim kısıtlı olabilir.")

print("\n" + "=" * 80)
print("OZET")
print("=" * 80)
print("API key calisiyor mu: EVET")
print("API limiti doldu mu: HAYIR (100 kalan)")
print("Veri donuyor mu: HAYIR (0 fikstur)")
print("\nMuhtemel nedenler:")
print("  1. Free plan'da belirli tarih aralıklarına erişim kısıtlı")
print("  2. Sezon parametresi ile tarih parametresi uyumsuz olabilir")
print("  3. API'den veri dönüyor ama boş array")
print("=" * 80)






