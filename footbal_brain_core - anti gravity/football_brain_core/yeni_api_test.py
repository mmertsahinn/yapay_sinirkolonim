"""Yeni API key testi"""
import os
import sys
import requests
from pathlib import Path

os.environ["API_FOOTBALL_KEY"] = "5abc4531c6a98fedb6a657d7f439d1c0"
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("YENI API KEY TESTI")
print("=" * 80)

api_key = os.environ["API_FOOTBALL_KEY"]
url = "https://v3.football.api-sports.io/fixtures"
headers = {
    "x-apisports-key": api_key,
    "Content-Type": "application/json"
}

# Test: Premier League 2023
print(f"\nAPI Key: {api_key[:20]}...")
print("Test: Premier League 2023 (Ağustos)")

params = {
    "league": 39,  # Premier League
    "season": 2023,
    "from": "2023-08-01",
    "to": "2023-08-31"
}

try:
    response = requests.get(url, headers=headers, params=params, timeout=30)
    print(f"Status Code: {response.status_code}")
    
    data = response.json()
    
    if "errors" in data and data["errors"]:
        print(f"[HATA] API Errors: {data['errors']}")
        if "access" in data["errors"]:
            print(f"[UYARI] Hesap durumu: {data['errors']['access']}")
    else:
        print("[OK] Hata yok!")
    
    if "response" in data:
        count = len(data['response'])
        print(f"Fikstur sayisi: {count}")
        if count > 0:
            print(f"[OK] API CALISIYOR! {count} fikstur bulundu!")
            print(f"Ilk fikstur: {data['response'][0].get('fixture', {}).get('date', 'N/A')}")
        else:
            print("[UYARI] 0 fikstur dondurdu")
    
    # Rate limit
    if "x-ratelimit-requests-limit" in response.headers:
        limit = response.headers["x-ratelimit-requests-limit"]
        remaining = response.headers.get("x-ratelimit-requests-remaining", "N/A")
        print(f"\nRate Limit: {limit}")
        print(f"Remaining: {remaining}")
    
except Exception as e:
    print(f"[HATA] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)






