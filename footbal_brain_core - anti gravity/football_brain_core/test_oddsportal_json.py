"""
OddsPortal JSON data'sını bul
"""
import sys
import io
import re
import json
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
from bs4 import BeautifulSoup

url = "https://www.oddsportal.com/football/italy/serie-a-2021-2022/results/"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
}

print(f"URL: {url}")
print("=" * 80)

try:
    response = requests.get(url, headers=headers, timeout=30)
    print(f"Status Code: {response.status_code}")
    print()
    
    # HTML içinde JSON data arayalım
    html = response.text
    
    # window.__INITIAL_STATE__ veya benzeri pattern'ler
    patterns = [
        r'window\.__INITIAL_STATE__\s*=\s*({.+?});',
        r'window\.__NEXT_DATA__\s*=\s*({.+?});',
        r'var\s+initialState\s*=\s*({.+?});',
        r'data:\s*({.+?}),',
    ]
    
    print("JSON pattern'leri araniyor...")
    for pattern in patterns:
        matches = re.findall(pattern, html, re.DOTALL)
        if matches:
            print(f"✅ Pattern bulundu: {pattern[:30]}...")
            print(f"   Match sayisi: {len(matches)}")
            if matches:
                try:
                    data = json.loads(matches[0][:5000])  # İlk 5000 karakter
                    print(f"   JSON parse edildi (ilk 5000 karakter)")
                    print(f"   Keys: {list(data.keys())[:10]}")
                except:
                    print(f"   JSON parse edilemedi")
            print()
    
    # Script tag'lerinde data arayalım
    soup = BeautifulSoup(response.content, 'lxml')
    scripts = soup.find_all('script')
    print(f"Script tag sayisi: {len(scripts)}")
    print()
    
    for i, script in enumerate(scripts[:10]):
        if script.string:
            text = script.string
            # JSON benzeri yapılar
            if 'match' in text.lower() or 'result' in text.lower() or 'odds' in text.lower():
                print(f"Script {i+1}:")
                print(f"  Length: {len(text)}")
                print(f"  Preview: {text[:200]}...")
                print()
                
                # JSON objesi var mı?
                json_matches = re.findall(r'\{[^{}]{100,}\}', text)
                if json_matches:
                    print(f"  ✅ JSON benzeri yapı bulundu ({len(json_matches)} adet)")
                    print()
    
    # API endpoint'leri arayalım
    print("=" * 80)
    print("API ENDPOINT'LERİ:")
    print("=" * 80)
    
    api_patterns = [
        r'["\']([^"\']*api[^"\']*match[^"\']*)["\']',
        r'["\']([^"\']*api[^"\']*result[^"\']*)["\']',
        r'["\']([^"\']*api[^"\']*odds[^"\']*)["\']',
        r'fetch\(["\']([^"\']+)["\']',
        r'axios\.(?:get|post)\(["\']([^"\']+)["\']',
    ]
    
    for pattern in api_patterns:
        matches = re.findall(pattern, html, re.IGNORECASE)
        if matches:
            print(f"Pattern: {pattern[:40]}...")
            for match in set(matches[:5]):
                print(f"  - {match}")
            print()

except Exception as e:
    print(f"HATA: {e}")
    import traceback
    traceback.print_exc()





