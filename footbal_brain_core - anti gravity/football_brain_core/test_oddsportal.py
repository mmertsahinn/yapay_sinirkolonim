"""
OddsPortal HTML yapısını test eder
"""
import sys
import io

# Windows encoding sorunu için
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
from bs4 import BeautifulSoup

url = "https://www.oddsportal.com/football/italy/serie-a-2021-2022/results/"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

print("📥 Sayfa çekiliyor...")
response = requests.get(url, headers=headers, timeout=30)
print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'lxml')
    
    # HTML yapısını incele
    print("\n📋 Sayfa başlığı:")
    print(soup.title.string if soup.title else "Başlık yok")
    
    # Tabloları bul
    tables = soup.find_all('table')
    print(f"\n📊 Tablo sayısı: {len(tables)}")
    
    for i, table in enumerate(tables[:3]):  # İlk 3 tabloyu göster
        print(f"\n--- Tablo {i+1} ---")
        print(f"Class: {table.get('class')}")
        rows = table.find_all('tr')
        print(f"Satır sayısı: {len(rows)}")
        if rows:
            print(f"İlk satır örneği:")
            print(rows[0].get_text()[:200])
    
    # Maç verilerini içeren div'leri bul
    match_divs = soup.find_all('div', class_=lambda x: x and ('match' in x.lower() or 'result' in x.lower()))
    print(f"\n📋 Maç div sayısı: {len(match_divs)}")
    
    # Script tag'lerini kontrol et (JavaScript ile yükleniyor olabilir)
    scripts = soup.find_all('script')
    print(f"\n📜 Script tag sayısı: {len(scripts)}")
    
    # HTML'in bir kısmını kaydet
    with open('oddsportal_test.html', 'w', encoding='utf-8') as f:
        f.write(str(soup.prettify()[:50000]))  # İlk 50KB
    print("\n✅ HTML kaydedildi: oddsportal_test.html")
    
else:
    print(f"❌ Hata: {response.status_code}")

