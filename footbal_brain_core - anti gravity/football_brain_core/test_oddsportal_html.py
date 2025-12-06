"""
OddsPortal HTML yapısını test et
"""
import sys
import io
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
    print(f"Content Length: {len(response.content)} bytes")
    print()
    
    soup = BeautifulSoup(response.content, 'lxml')
    
    # Tüm tabloları bul
    tables = soup.find_all('table')
    print(f"Toplam tablo sayisi: {len(tables)}")
    print()
    
    for i, table in enumerate(tables[:5]):
        classes = table.get('class', [])
        print(f"Tablo {i+1}:")
        print(f"  Classes: {classes}")
        print(f"  ID: {table.get('id', 'N/A')}")
        
        # İlk birkaç satırı göster
        rows = table.find_all('tr')[:3]
        print(f"  Satir sayisi: {len(table.find_all('tr'))}")
        print(f"  İlk 3 satir:")
        for j, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            print(f"    Satir {j+1}: {len(cells)} hücre")
            for k, cell in enumerate(cells[:5]):
                classes = cell.get('class', [])
                text = cell.get_text(strip=True)[:50]
                print(f"      Hücre {k+1}: classes={classes}, text='{text}'")
        print()
    
    # Div yapısını da kontrol et
    print("=" * 80)
    print("DIV YAPISI KONTROLÜ:")
    print("=" * 80)
    
    # Maç içeren div'leri bul
    match_divs = soup.find_all('div', class_=lambda x: x and ('match' in str(x).lower() or 'result' in str(x).lower()))
    print(f"Match div sayisi: {len(match_divs)}")
    
    # Participant içeren div'leri bul
    participant_divs = soup.find_all('div', class_=lambda x: x and 'participant' in str(x).lower())
    print(f"Participant div sayisi: {len(participant_divs)}")
    
    # Tüm class'ları listele
    print()
    print("Tüm unique class'lar (ilk 50):")
    all_classes = set()
    for tag in soup.find_all(class_=True):
        classes = tag.get('class', [])
        if isinstance(classes, list):
            all_classes.update(classes)
        else:
            all_classes.add(classes)
    
    for cls in sorted(list(all_classes))[:50]:
        print(f"  - {cls}")
    
except Exception as e:
    print(f"HATA: {e}")
    import traceback
    traceback.print_exc()





