"""Tarih dışında kalan maçların yıllarını kontrol et"""
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
football_data_dir = project_root / "football_data"

def parse_date(date_str: str, year: int) -> Optional[datetime]:
    """Tarih string'ini parse et"""
    try:
        date_str = date_str.strip().replace("[", "").replace("]", "")
        
        month_map = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
        }
        
        for month_name, month_num in month_map.items():
            if month_name in date_str:
                day_match = re.search(r'/(\d+)', date_str)
                if day_match:
                    day = int(day_match.group(1))
                    if month_num >= 8:
                        season_year = year
                    else:
                        season_year = year + 1
                    return datetime(season_year, month_num, day)
    except:
        pass
    return None

# Tarih dışında kalan maçların yıllarını topla
tarih_disinda_yillar = {}

# France ve Portugal dosyalarını kontrol et (2021 sezonu sorunlu görünüyor)
for country in ["france", "portugal"]:
    country_dir = football_data_dir / country
    if not country_dir.exists():
        continue
    
    print(f"\n{country.upper()} - Tarih Dışında Kalan Maçlar:")
    print("-" * 60)
    
    for txt_file in country_dir.glob("*.txt"):
        if "2021-22" in txt_file.name:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            current_date = None
            tarih_disinda_count = 0
            yil_dagilimi = {}
            
            for line in lines:
                line = line.strip()
                
                # Tarih satırı
                if line.startswith('[') and ']' in line:
                    date_str = line[1:line.index(']')]
                    current_date = parse_date(date_str, 2021)
                elif re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+', line):
                    date_match = re.search(r'(\w+\s+\w+/\d+)\s+(\d{4})', line)
                    if date_match:
                        date_str = date_match.group(1)
                        year = int(date_match.group(2))
                        current_date = parse_date(date_str, year - 1 if year > 2020 else year)
                
                # Maç satırı
                elif current_date and (re.match(r'^\s*\d+\.\d+\s+', line) or re.match(r'^\s+\d+\.\d+\s+', line)):
                    match_year = current_date.year
                    
                    if match_year < 2021 or match_year > 2025:
                        tarih_disinda_count += 1
                        if match_year not in yil_dagilimi:
                            yil_dagilimi[match_year] = 0
                        yil_dagilimi[match_year] += 1
            
            if tarih_disinda_count > 0:
                print(f"\n  {txt_file.name}:")
                print(f"    Toplam tarih dışı: {tarih_disinda_count} maç")
                for yil, count in sorted(yil_dagilimi.items()):
                    print(f"      {yil}: {count} maç")

print("\n" + "=" * 60)
print("ÖZET: Tarih dışında kalan maçlar muhtemelen 2020 veya 2026 yılında")
print("(2021-2025 aralığı dışında)")

