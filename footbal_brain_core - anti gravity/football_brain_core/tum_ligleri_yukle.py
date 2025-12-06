"""
TÜM LİGLERİ YÜKLE - Tüm master klasörlerinden
england-master, espana-master, europe-master, italy-master
"""
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from src.db.connection import get_session
from src.db.schema import League, Team, Match
from src.db.repositories import LeagueRepository, TeamRepository, MatchRepository
from src.config import Config

class DataFileParser:
    """football-data.co.uk formatındaki dosyaları parse eder"""
    
    def __init__(self):
        self.config = Config()
    
    def parse_date(self, date_str: str, year: int) -> Optional[datetime]:
        """Tarih string'ini parse et (örn: 'Fri Aug/13')"""
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
    
    def parse_file(self, file_path: Path, league_name: str, season: int) -> int:
        """Dosyayı parse et ve veritabanına yükle"""
        session = get_session()
        loaded_count = 0
        
        try:
            league = LeagueRepository.get_or_create(session, league_name)
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            current_date = None
            
            for line in lines:
                line = line.strip()
                
                # Tarih satırı: [Fri Aug/13] veya Fri Aug/6 2021
                if line.startswith('[') and ']' in line:
                    date_str = line[1:line.index(']')]
                    current_date = self.parse_date(date_str, season)
                    if not current_date:
                        continue
                elif re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+', line) and not current_date:
                    date_match = re.search(r'(\w+\s+\w+/\d+)\s+(\d{4})', line)
                    if date_match:
                        date_str = date_match.group(1)
                        year = int(date_match.group(2))
                        current_date = self.parse_date(date_str, year - 1 if year > 2020 else year)
                
                # Maç satırı
                elif current_date and (re.match(r'^\s*\d+\.\d+\s+', line) or re.match(r'^\s+\d+\.\d+\s+', line)):
                    parts = re.split(r'\s{2,}', line.strip())
                    
                    if len(parts) >= 3:
                        time_str = parts[0]
                        try:
                            hour, minute = map(int, time_str.split('.'))
                            match_datetime = current_date.replace(hour=hour, minute=minute)
                        except:
                            match_datetime = current_date.replace(hour=12, minute=0)
                        
                        remaining = ' '.join(parts[1:])
                        score_match = re.search(r'(\d+)\s*-\s*(\d+)', remaining)
                        
                        if score_match:
                            home_score = int(score_match.group(1))
                            away_score = int(score_match.group(2))
                            
                            if ' v ' in remaining.lower():
                                v_pos = remaining.lower().find(' v ')
                                home_team_name = remaining[:v_pos].strip()
                                away_part = remaining[v_pos + 3:].strip()
                                away_team_name = re.sub(r'\s*\d+\s*-\s*\d+.*', '', away_part).strip()
                            else:
                                score_pos = remaining.find(score_match.group(0))
                                home_team_name = remaining[:score_pos].strip()
                                away_team_name = remaining[score_pos + len(score_match.group(0)):].strip()
                            
                            if home_team_name and away_team_name:
                                home_team = TeamRepository.get_or_create(
                                    session, name=home_team_name, league_id=league.id
                                )
                                away_team = TeamRepository.get_or_create(
                                    session, name=away_team_name, league_id=league.id
                                )
                                
                                match_id = f"{league.id}_{season}_{home_team.id}_{away_team.id}_{match_datetime.strftime('%Y%m%d%H%M')}"
                                
                                MatchRepository.get_or_create(
                                    session,
                                    match_id=match_id,
                                    league_id=league.id,
                                    home_team_id=home_team.id,
                                    away_team_id=away_team.id,
                                    match_date=match_datetime,
                                    home_score=home_score,
                                    away_score=away_score,
                                    status="FT"
                                )
                                loaded_count += 1
            
            session.commit()
            print(f"  [OK] {loaded_count} mac yuklendi")
            
        except Exception as e:
            session.rollback()
            print(f"  [HATA] {e}")
        finally:
            session.close()
        
        return loaded_count

def yukle_england_master():
    """England-master klasöründen tüm ligleri yükle"""
    parser = DataFileParser()
    england_dir = project_root / "england-master"
    
    # Sezon klasörlerini bul - 2021-2025 (2025-26 dahil)
    season_dirs = [d for d in england_dir.iterdir() if d.is_dir() and re.match(r'\d{4}-\d{2}', d.name)]
    # 2021-2025 arası filtrele (2025-26 dahil)
    season_dirs = [d for d in season_dirs if 2021 <= int(d.name.split('-')[0]) <= 2025]
    
    total_loaded = 0
    
    for season_dir in sorted(season_dirs):
        season_str = season_dir.name
        season = int(season_str.split('-')[0])
        
        # Tüm .txt dosyalarını bul
        txt_files = list(season_dir.glob("*.txt"))
        
        for txt_file in txt_files:
            # Lig adını dosya adından çıkar
            file_name = txt_file.stem
            if 'premierleague' in file_name.lower() or 'premier' in file_name.lower():
                league_name = "Premier League"
            elif 'championship' in file_name.lower():
                league_name = "Championship"
            elif 'league1' in file_name.lower() or 'league 1' in file_name.lower():
                league_name = "League One"
            elif 'league2' in file_name.lower() or 'league 2' in file_name.lower():
                league_name = "League Two"
            elif 'nationalleague' in file_name.lower():
                league_name = "National League"
            else:
                continue  # Cup maçları vb. atla
            
            print(f"  {season}: {league_name} - {txt_file.name}")
            count = parser.parse_file(txt_file, league_name, season)
            total_loaded += count
    
    return total_loaded

def yukle_espana_master():
    """Espana-master klasöründen tüm ligleri yükle"""
    parser = DataFileParser()
    espana_dir = project_root / "espana-master"
    
    season_dirs = [d for d in espana_dir.iterdir() if d.is_dir() and re.match(r'\d{4}-\d{2}', d.name)]
    # 2021-2025 arası filtrele
    season_dirs = [d for d in season_dirs if 2021 <= int(d.name.split('-')[0]) <= 2025]
    
    total_loaded = 0
    
    for season_dir in sorted(season_dirs):
        season_str = season_dir.name
        season = int(season_str.split('-')[0])
        
        txt_files = list(season_dir.glob("*.txt"))
        
        for txt_file in txt_files:
            file_name = txt_file.stem
            if 'liga' in file_name.lower() and '1' in file_name:
                league_name = "La Liga"
            elif 'liga' in file_name.lower() and '2' in file_name:
                league_name = "La Liga 2"
            else:
                continue
            
            print(f"  {season}: {league_name} - {txt_file.name}")
            count = parser.parse_file(txt_file, league_name, season)
            total_loaded += count
    
    return total_loaded

def yukle_italy_master():
    """Italy-master klasöründen tüm ligleri yükle"""
    parser = DataFileParser()
    italy_dir = project_root / "italy-master"
    
    season_dirs = [d for d in italy_dir.iterdir() if d.is_dir() and re.match(r'\d{4}-\d{2}', d.name)]
    # 2021-2025 arası filtrele
    season_dirs = [d for d in season_dirs if 2021 <= int(d.name.split('-')[0]) <= 2025]
    
    total_loaded = 0
    
    for season_dir in sorted(season_dirs):
        season_str = season_dir.name
        season = int(season_str.split('-')[0])
        
        txt_files = list(season_dir.glob("*.txt"))
        
        for txt_file in txt_files:
            file_name = txt_file.stem
            if 'seriea' in file_name.lower() or 'serie a' in file_name.lower():
                league_name = "Serie A"
            elif 'serieb' in file_name.lower() or 'serie b' in file_name.lower():
                league_name = "Serie B"
            else:
                continue
            
            print(f"  {season}: {league_name} - {txt_file.name}")
            count = parser.parse_file(txt_file, league_name, season)
            total_loaded += count
    
    return total_loaded

def yukle_europe_master():
    """Europe-master klasöründen tüm ligleri yükle"""
    parser = DataFileParser()
    europe_dir = project_root / "europe-master"
    
    # Ülke klasörlerini bul
    country_dirs = [d for d in europe_dir.iterdir() if d.is_dir()]
    
    total_loaded = 0
    
    for country_dir in sorted(country_dirs):
        country_name = country_dir.name
        
        # Ülke klasöründeki tüm .txt dosyalarını bul
        txt_files = list(country_dir.glob("*.txt"))
        
        for txt_file in txt_files:
            file_name = txt_file.name
            
            # Sezon ve lig kodunu çıkar (örn: 2021-22_fr1.txt)
            season_match = re.search(r'(\d{4})-(\d{2})', file_name)
            if not season_match:
                continue
            
            season = int(season_match.group(1))
            
            # Sadece 2021-2025 arası
            if season < 2021 or season > 2025:
                continue
            
            # Lig kodunu çıkar (fr1, pt1, tr1, vb.)
            lig_code_match = re.search(r'_([a-z]{2}\d)', file_name)
            if not lig_code_match:
                continue
            
            lig_code = lig_code_match.group(1)
            
            # Ülke ve lig adını belirle
            country_league_map = {
                'fr1': ('France', 'Ligue 1'),
                'fr2': ('France', 'Ligue 2'),
                'pt1': ('Portugal', 'Liga Portugal'),
                'pt2': ('Portugal', 'Liga Portugal 2'),
                'tr1': ('Turkey', 'Süper Lig'),
                'nl1': ('Netherlands', 'Eredivisie'),
                'nl2': ('Netherlands', 'Eerste Divisie'),
                'be1': ('Belgium', 'Pro League'),
                'gr1': ('Greece', 'Super League'),
                'cz1': ('Czech Republic', 'First League'),
            }
            
            if lig_code in country_league_map:
                country, league_name = country_league_map[lig_code]
                print(f"  {season}: {league_name} ({country}) - {txt_file.name}")
                count = parser.parse_file(txt_file, league_name, season)
                total_loaded += count
    
    return total_loaded

def tum_ligleri_yukle():
    """Tüm master klasörlerinden tüm ligleri yükle"""
    print("=" * 80)
    print("TUM LIGLERI YUKLEME")
    print("=" * 80)
    print("\nKaynaklar:")
    print("  • england-master")
    print("  • espana-master")
    print("  • italy-master")
    print("  • europe-master")
    print("=" * 80)
    
    total_all = 0
    
    # 1. England
    print("\n1. ENGLAND-MASTER")
    print("-" * 80)
    total_england = yukle_england_master()
    total_all += total_england
    print(f"   Toplam: {total_england} mac")
    
    # 2. Espana
    print("\n2. ESPANA-MASTER")
    print("-" * 80)
    total_espana = yukle_espana_master()
    total_all += total_espana
    print(f"   Toplam: {total_espana} mac")
    
    # 3. Italy
    print("\n3. ITALY-MASTER")
    print("-" * 80)
    total_italy = yukle_italy_master()
    total_all += total_italy
    print(f"   Toplam: {total_italy} mac")
    
    # 4. Europe
    print("\n4. EUROPE-MASTER")
    print("-" * 80)
    total_europe = yukle_europe_master()
    total_all += total_europe
    print(f"   Toplam: {total_europe} mac")
    
    print("\n" + "=" * 80)
    print(f"TOPLAM YUKLENEN MAC: {total_all}")
    print("=" * 80)
    print("\n✅ Tüm ligler veritabanına yüklendi!")

if __name__ == "__main__":
    tum_ligleri_yukle()

