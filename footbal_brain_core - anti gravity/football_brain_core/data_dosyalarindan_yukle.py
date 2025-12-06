"""
Data dosyalarından (.txt) veritabanına veri yükle
IDEMPOTENT: Var olan maçlara dokunmaz, sadece yeni maçları ekler
Sadece 2020-2025 arası maçları yükler
"""
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict

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
    
    def parse_date(self, date_str: str, season_start_year: int) -> Optional[datetime]:
        """
        Tarih string'ini parse et (örn: 'Fri Aug/13', 'Sat Nov/1')
        season_start_year: Sezon başlangıç yılı (örn: 2025-26 sezonu için 2025)
        """
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
                        # Sezon Ağustos'ta başlar: Ağustos-Aralık = sezon yılı, Ocak-Temmuz = sezon yılı + 1
                        # Örnek: 2025-26 sezonu -> Ağustos 2025 - Temmuz 2026
                        # Kasım 2025, 2025 yılında (season_start_year)
                        # Şubat 2026, 2026 yılında (season_start_year + 1)
                        if month_num >= 8:
                            # Ağustos-Aralık: sezon başlangıç yılı
                            match_year = season_start_year
                        else:
                            # Ocak-Temmuz: sezon başlangıç yılı + 1
                            match_year = season_start_year + 1
                        return datetime(match_year, month_num, day)
        except:
            pass
        return None
    
    def parse_file(self, file_path: Path, league_name: str, season: int) -> Tuple[int, int, int, int]:
        """
        Dosyayı parse et ve veritabanına yükle
        IDEMPOTENT: Var olan maçlara dokunmaz
        
        Returns: (yeni_eklenen, zaten_var, tarih_disinda, toplam_okunan)
        """
        session = get_session()
        yeni_eklenen = 0
        zaten_var = 0
        tarih_disinda = 0
        toplam_okunan = 0
        
        try:
            # Lig'i bul veya oluştur
            league = LeagueRepository.get_or_create(session, league_name)
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            current_date = None
            last_match_time = None  # Son maçın saatini sakla (saat bilgisi olmayan maçlar için)
            
            for line in lines:
                line = line.strip()
                
                # Tarih satırı: [Fri Aug/13] veya Fri Aug/6 2021 veya Sat Nov/1
                if line.startswith('[') and ']' in line:
                    date_str = line[1:line.index(']')]
                    current_date = self.parse_date(date_str, season)
                    last_match_time = None  # Yeni tarih, saati sıfırla
                    if not current_date:
                        continue
                elif re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+', line):
                    # Sat Nov/1 formatı (yıl yok, sezondan çıkarılacak)
                    date_match = re.search(r'(\w+\s+\w+/\d+)\s+(\d{4})', line)
                    if date_match:
                        # Yıl var: Fri Aug/6 2021
                        date_str = date_match.group(1)
                        year = int(date_match.group(2))
                        current_date = self.parse_date(date_str, year - 1 if year > 2020 else year)
                        last_match_time = None  # Yeni tarih, saati sıfırla
                    else:
                        # Yıl yok: Sat Nov/1 - sezondan çıkar
                        date_match = re.search(r'(\w+\s+\w+/\d+)', line)
                        if date_match:
                            date_str = date_match.group(1)
                            # 2025-26 sezonunda Kasım 2025'te oynanır, 2026'da değil
                            current_date = self.parse_date(date_str, season)
                            last_match_time = None  # Yeni tarih, saati sıfırla
                
                # Maç satırı - saat bilgisi VAR (örn: "19.00  Eyüpspor v Antalyaspor")
                elif current_date and (re.match(r'^\s*\d+\.\d+\s+', line) or re.match(r'^\s+\d+\.\d+\s+', line)):
                    parts = re.split(r'\s{2,}', line.strip())
                    
                    if len(parts) >= 3:
                        time_str = parts[0]
                        try:
                            hour, minute = map(int, time_str.split('.'))
                            match_datetime = current_date.replace(hour=hour, minute=minute)
                            last_match_time = (hour, minute)  # Saati sakla
                        except:
                            match_datetime = current_date.replace(hour=12, minute=0)
                            last_match_time = (12, 0)
                        
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
                                # Tarih kontrolü: 2020-2026 arası (2025-26 sezonu Kasım 2025 maçları dahil)
                                match_year = match_datetime.year
                                
                                # 2020-2026 yıllarındaki maçları al (2025-26 sezonu için)
                                if match_year < 2020 or match_year > 2026:
                                    tarih_disinda += 1
                                    continue
                                
                                toplam_okunan += 1
                                
                                home_team = TeamRepository.get_or_create(
                                    session, name=home_team_name, league_id=league.id
                                )
                                away_team = TeamRepository.get_or_create(
                                    session, name=away_team_name, league_id=league.id
                                )
                                
                                # Benzersiz match_id oluştur
                                match_id = f"{league.id}_{season}_{home_team.id}_{away_team.id}_{match_datetime.strftime('%Y%m%d%H%M')}"
                                
                                # VAR OLAN MAÇA DOKUNMA - Sadece kontrol et
                                existing_match = session.query(Match).filter(Match.match_id == match_id).first()
                                if existing_match:
                                    zaten_var += 1
                                    continue  # Var olan maça dokunma, atla
                                
                                # YENİ MAÇ EKLE
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
                                yeni_eklenen += 1
                
                # Maç satırı - saat bilgisi YOK (örn: "           Alanyaspor v Gaziantep FK")
                # Önceki maçın saatini kullan veya varsayılan saat kullan
                elif current_date and last_match_time and (' v ' in line.lower() or re.search(r'\d+\s*-\s*\d+', line)):
                    # Saat bilgisi olmayan maç satırı - önceki maçın saatini kullan
                    # Örnek: "           Alanyaspor              v Gaziantep FK             0-0"
                    parts = re.split(r'\s{2,}', line.strip())
                    
                    if len(parts) >= 2:
                        # Skor var mı kontrol et
                        score_match = re.search(r'(\d+)\s*-\s*(\d+)', line)
                        
                        if score_match:
                            home_score = int(score_match.group(1))
                            away_score = int(score_match.group(2))
                            
                            if ' v ' in line.lower():
                                v_pos = line.lower().find(' v ')
                                home_team_name = line[:v_pos].strip()
                                away_part = line[v_pos + 3:].strip()
                                away_team_name = re.sub(r'\s*\d+\s*-\s*\d+.*', '', away_part).strip()
                            else:
                                score_pos = line.find(score_match.group(0))
                                home_team_name = line[:score_pos].strip()
                                away_team_name = line[score_pos + len(score_match.group(0)):].strip()
                            
                            if home_team_name and away_team_name:
                                # Tarih kontrolü: 2020-2026 arası
                                match_datetime = current_date.replace(hour=last_match_time[0], minute=last_match_time[1])
                                match_year = match_datetime.year
                                
                                if match_year < 2020 or match_year > 2026:
                                    tarih_disinda += 1
                                    continue
                                
                                toplam_okunan += 1
                                
                                home_team = TeamRepository.get_or_create(
                                    session, name=home_team_name, league_id=league.id
                                )
                                away_team = TeamRepository.get_or_create(
                                    session, name=away_team_name, league_id=league.id
                                )
                                
                                # Benzersiz match_id oluştur
                                match_id = f"{league.id}_{season}_{home_team.id}_{away_team.id}_{match_datetime.strftime('%Y%m%d%H%M')}"
                                
                                # VAR OLAN MAÇA DOKUNMA - Sadece kontrol et
                                existing_match = session.query(Match).filter(Match.match_id == match_id).first()
                                if existing_match:
                                    zaten_var += 1
                                    continue
                                
                                # YENİ MAÇ EKLE
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
                                yeni_eklenen += 1
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            print(f"  [HATA] {e}")
            import traceback
            traceback.print_exc()
        finally:
            session.close()
        
        return (yeni_eklenen, zaten_var, tarih_disinda, toplam_okunan)

def load_from_data_files():
    """football_data klasöründen TÜM dosyaları otomatik bulup yükle - IDEMPOTENT"""
    print("=" * 80)
    print("FOOTBALL_DATA KLASÖRÜNDEN OTOMATIK YÜKLEME (IDEMPOTENT)")
    print("=" * 80)
    print("\nKurallar:")
    print("  • Mevcut .db dosyası kullanılacak (yeni oluşturulmayacak)")
    print("  • Var olan maçlara dokunulmayacak (update/silme yok)")
    print("  • Sadece yeni/eksik maçlar eklenecek")
    print("  • Sadece 2020-2026 arası maçlar yüklenecek (2025-26 sezonu dahil)")
    print("  • match_id ile benzersiz kontrol yapılacak")
    print("  • football_data klasöründeki TÜM dosyalar otomatik tespit edilecek")
    print("=" * 80)
    
    parser = DataFileParser()
    project_root = Path(__file__).parent
    football_data_dir = project_root / "football_data"
    
    if not football_data_dir.exists():
        print(f"\n❌ HATA: football_data klasörü bulunamadı: {football_data_dir}")
        return
    
    # Lig isim mapping
    league_mapping = {
        "england-master": {
            "1-premierleague.txt": "Premier League",
            "1-championship.txt": "Championship",
            "1-league1.txt": "League One",
            "1-league2.txt": "League Two",
        },
        "espana-master": {
            "1-liga.txt": "La Liga",
            "2-liga2.txt": "Segunda División",
        },
        "italy-master": {
            "1-seriea.txt": "Serie A",
            "2-serieb.txt": "Serie B",
        },
        "deutschland-master": {
            "1-bundesliga.txt": "Bundesliga",
            "2-bundesliga2.txt": "2. Bundesliga",
        },
        "france": {
            "fr1.txt": "Ligue 1",
            "fr2.txt": "Ligue 2",
        },
        "portugal": {
            "pt1.txt": "Liga Portugal",
            "pt2.txt": "Segunda Liga",
        },
        "turkey": {
            "tr1.txt": "Süper Lig",
        }
    }
    
    # İstatistikler
    toplam_istatistik = {
        "toplam_okunan": 0,
        "zaten_var": 0,
        "tarih_disinda": 0,
        "yeni_eklenen": 0
    }
    
    # Sezonlar - 2020-2025 (2020 sezonu 2020-21)
    seasons = [2020, 2021, 2022, 2023, 2024, 2025]
    
    # 1. MASTER KLASÖRLERİ (england-master, italy-master, espana-master, deutschland-master)
    master_dirs = ["england-master", "italy-master", "espana-master", "deutschland-master"]
    
    for master_dir_name in master_dirs:
        master_dir = football_data_dir / master_dir_name
        if not master_dir.exists():
            continue
        
        print(f"\n{master_dir_name.upper().replace('-MASTER', '')}:")
        print("-" * 80)
        
        # Sezon klasörlerini bul (2021-22, 2022-23, vs.)
        for season in seasons:
            season_str = f"{season}-{str(season+1)[-2:]}"
            season_dir = master_dir / season_str
            
            if not season_dir.exists():
                continue
            
            # Bu sezon klasöründeki tüm .txt dosyalarını bul
            txt_files = list(season_dir.glob("*.txt"))
            
            for txt_file in txt_files:
                file_name = txt_file.name
                league_name = None
                
                # Dosya adına göre lig ismini bul
                if master_dir_name in league_mapping:
                    for pattern, league in league_mapping[master_dir_name].items():
                        if pattern in file_name.lower():
                            league_name = league
                            break
                
                # Eğer mapping'de yoksa, dosya adından tahmin et
                if not league_name:
                    if "premier" in file_name.lower():
                        league_name = "Premier League"
                    elif "championship" in file_name.lower():
                        league_name = "Championship"
                    elif "seriea" in file_name.lower() or "serie a" in file_name.lower():
                        league_name = "Serie A"
                    elif "serieb" in file_name.lower() or "serie b" in file_name.lower():
                        league_name = "Serie B"
                    elif "bundesliga" in file_name.lower() and "2" not in file_name:
                        league_name = "Bundesliga"
                    elif "liga" in file_name.lower() and "2" not in file_name:
                        league_name = "La Liga"
                    else:
                        continue  # Bilinmeyen dosya, atla
                
                print(f"  {season}: {file_name} -> {league_name}")
                yeni, var, tarih_dis, toplam = parser.parse_file(txt_file, league_name, season)
                
                toplam_istatistik["toplam_okunan"] += toplam
                toplam_istatistik["zaten_var"] += var
                toplam_istatistik["tarih_disinda"] += tarih_dis
                toplam_istatistik["yeni_eklenen"] += yeni
                
                if yeni > 0:
                    print(f"    -> {yeni} yeni mac eklendi")
                if var > 0:
                    print(f"    -> {var} mac zaten vardi (atlandi)")
                if tarih_dis > 0:
                    print(f"    -> {tarih_dis} mac tarih araligi disinda (atlandi)")
    
    # 2. DOĞRUDAN DOSYA KLASÖRLERİ (france, portugal, turkey)
    direct_dirs = ["france", "portugal", "turkey"]
    
    for dir_name in direct_dirs:
        dir_path = football_data_dir / dir_name
        if not dir_path.exists():
            continue
        
        print(f"\n{dir_name.upper()}:")
        print("-" * 80)
        
        # Tüm .txt ve .TXT dosyalarını bul
        txt_files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.TXT"))
        
        for txt_file in txt_files:
            file_name = txt_file.name
            
            # Sezon ve lig bilgisini dosya adından çıkar
            # Örnek: 2021-22_fr1.txt, 2023-24_tr1.txt, 2021-2022_tr.TXT
            season = None
            league_name = None
            
            # Sezon bul (2020-21, 2021-22, 2021-2022, vs.)
            season_match = re.search(r'(\d{4})-(\d{2,4})', file_name)
            if season_match:
                start_year = int(season_match.group(1))
                if 2020 <= start_year <= 2025:
                    season = start_year
            
            if not season:
                continue  # 2021-2025 dışı sezon, atla
            
            # Lig ismini bul
            if dir_name in league_mapping:
                for pattern, league in league_mapping[dir_name].items():
                    if pattern.replace(".txt", "") in file_name.lower():
                        league_name = league
                        break
            
            if not league_name:
                if "fr1" in file_name.lower():
                    league_name = "Ligue 1"
                elif "pt1" in file_name.lower():
                    league_name = "Liga Portugal"
                elif "tr1" in file_name.lower() or "tr" in file_name.lower():
                    league_name = "Süper Lig"
                else:
                    continue  # Bilinmeyen dosya, atla
            
            print(f"  {season}: {file_name} -> {league_name}")
            yeni, var, tarih_dis, toplam = parser.parse_file(txt_file, league_name, season)
            
            toplam_istatistik["toplam_okunan"] += toplam
            toplam_istatistik["zaten_var"] += var
            toplam_istatistik["tarih_disinda"] += tarih_dis
            toplam_istatistik["yeni_eklenen"] += yeni
            
            if yeni > 0:
                print(f"    -> {yeni} yeni mac eklendi")
            if var > 0:
                print(f"    -> {var} mac zaten vardi (atlandi)")
            if tarih_dis > 0:
                print(f"    -> {tarih_dis} mac tarih araligi disinda (atlandi)")
    
    print("\n" + "=" * 80)
    print("YUKLEME ISTATISTIKLERI")
    print("=" * 80)
    print(f"Toplam okunan mac: {toplam_istatistik['toplam_okunan']}")
    print(f"Zaten var (atlandi): {toplam_istatistik['zaten_var']}")
    print(f"Tarih araligi disinda (atlandi): {toplam_istatistik['tarih_disinda']}")
    print(f"Yeni eklenen mac: {toplam_istatistik['yeni_eklenen']}")
    print("=" * 80)
    print("\n✅ Yükleme tamamlandı!")
    print("💡 Script idempotent: Tekrar çalıştırıldığında sadece yeni maçlar eklenecek.")

if __name__ == "__main__":
    load_from_data_files()
