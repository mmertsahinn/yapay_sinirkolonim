"""
CSV dosyalarındaki odds satır sayısını veritabanındaki maç sayısıyla karşılaştırır
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import csv
from collections import defaultdict

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import and_, extract
from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League
from src.db.repositories import LeagueRepository

def analyze_csv_vs_db():
    """CSV dosyalarındaki satır sayısını DB'deki maç sayısıyla karşılaştırır"""
    session = get_session()
    
    try:
        print("=" * 80)
        print("📊 CSV vs VERİTABANI KARŞILAŞTIRMASI")
        print("=" * 80)
        print()
        
        # Lig kodları ve klasör eşleştirmesi
        LEAGUE_MAPPING = {
            "E0": ("england", "Premier League"),
            "E1": ("england", "Championship"),
            "E2": ("england", "League One"),
            "E3": ("england", "League Two"),
            "I1": ("italy", "Serie A"),
            "I2": ("italy", "Serie B"),
            "D1": ("bundesliga", "Bundesliga"),
            "D2": ("bundesliga", "2. Bundesliga"),
            "F1": ("france", "Ligue 1"),
            "F2": ("france", "Ligue 2"),
            "P1": ("portugal", "Liga Portugal"),
            "P2": ("portugal", "Liga Portugal 2"),
            "T1": ("turkey", "Süper Lig"),
            "SP1": ("espana", "La Liga"),
            "SP2": ("espana", "Segunda División"),
        }
        
        odds_dir = project_root / "odds"
        
        if not odds_dir.exists():
            print(f"❌ Odds klasörü bulunamadı: {odds_dir}")
            return
        
        # CSV dosyalarını tara
        csv_stats = defaultdict(lambda: {
            'total_rows': 0,
            'valid_rows': 0,
            'files': []
        })
        
        print("📂 CSV dosyaları taranıyor...")
        print()
        
        for league_folder in odds_dir.iterdir():
            if not league_folder.is_dir():
                continue
            
            csv_files = list(league_folder.glob("*.csv"))
            
            for csv_file in csv_files:
                try:
                    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                        reader = csv.DictReader(f)
                        row_count = 0
                        valid_count = 0
                        
                        for row in reader:
                            row_count += 1
                            # Geçerli satır kontrolü (tarih ve takım isimleri var mı?)
                            if row.get('Date') and row.get('HomeTeam') and row.get('AwayTeam'):
                                valid_count += 1
                        
                        # Dosya adından lig kodunu bul
                        file_name = csv_file.name.upper()
                        league_code = None
                        league_folder_name = league_folder.name.lower()
                        
                        for code, (folder, _) in LEAGUE_MAPPING.items():
                            if folder.lower() == league_folder_name:
                                # Dosya adında lig kodu var mı kontrol et
                                if code.replace('SP', 'SP').replace('D', 'D').replace('E', 'E').replace('I', 'I').replace('F', 'F').replace('P', 'P').replace('T', 'T') in file_name:
                                    league_code = code
                                    break
                        
                        # Eğer dosya adından bulunamazsa, klasör adına göre varsayılan al
                        if not league_code:
                            for code, (folder, _) in LEAGUE_MAPPING.items():
                                if folder.lower() == league_folder_name:
                                    # İlk eşleşen lig kodunu al (daha iyi eşleştirme için dosya adına bakılabilir)
                                    if not any(code in f.upper() for f in csv_stats[code]['files']):
                                        league_code = code
                                        break
                        
                        # Hala bulunamazsa, klasör adına göre tahmin et
                        if not league_code:
                            folder_to_code = {
                                'england': 'E0',
                                'italy': 'I1',
                                'bundesliga': 'D1',
                                'france': 'F1',
                                'portugal': 'P1',
                                'turkey': 'T1',
                                'espana': 'SP1'
                            }
                            league_code = folder_to_code.get(league_folder_name, None)
                        
                        if league_code:
                            csv_stats[league_code]['total_rows'] += row_count
                            csv_stats[league_code]['valid_rows'] += valid_count
                            csv_stats[league_code]['files'].append(csv_file.name)
                
                except Exception as e:
                    print(f"⚠️ Hata ({csv_file.name}): {e}")
        
        # Veritabanındaki maç sayılarını al
        print("📊 Veritabanı maç sayıları alınıyor...")
        print()
        
        db_stats = {}
        for code, (folder, league_name) in LEAGUE_MAPPING.items():
            league = LeagueRepository.get_by_name(session, league_name)
            if league:
                total_matches = session.query(Match).filter(
                    and_(
                        Match.league_id == league.id,
                        Match.home_score.isnot(None),
                        Match.away_score.isnot(None),
                        extract('year', Match.match_date) >= 2020,
                        extract('year', Match.match_date) <= 2025
                    )
                ).count()
                
                matches_with_odds = session.query(Match).join(MatchOdds).filter(
                    and_(
                        Match.league_id == league.id,
                        Match.home_score.isnot(None),
                        Match.away_score.isnot(None),
                        extract('year', Match.match_date) >= 2020,
                        extract('year', Match.match_date) <= 2025
                    )
                ).count()
                
                db_stats[code] = {
                    'league_name': league_name,
                    'total_matches': total_matches,
                    'matches_with_odds': matches_with_odds,
                    'matches_without_odds': total_matches - matches_with_odds
                }
        
        # Karşılaştırma tablosu
        print("=" * 100)
        print(f"{'Lig':<30} {'CSV Satır':<12} {'CSV Geçerli':<12} {'DB Maç':<12} {'DB Odds Var':<12} {'DB Odds Yok':<12} {'Eşleşme %':<12}")
        print("-" * 100)
        
        total_csv_rows = 0
        total_csv_valid = 0
        total_db_matches = 0
        total_db_with_odds = 0
        
        for code in sorted(LEAGUE_MAPPING.keys()):
            league_name = LEAGUE_MAPPING[code][1]
            csv_data = csv_stats.get(code, {'total_rows': 0, 'valid_rows': 0})
            db_data = db_stats.get(code, {'total_matches': 0, 'matches_with_odds': 0})
            
            csv_rows = csv_data['total_rows']
            csv_valid = csv_data['valid_rows']
            db_matches = db_data['total_matches']
            db_with_odds = db_data['matches_with_odds']
            
            # Eşleşme yüzdesi (CSV'deki geçerli satırların DB'deki maçlara oranı)
            if db_matches > 0:
                match_percentage = (csv_valid / db_matches * 100) if db_matches > 0 else 0
            else:
                match_percentage = 0
            
            total_csv_rows += csv_rows
            total_csv_valid += csv_valid
            total_db_matches += db_matches
            total_db_with_odds += db_with_odds
            
            status = "✅" if match_percentage >= 80 else "⚠️" if match_percentage >= 50 else "❌"
            
            print(f"{status} {league_name:<28} {csv_rows:<12} {csv_valid:<12} {db_matches:<12} {db_with_odds:<12} {db_matches - db_with_odds:<12} {match_percentage:>10.1f}%")
        
        print("-" * 100)
        print(f"{'TOPLAM':<30} {total_csv_rows:<12} {total_csv_valid:<12} {total_db_matches:<12} {total_db_with_odds:<12} {total_db_matches - total_db_with_odds:<12} {(total_csv_valid / total_db_matches * 100) if total_db_matches > 0 else 0:>10.1f}%")
        
        print()
        print("=" * 100)
        print("📊 ÖZET")
        print("=" * 100)
        print(f"CSV Toplam Satır: {total_csv_rows:,}")
        print(f"CSV Geçerli Satır: {total_csv_valid:,}")
        print(f"DB Toplam Maç: {total_db_matches:,}")
        print(f"DB Odds'ı Olan Maç: {total_db_with_odds:,}")
        print(f"DB Odds'ı Olmayan Maç: {total_db_matches - total_db_with_odds:,}")
        print()
        
        if total_db_matches > 0:
            csv_to_db_ratio = (total_csv_valid / total_db_matches) * 100
            odds_coverage = (total_db_with_odds / total_db_matches) * 100
            print(f"CSV/DB Oranı: {csv_to_db_ratio:.1f}% (CSV'deki geçerli satırların DB maçlarına oranı)")
            print(f"Odds Kapsamı: {odds_coverage:.1f}% (DB'deki maçların odds'a sahip olma oranı)")
            print()
            
            if csv_to_db_ratio > 100:
                print(f"💡 CSV'de DB'den fazla satır var! ({csv_to_db_ratio - 100:.1f}% fazla)")
                print("   Bu normal olabilir çünkü:")
                print("   - CSV'de gelecek maçlar olabilir")
                print("   - CSV'de sonuçsuz maçlar olabilir")
                print("   - Bazı maçlar DB'ye yüklenmemiş olabilir")
            elif csv_to_db_ratio < 80:
                print(f"⚠️ CSV'de DB'den az satır var! ({100 - csv_to_db_ratio:.1f}% eksik)")
                print("   Olası nedenler:")
                print("   - CSV dosyaları eksik olabilir")
                print("   - Bazı sezonlar CSV'de yok")
            else:
                print("✅ CSV ve DB sayıları uyumlu görünüyor")
            
            print()
            if odds_coverage < 50:
                print(f"❌ Odds kapsamı düşük! ({odds_coverage:.1f}%)")
                print("   Olası nedenler:")
                print("   - Takım isimleri eşleşmiyor")
                print("   - Tarih eşleşmesi sorunları")
                print("   - Eşleştirme algoritması yetersiz")
                print()
                print("🔧 Çözüm: odds_yukle.py scriptini tekrar çalıştırın veya")
                print("   eşleştirme algoritmasını iyileştirin")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    analyze_csv_vs_db()





