"""
Odds eksik olan maçları analiz eder ve nedenlerini bulur
"""
import sys
import io
from pathlib import Path
from datetime import datetime
import csv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import and_, extract
from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team
from src.db.repositories import LeagueRepository, TeamRepository

session = get_session()

try:
    print("=" * 80)
    print("🔍 SERIE A ODDS EKSİK ANALİZİ")
    print("=" * 80)
    
    # Serie A'yı bul
    serie_a = LeagueRepository.get_by_name(session, "Serie A")
    if not serie_a:
        print("❌ Serie A bulunamadı!")
        exit(1)
    
    # 2020-2022 arası Serie A maçları
    matches = session.query(Match).filter(
        and_(
            Match.league_id == serie_a.id,
            Match.home_score.isnot(None),
            Match.away_score.isnot(None),
            extract('year', Match.match_date) >= 2020,
            extract('year', Match.match_date) <= 2022
        )
    ).all()
    
    print(f"\n📊 Toplam Serie A maçı (2020-2022): {len(matches)}")
    
    # Odds'ı olan ve olmayan maçlar
    matches_with_odds = []
    matches_without_odds = []
    
    for match in matches:
        odds = session.query(MatchOdds).filter(MatchOdds.match_id == match.id).first()
        if odds:
            matches_with_odds.append(match)
        else:
            matches_without_odds.append(match)
    
    print(f"✅ Odds'ı olan: {len(matches_with_odds)}")
    print(f"❌ Odds'ı olmayan: {len(matches_without_odds)}")
    
    # CSV dosyalarını kontrol et
    print("\n" + "=" * 80)
    print("📂 CSV DOSYALARINDAKİ MAÇLAR")
    print("=" * 80)
    
    odds_dir = project_root / "odds" / "italy"
    csv_files = list(odds_dir.glob("I1*.csv"))
    
    csv_matches = {}
    total_csv_rows = 0
    
    for csv_file in csv_files:
        print(f"\n📄 {csv_file.name} kontrol ediliyor...")
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_csv_rows += 1
                    date_str = row.get('Date', '')
                    home_team = row.get('HomeTeam', '').strip()
                    away_team = row.get('AwayTeam', '').strip()
                    
                    if date_str and home_team and away_team:
                        key = f"{date_str}|{home_team}|{away_team}"
                        if key not in csv_matches:
                            csv_matches[key] = {
                                'date': date_str,
                                'home': home_team,
                                'away': away_team,
                                'file': csv_file.name
                            }
        except Exception as e:
            print(f"   ⚠️ Hata: {e}")
    
    print(f"\n📊 CSV'de toplam satır: {total_csv_rows}")
    print(f"📊 CSV'de benzersiz maç: {len(csv_matches)}")
    
    # Eşleşmeyen maçları analiz et
    print("\n" + "=" * 80)
    print("🔍 EŞLEŞMEYEN MAÇLAR ANALİZİ")
    print("=" * 80)
    
    # Örnek eşleşmeyen maçları göster
    print(f"\n📋 İlk 10 eşleşmeyen maç örneği:")
    print(f"{'Tarih':<12} {'Ev Sahibi':<25} {'Deplasman':<25} {'Neden':<30}")
    print("-" * 100)
    
    sample_count = 0
    date_mismatch = 0
    team_mismatch = 0
    not_in_csv = 0
    
    for match in matches_without_odds[:20]:  # İlk 20'yi kontrol et
        home_team = TeamRepository.get_by_id(session, match.home_team_id)
        away_team = TeamRepository.get_by_id(session, match.away_team_id)
        
        if not home_team or not away_team:
            continue
        
        match_date_str = match.match_date.strftime("%d/%m/%Y") if match.match_date else ""
        
        # CSV'de bu maçı ara
        found = False
        reason = ""
        
        for csv_key, csv_data in csv_matches.items():
            csv_date = csv_data['date']
            csv_home = csv_data['home']
            csv_away = csv_data['away']
            
            # Tarih eşleşmesi (toleranslı)
            if csv_date:
                try:
                    # CSV tarihini parse et
                    if "/" in csv_date:
                        parts = csv_date.split("/")
                        if len(parts) == 3:
                            day, month, year = map(int, parts)
                            if year < 100:
                                year += 2000
                            csv_date_obj = datetime(year, month, day)
                            
                            # Match tarihi ile karşılaştır (±1 gün tolerans)
                            from datetime import timedelta
                            if abs((match.match_date.date() - csv_date_obj.date()).days) <= 1:
                                # Tarih eşleşti, takım isimlerini kontrol et
                                if (home_team.name.lower() in csv_home.lower() or 
                                    csv_home.lower() in home_team.name.lower()):
                                    if (away_team.name.lower() in csv_away.lower() or 
                                        csv_away.lower() in away_team.name.lower()):
                                        found = True
                                        reason = "✅ CSV'de var ama eşleşmedi"
                                        break
                                    else:
                                        reason = f"⚠️ Takım eşleşmedi (Away: DB={away_team.name}, CSV={csv_away})"
                                        team_mismatch += 1
                                else:
                                    reason = f"⚠️ Takım eşleşmedi (Home: DB={home_team.name}, CSV={csv_home})"
                                    team_mismatch += 1
                            else:
                                date_mismatch += 1
                except:
                    pass
        
        if not found and not reason:
            reason = "❌ CSV'de bulunamadı"
            not_in_csv += 1
        
        if sample_count < 10:
            print(f"{match_date_str:<12} {home_team.name[:24]:<25} {away_team.name[:24]:<25} {reason[:29]:<30}")
            sample_count += 1
    
    print("\n" + "=" * 80)
    print("📊 EKSİK OLMA SEBEPLERİ (İlk 20 maç analizi)")
    print("=" * 80)
    print(f"❌ CSV'de bulunamadı: {not_in_csv}")
    print(f"⚠️ Takım ismi eşleşmedi: {team_mismatch}")
    print(f"⚠️ Tarih eşleşmedi: {date_mismatch}")
    
    # Takım isim farklılıklarını göster
    print("\n" + "=" * 80)
    print("🏷️ TAKIM İSİM FARKLILIKLARI ÖRNEKLERİ")
    print("=" * 80)
    
    # CSV'deki takım isimlerini topla
    csv_teams = set()
    for csv_data in csv_matches.values():
        csv_teams.add(csv_data['home'])
        csv_teams.add(csv_data['away'])
    
    # DB'deki takım isimlerini topla
    db_teams = session.query(Team).filter(Team.league_id == serie_a.id).all()
    db_team_names = {team.name for team in db_teams}
    
    print(f"\n📊 CSV'deki takım sayısı: {len(csv_teams)}")
    print(f"📊 DB'deki takım sayısı: {len(db_team_names)}")
    
    # Farklı olanları göster
    csv_only = csv_teams - db_team_names
    db_only = db_team_names - csv_teams
    
    if csv_only:
        print(f"\n⚠️ Sadece CSV'de olan takımlar (ilk 10):")
        for team in list(csv_only)[:10]:
            print(f"   - {team}")
    
    if db_only:
        print(f"\n⚠️ Sadece DB'de olan takımlar (ilk 10):")
        for team in list(db_only)[:10]:
            print(f"   - {team}")
    
    print("\n" + "=" * 80)
    print("💡 ÖNERİLER")
    print("=" * 80)
    print("""
1. Takım isimleri farklı olabilir (örn: "AC Milan" vs "Milan")
2. CSV'de bazı maçlar eksik olabilir
3. Tarih formatı farklı olabilir
4. odds_yukle.py scriptindeki eşleştirme algoritması geliştirilebilir
    """)
    
finally:
    session.close()





