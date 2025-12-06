"""
Takım isimlerindeki hataları kontrol eder
"""
import sys
import io
from pathlib import Path
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import extract
from src.db.connection import get_session
from src.db.schema import Team, League
from src.db.repositories import LeagueRepository

session = get_session()

try:
    print("=" * 80)
    print("🔍 TAKIM İSİMLERİ KONTROLÜ")
    print("=" * 80)
    
    # Serie A'yı bul
    serie_a = LeagueRepository.get_by_name(session, "Serie A")
    if not serie_a:
        print("❌ Serie A bulunamadı!")
        exit(1)
    
    teams = session.query(Team).filter(Team.league_id == serie_a.id).all()
    
    print(f"\n📊 Serie A'da toplam takım: {len(teams)}")
    
    # Skor içeren takım isimlerini bul
    problematic_teams = []
    for team in teams:
        # Skor pattern'i: (0-1), (1-0), (2-0), vb.
        if re.search(r'\(\d+-\d+\)', team.name):
            problematic_teams.append(team)
    
    print(f"⚠️ Skor içeren takım isimleri: {len(problematic_teams)}")
    
    if problematic_teams:
        print("\n📋 Problemli takım isimleri:")
        print(f"{'ID':<6} {'Takım İsmi':<50}")
        print("-" * 60)
        for team in problematic_teams[:20]:  # İlk 20'yi göster
            print(f"{team.id:<6} {team.name[:49]:<50}")
        
        # Temizlenmiş isimleri göster
        print("\n💡 Temizlenmiş isimler (örnek):")
        for team in problematic_teams[:10]:
            clean_name = re.sub(r'\(\d+-\d+\)\s*', '', team.name).strip()
            print(f"   '{team.name}' -> '{clean_name}'")
    
    # CSV'deki takım isimleriyle karşılaştır
    print("\n" + "=" * 80)
    print("📂 CSV'DEKİ TAKIM İSİMLERİ")
    print("=" * 80)
    
    import csv
    odds_dir = project_root / "odds" / "italy"
    csv_files = list(odds_dir.glob("I1*.csv"))
    
    csv_teams = set()
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    home = row.get('HomeTeam', '').strip()
                    away = row.get('AwayTeam', '').strip()
                    if home:
                        csv_teams.add(home)
                    if away:
                        csv_teams.add(away)
        except:
            pass
    
    print(f"\n📊 CSV'deki benzersiz takım sayısı: {len(csv_teams)}")
    print("\n📋 CSV'deki takımlar:")
    for team in sorted(csv_teams)[:30]:
        print(f"   - {team}")
    
    # DB'deki temiz takım isimleri
    db_clean_teams = set()
    for team in teams:
        clean_name = re.sub(r'\(\d+-\d+\)\s*', '', team.name).strip()
        db_clean_teams.add(clean_name)
    
    print(f"\n📊 DB'deki temiz takım sayısı: {len(db_clean_teams)}")
    
    # Eşleşen ve eşleşmeyen takımlar
    matching = csv_teams & db_clean_teams
    csv_only = csv_teams - db_clean_teams
    db_only = db_clean_teams - csv_teams
    
    print(f"\n✅ Eşleşen takımlar: {len(matching)}")
    print(f"⚠️ Sadece CSV'de: {len(csv_only)}")
    print(f"⚠️ Sadece DB'de: {len(db_only)}")
    
    if csv_only:
        print("\n📋 Sadece CSV'de olan takımlar:")
        for team in sorted(csv_only):
            print(f"   - {team}")
    
    if db_only:
        print("\n📋 Sadece DB'de olan takımlar (ilk 20):")
        for team in sorted(list(db_only))[:20]:
            print(f"   - {team}")
    
    print("\n" + "=" * 80)
    print("💡 SORUN TESPİTİ")
    print("=" * 80)
    print("""
❌ PROBLEM: DB'deki takım isimlerinde skor bilgisi var!
   Örnek: "(0-1) FC Internazionale Milano"
   
🔧 ÇÖZÜM: 
   1. Takım isimlerini temizlemek gerekiyor
   2. odds_yukle.py'de takım eşleştirmesini geliştirmek gerekiyor
   3. Skor pattern'ini kaldırarak eşleştirme yapılmalı
    """)
    
finally:
    session.close()





