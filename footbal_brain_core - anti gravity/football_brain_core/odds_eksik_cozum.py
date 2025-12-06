"""
Odds eksikliklerini analiz eder ve çözüm üretir
Eksiksiz odds verisi için geliştirilmiş yükleme scripti
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import csv
import logging
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import and_, extract, or_
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team
from src.db.repositories import MatchRepository, LeagueRepository, TeamRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_odds_coverage():
    """Odds kapsamını analiz eder ve eksiklikleri raporlar"""
    session = get_session()
    
    try:
        print("=" * 80)
        print("📊 ODDS KAPSAM ANALİZİ")
        print("=" * 80)
        print()
        
        # Genel durum
        total_matches = session.query(Match).filter(
            and_(
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025
            )
        ).count()
        
        matches_with_odds = session.query(Match).join(MatchOdds).filter(
            and_(
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025
            )
        ).count()
        
        percentage = (matches_with_odds / total_matches * 100) if total_matches > 0 else 0
        
        print(f"📊 Genel Durum:")
        print(f"   Toplam maç: {total_matches:,}")
        print(f"   Odds'ı olan: {matches_with_odds:,}")
        print(f"   Odds'ı olmayan: {total_matches - matches_with_odds:,}")
        print(f"   Kapsam: {percentage:.1f}%")
        print()
        
        # Lig bazında detaylı analiz
        leagues = session.query(League).order_by(League.name).all()
        
        league_analysis = []
        
        for league in leagues:
            league_matches = session.query(Match).filter(
                and_(
                    Match.league_id == league.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025
                )
            ).count()
            
            league_matches_with_odds = session.query(Match).join(MatchOdds).filter(
                and_(
                    Match.league_id == league.id,
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025
                )
            ).count()
            
            if league_matches > 0:
                league_pct = (league_matches_with_odds / league_matches) * 100
                league_analysis.append({
                    'league': league,
                    'name': league.name,
                    'total': league_matches,
                    'with_odds': league_matches_with_odds,
                    'without_odds': league_matches - league_matches_with_odds,
                    'percentage': league_pct
                })
        
        # Eksik ligleri bul
        missing_leagues = [l for l in league_analysis if l['percentage'] < 80]
        missing_leagues.sort(key=lambda x: x['percentage'])
        
        print("=" * 80)
        print("❌ EKSİK LİGLER (Kapsam < 80%)")
        print("=" * 80)
        print(f"{'Lig':<30} {'Toplam':<10} {'Odds Var':<12} {'Odds Yok':<12} {'%':<10}")
        print("-" * 80)
        
        for stat in missing_leagues:
            print(f"{stat['name']:<30} {stat['total']:<10} {stat['with_odds']:<12} {stat['without_odds']:<12} {stat['percentage']:.1f}%")
        
        print()
        print("=" * 80)
        print("✅ YETERLİ LİGLER (Kapsam >= 80%)")
        print("=" * 80)
        
        good_leagues = [l for l in league_analysis if l['percentage'] >= 80]
        good_leagues.sort(key=lambda x: -x['percentage'])
        
        if good_leagues:
            print(f"{'Lig':<30} {'Toplam':<10} {'Odds Var':<12} {'Odds Yok':<12} {'%':<10}")
            print("-" * 80)
            for stat in good_leagues:
                print(f"{stat['name']:<30} {stat['total']:<10} {stat['with_odds']:<12} {stat['without_odds']:<12} {stat['percentage']:.1f}%")
        else:
            print("⚠️ Hiçbir ligde yeterli kapsam yok!")
        
        return {
            'total_matches': total_matches,
            'matches_with_odds': matches_with_odds,
            'percentage': percentage,
            'missing_leagues': missing_leagues,
            'good_leagues': good_leagues
        }
        
    finally:
        session.close()


def analyze_matching_issues(league_name: str, sample_size: int = 50):
    """Belirli bir lig için eşleştirme sorunlarını analiz eder"""
    session = get_session()
    
    try:
        league = LeagueRepository.get_by_name(session, league_name)
        if not league:
            logger.error(f"Lig bulunamadı: {league_name}")
            return
        
        print(f"\n{'=' * 80}")
        print(f"🔍 {league_name} EŞLEŞTİRME SORUNLARI ANALİZİ")
        print(f"{'=' * 80}")
        
        # Odds'ı olmayan maçları bul
        matches_without_odds = session.query(Match).filter(
            and_(
                Match.league_id == league.id,
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
                extract('year', Match.match_date) >= 2020,
                extract('year', Match.match_date) <= 2025
            )
        ).outerjoin(MatchOdds).filter(MatchOdds.id.is_(None)).limit(sample_size).all()
        
        print(f"\n📊 Analiz edilen maç sayısı: {len(matches_without_odds)}")
        
        # CSV dosyalarını kontrol et
        league_mapping = {
            "Premier League": "england",
            "Championship": "england",
            "League One": "england",
            "League Two": "england",
            "Serie A": "italy",
            "Serie B": "italy",
            "Bundesliga": "bundesliga",
            "2. Bundesliga": "bundesliga",
            "Ligue 1": "france",
            "Ligue 2": "france",
            "Liga Portugal": "portugal",
            "Liga Portugal 2": "portugal",
            "Süper Lig": "turkey",
            "La Liga": "espana",
            "Segunda División": "espana",
        }
        
        csv_folder = league_mapping.get(league_name)
        if not csv_folder:
            print(f"⚠️ {league_name} için CSV klasörü bulunamadı")
            return
        
        odds_dir = project_root / "odds" / csv_folder
        if not odds_dir.exists():
            print(f"⚠️ Odds klasörü bulunamadı: {odds_dir}")
            return
        
        # CSV dosyalarındaki maçları topla
        csv_files = list(odds_dir.glob("*.csv"))
        csv_matches = {}
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        date_str = row.get('Date', '').strip()
                        home_team = row.get('HomeTeam', '').strip()
                        away_team = row.get('AwayTeam', '').strip()
                        
                        if date_str and home_team and away_team:
                            key = f"{date_str}|{home_team}|{away_team}"
                            csv_matches[key] = {
                                'date': date_str,
                                'home': home_team,
                                'away': away_team,
                                'file': csv_file.name
                            }
            except Exception as e:
                logger.debug(f"CSV okuma hatası ({csv_file}): {e}")
        
        print(f"📂 CSV'de bulunan maç sayısı: {len(csv_matches)}")
        
        # Eşleşmeyen maçları analiz et
        issues = {
            'date_mismatch': 0,
            'team_name_mismatch': 0,
            'not_in_csv': 0,
            'found_in_csv': 0
        }
        
        print(f"\n📋 Örnek eşleşmeyen maçlar (ilk 20):")
        print(f"{'Tarih':<12} {'Ev Sahibi (DB)':<25} {'Deplasman (DB)':<25} {'Durum':<30}")
        print("-" * 100)
        
        for match in matches_without_odds[:20]:
            home_team = TeamRepository.get_by_id(session, match.home_team_id)
            away_team = TeamRepository.get_by_id(session, match.away_team_id)
            
            if not home_team or not away_team:
                continue
            
            match_date_str = match.match_date.strftime("%d/%m/%Y") if match.match_date else ""
            
            # CSV'de ara
            found = False
            reason = ""
            
            for csv_key, csv_data in csv_matches.items():
                csv_date = csv_data['date']
                csv_home = csv_data['home']
                csv_away = csv_data['away']
                
                # Tarih kontrolü
                try:
                    if "/" in csv_date:
                        parts = csv_date.split("/")
                        if len(parts) == 3:
                            day, month, year = map(int, parts)
                            if year < 100:
                                year += 2000
                            csv_date_obj = datetime(year, month, day)
                            
                            # ±3 gün tolerans
                            if abs((match.match_date.date() - csv_date_obj.date()).days) <= 3:
                                # Takım isimleri kontrolü (esnek)
                                home_match = (
                                    home_team.name.lower() in csv_home.lower() or
                                    csv_home.lower() in home_team.name.lower() or
                                    home_team.name.lower().replace('fc ', '').replace(' ac', '') in csv_home.lower() or
                                    csv_home.lower() in home_team.name.lower().replace('fc ', '').replace(' ac', '')
                                )
                                
                                away_match = (
                                    away_team.name.lower() in csv_away.lower() or
                                    csv_away.lower() in away_team.name.lower() or
                                    away_team.name.lower().replace('fc ', '').replace(' ac', '') in csv_away.lower() or
                                    csv_away.lower() in away_team.name.lower().replace('fc ', '').replace(' ac', '')
                                )
                                
                                if home_match and away_match:
                                    found = True
                                    issues['found_in_csv'] += 1
                                    reason = "✅ CSV'de var (eşleştirme sorunu)"
                                    break
                                elif home_match or away_match:
                                    issues['team_name_mismatch'] += 1
                                    reason = f"⚠️ Takım eşleşmedi (H:{home_match}, A:{away_match})"
                                else:
                                    issues['date_mismatch'] += 1
                                    reason = "⚠️ Tarih eşleşti ama takımlar farklı"
                except:
                    pass
            
            if not found and not reason:
                issues['not_in_csv'] += 1
                reason = "❌ CSV'de bulunamadı"
            
            print(f"{match_date_str:<12} {home_team.name[:24]:<25} {away_team.name[:24]:<25} {reason[:29]:<30}")
        
        print(f"\n📊 Sorun Analizi:")
        print(f"   CSV'de bulunan ama eşleşmeyen: {issues['found_in_csv']}")
        print(f"   Takım ismi eşleşmedi: {issues['team_name_mismatch']}")
        print(f"   Tarih eşleşti ama takımlar farklı: {issues['date_mismatch']}")
        print(f"   CSV'de bulunamadı: {issues['not_in_csv']}")
        
    finally:
        session.close()


def main():
    """Ana analiz fonksiyonu"""
    print("=" * 80)
    print("🔍 ODDS EKSİKLİK ANALİZİ VE ÇÖZÜM")
    print("=" * 80)
    print()
    
    # Genel kapsam analizi
    analysis = analyze_odds_coverage()
    
    print()
    print("=" * 80)
    print("💡 ÇÖZÜM ÖNERİLERİ")
    print("=" * 80)
    print()
    
    if analysis['percentage'] < 50:
        print("❌ CRİTİK: Odds kapsamı %50'nin altında!")
        print()
        print("🔧 Yapılacaklar:")
        print("   1. Takım ismi eşleştirme algoritması geliştirilmeli")
        print("   2. Tarih toleransı artırılmalı (±3 gün)")
        print("   3. CSV dosyaları kontrol edilmeli")
        print("   4. Eksik ligler için alternatif kaynaklar bulunmalı")
        print()
        
        # En eksik 3 ligi detaylı analiz et
        print("📋 En eksik 3 lig detaylı analiz ediliyor...")
        for league_stat in analysis['missing_leagues'][:3]:
            analyze_matching_issues(league_stat['name'], sample_size=30)
    
    print()
    print("=" * 80)
    print("✅ Analiz tamamlandı!")
    print("=" * 80)


if __name__ == "__main__":
    main()





