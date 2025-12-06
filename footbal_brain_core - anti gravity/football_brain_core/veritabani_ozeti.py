"""
Veritabanı İçeriği Özeti
Bu script veritabanında hangi verilerin olduğunu gösterir
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

try:
    from football_brain_core.src.db.connection import get_session
    from football_brain_core.src.db.schema import (
        League, Team, Match, Stat, Market, Prediction, 
        Result, ModelVersion, Explanation, Experiment
    )
    
    print("=" * 80)
    print("VERITABANI ICERIK OZETI")
    print("=" * 80)
    
    session = get_session()
    
    try:
        # 1. LİGLER
        print("\n1. LİGLER (Leagues)")
        print("-" * 80)
        leagues = session.query(League).all()
        print(f"Toplam lig sayisi: {len(leagues)}")
        if leagues:
            print("\nIlk 10 lig:")
            for i, league in enumerate(leagues[:10], 1):
                print(f"  {i}. {league.name} ({league.country or 'N/A'}) - ID: {league.id}")
        
        # 2. TAKIMLAR
        print("\n2. TAKIMLAR (Teams)")
        print("-" * 80)
        teams = session.query(Team).all()
        print(f"Toplam takim sayisi: {len(teams)}")
        if teams:
            print("\nIlk 10 takim:")
            for i, team in enumerate(teams[:10], 1):
                league_name = team.league.name if team.league else "N/A"
                print(f"  {i}. {team.name} (Lig: {league_name}) - ID: {team.id}")
        
        # 3. MAÇLAR
        print("\n3. MACLAR (Matches)")
        print("-" * 80)
        matches = session.query(Match).all()
        print(f"Toplam mac sayisi: {len(matches)}")
        if matches:
            print("\nIlk 10 mac:")
            for i, match in enumerate(matches[:10], 1):
                home = match.home_team.name if match.home_team else "N/A"
                away = match.away_team.name if match.away_team else "N/A"
                score = f"{match.home_score or '?'}-{match.away_score or '?'}" if match.home_score is not None else "Henuz oynanmadi"
                date_str = match.match_date.strftime("%Y-%m-%d") if match.match_date else "N/A"
                print(f"  {i}. {home} vs {away} | Skor: {score} | Tarih: {date_str}")
            
            # Tarih aralığı
            dates = [m.match_date for m in matches if m.match_date]
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                print(f"\nTarih araligi: {min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}")
        
        # 4. İSTATİSTİKLER
        print("\n4. ISTATISTIKLER (Stats)")
        print("-" * 80)
        stats = session.query(Stat).all()
        print(f"Toplam istatistik kaydi: {len(stats)}")
        if stats:
            stat_types = {}
            for stat in stats:
                stat_type = stat.stat_type
                stat_types[stat_type] = stat_types.get(stat_type, 0) + 1
            print("\nIstatistik turleri:")
            for stat_type, count in sorted(stat_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {stat_type}: {count} kayit")
        
        # 5. MARKETLER
        print("\n5. MARKETLER (Markets - Bahis Piyasalari)")
        print("-" * 80)
        markets = session.query(Market).all()
        print(f"Toplam market sayisi: {len(markets)}")
        if markets:
            print("\nMarketler:")
            for market in markets:
                print(f"  - {market.name}: {market.description or 'Aciklama yok'}")
        
        # 6. TAHMİNLER
        print("\n6. TAHMINLER (Predictions)")
        print("-" * 80)
        predictions = session.query(Prediction).all()
        print(f"Toplam tahmin sayisi: {len(predictions)}")
        if predictions:
            print("\nIlk 5 tahmin:")
            for i, pred in enumerate(predictions[:5], 1):
                match = pred.match
                home = match.home_team.name if match.home_team else "N/A"
                away = match.away_team.name if match.away_team else "N/A"
                market = pred.market.name if pred.market else "N/A"
                print(f"  {i}. {home} vs {away} | Market: {market} | Tahmin: {pred.predicted_outcome} | Olasilik: {pred.p_hat or 'N/A'}")
        
        # 7. SONUÇLAR
        print("\n7. SONUCLAR (Results - Gercek Sonuclar)")
        print("-" * 80)
        results = session.query(Result).all()
        print(f"Toplam sonuc sayisi: {len(results)}")
        
        # 8. MODEL VERSİYONLARI
        print("\n8. MODEL VERSIYONLARI (Model Versions)")
        print("-" * 80)
        model_versions = session.query(ModelVersion).all()
        print(f"Toplam model versiyonu: {len(model_versions)}")
        if model_versions:
            for mv in model_versions:
                status = "Aktif" if mv.is_active else "Pasif"
                print(f"  - {mv.version}: {mv.description or 'Aciklama yok'} ({status})")
        
        # 9. AÇIKLAMALAR
        print("\n9. ACIKLAMALAR (Explanations - LLM Ciktilari)")
        print("-" * 80)
        explanations = session.query(Explanation).all()
        print(f"Toplam aciklama sayisi: {len(explanations)}")
        
        # 10. DENEYLER
        print("\n10. DENEYLER (Experiments)")
        print("-" * 80)
        experiments = session.query(Experiment).all()
        print(f"Toplam deney sayisi: {len(experiments)}")
        
        # ÖZET
        print("\n" + "=" * 80)
        print("OZET")
        print("=" * 80)
        print(f"Ligler: {len(leagues)}")
        print(f"Takimlar: {len(teams)}")
        print(f"Maclar: {len(matches)}")
        print(f"Istatistikler: {len(stats)}")
        print(f"Marketler: {len(markets)}")
        print(f"Tahminler: {len(predictions)}")
        print(f"Sonuclar: {len(results)}")
        print(f"Model Versiyonlari: {len(model_versions)}")
        print(f"Aciklamalar: {len(explanations)}")
        print(f"Deneyler: {len(experiments)}")
        
        # Veri kaynağı bilgisi
        print("\n" + "=" * 80)
        print("VERI KAYNAGI")
        print("=" * 80)
        print("Bu veriler API-FOOTBALL'dan cekilmistir:")
        print("  - Lig bilgileri (Premier League, La Liga, Serie A, vs.)")
        print("  - Takim bilgileri")
        print("  - Mac fiksturleri ve sonuclari")
        print("  - Mac istatistikleri (gol, pas, top kontrolu, vs.)")
        print("  - Bahis piyasalari (Match Result, BTTS, Over/Under, vs.)")
        print("\nTahminler ve aciklamalar model tarafindan uretilmistir.")
        
    finally:
        session.close()
        
except ImportError as e:
    print(f"[HATA] Import hatasi: {e}")
    print("Veritabani modulleri bulunamadi.")
except Exception as e:
    print(f"[HATA] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)






