"""
Database'den bir örnek maç getirip tüm bilgilerini gösterir
"""
import sys
from pathlib import Path
from datetime import datetime
import json

# Project root'u path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.db.connection import get_session
from src.db.schema import (
    Match, MatchOdds, League, Team, Stat, Market, Prediction, 
    Result, Explanation, ModelVersion
)
from sqlalchemy import and_, extract

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def format_value(value):
    """Değeri güzel formatta göster"""
    if value is None:
        return "❌ Yok"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def show_match_details(match_id: int = None):
    """Bir maçın tüm detaylarını gösterir"""
    session = get_session()
    
    try:
        # Maç seç (hype bilgisi olan bir maç)
        if match_id:
            match = session.query(Match).filter(Match.id == match_id).first()
        else:
            # 2020-2025 arası, skoru ve hype bilgisi olan bir maç seç
            match = session.query(Match).filter(
                and_(
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025,
                    Match.hype_updated_at.isnot(None)  # Hype bilgisi olan
                )
            ).first()
            
            # Eğer hype bilgisi olan maç yoksa, hype bilgisi olmayan bir maç seç
            if not match:
                match = session.query(Match).filter(
                    and_(
                        Match.home_score.isnot(None),
                        Match.away_score.isnot(None),
                        extract('year', Match.match_date) >= 2020,
                        extract('year', Match.match_date) <= 2025
                    )
                ).first()
        
        if not match:
            print("❌ Maç bulunamadı!")
            return
        
        print("=" * 80)
        print("📊 ÖRNEK MAÇ DETAYLARI")
        print("=" * 80)
        print()
        
        # 1. TEMEL MAÇ BİLGİLERİ
        print("🏟️  TEMEL MAÇ BİLGİLERİ")
        print("-" * 80)
        print(f"Match ID: {match.id}")
        print(f"Match ID (Unique): {match.match_id or 'Yok'}")
        print(f"Lig: {match.league.name if match.league else 'Bilinmiyor'}")
        print(f"Tarih: {match.match_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ev Sahibi: {match.home_team.name if match.home_team else 'Bilinmiyor'}")
        print(f"Deplasman: {match.away_team.name if match.away_team else 'Bilinmiyor'}")
        print(f"Skor: {match.home_score or '?'} - {match.away_score or '?'}")
        print(f"Durum: {match.status or 'Tamamlandı'}")
        print(f"Oluşturulma: {match.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Güncellenme: {match.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 2. HYPE BİLGİLERİ
        print("📱 HYPE BİLGİLERİ (Sosyal Medya)")
        print("-" * 80)
        home_support_pct = f"{match.home_support*100:.2f}%" if match.home_support is not None else "N/A"
        away_support_pct = f"{match.away_support*100:.2f}%" if match.away_support is not None else "N/A"
        print(f"Home Support: {format_value(match.home_support)} ({home_support_pct})")
        print(f"Away Support: {format_value(match.away_support)} ({away_support_pct})")
        print(f"Sentiment Score: {format_value(match.sentiment_score)}")
        print(f"Total Tweets: {format_value(match.total_tweets)}")
        print(f"Hype Güncellenme: {format_value(match.hype_updated_at)}")
        print()
        
        # 3. ODDS BİLGİLERİ
        print("🎲 ODDS BİLGİLERİ (Bahis Oranları)")
        print("-" * 80)
        odds = None
        try:
            odds = session.query(MatchOdds).filter(MatchOdds.match_id == match.id).first()
        except Exception as e:
            print(f"⚠️  Odds tablosu henüz oluşturulmamış: {type(e).__name__}")
        
        if odds:
            print("✅ Odds bilgileri mevcut!")
            print()
            print("Bet365 Odds:")
            print(f"  Home: {format_value(odds.b365_h)}")
            print(f"  Draw: {format_value(odds.b365_d)}")
            print(f"  Away: {format_value(odds.b365_a)}")
            print()
            print("Pinnacle Odds:")
            print(f"  Home: {format_value(odds.p_h)}")
            print(f"  Draw: {format_value(odds.p_d)}")
            print(f"  Away: {format_value(odds.p_a)}")
            print()
            print("William Hill Odds:")
            print(f"  Home: {format_value(odds.wh_h)}")
            print(f"  Draw: {format_value(odds.wh_d)}")
            print(f"  Away: {format_value(odds.wh_a)}")
            print()
            print("Market Averages:")
            print(f"  Avg Home: {format_value(odds.avg_h)}")
            print(f"  Avg Draw: {format_value(odds.avg_d)}")
            print(f"  Avg Away: {format_value(odds.avg_a)}")
            print()
            print("Over/Under 2.5:")
            print(f"  Bet365 Over: {format_value(odds.b365_over_25)}")
            print(f"  Bet365 Under: {format_value(odds.b365_under_25)}")
            print(f"  Market Avg Over: {format_value(odds.avg_over_25)}")
            print(f"  Market Avg Under: {format_value(odds.avg_under_25)}")
            print()
            print(f"Tüm Odds (JSON): {len(str(odds.all_odds)) if odds.all_odds else 0} karakter")
            print(f"Odds Oluşturulma: {odds.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Odds Güncellenme: {odds.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("❌ Odds bilgisi yok")
        print()
        
        # 4. İSTATİSTİKLER
        print("📈 MAÇ İSTATİSTİKLERİ")
        print("-" * 80)
        stats = session.query(Stat).filter(Stat.match_id == match.id).all()
        if stats:
            print(f"✅ {len(stats)} istatistik kaydı var:")
            for stat in stats[:10]:  # İlk 10'unu göster
                team = session.query(Team).filter(Team.id == stat.team_id).first()
                print(f"  - {team.name if team else 'Bilinmiyor'}: {stat.stat_type} = {stat.stat_value}")
            if len(stats) > 10:
                print(f"  ... ve {len(stats) - 10} tane daha")
        else:
            print("❌ İstatistik bilgisi yok")
        print()
        
        # 5. TAHMİNLER
        print("🔮 MODEL TAHMİNLERİ")
        print("-" * 80)
        predictions = session.query(Prediction).filter(Prediction.match_id == match.id).all()
        if predictions:
            print(f"✅ {len(predictions)} tahmin kaydı var:")
            for pred in predictions:
                market = session.query(Market).filter(Market.id == pred.market_id).first()
                model = session.query(ModelVersion).filter(ModelVersion.id == pred.model_version_id).first()
                print(f"  - Market: {market.name if market else 'Bilinmiyor'}")
                print(f"    Tahmin: {pred.predicted_outcome}")
                print(f"    Olasılık: {format_value(pred.p_hat)}")
                print(f"    Model: {model.version if model else 'Bilinmiyor'}")
                print(f"    Tarih: {pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        else:
            print("❌ Tahmin bilgisi yok")
        print()
        
        # 6. GERÇEK SONUÇLAR
        print("✅ GERÇEK SONUÇLAR")
        print("-" * 80)
        results = session.query(Result).filter(Result.match_id == match.id).all()
        if results:
            print(f"✅ {len(results)} sonuç kaydı var:")
            for result in results:
                market = session.query(Market).filter(Market.id == result.market_id).first()
                print(f"  - Market: {market.name if market else 'Bilinmiyor'}")
                print(f"    Sonuç: {result.actual_outcome}")
                print(f"    Tarih: {result.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        else:
            print(f"❌ Sonuç bilgisi yok (ama skor var: {match.home_score}-{match.away_score})")
        print()
        
        # 7. AÇIKLAMALAR
        print("💬 LLM AÇIKLAMALARI")
        print("-" * 80)
        explanations = []
        try:
            explanations = session.query(Explanation).filter(Explanation.match_id == match.id).all()
        except Exception as e:
            print(f"⚠️  Açıklamalar tablosu henüz güncellenmemiş: {type(e).__name__}")
        
        if explanations:
            print(f"✅ {len(explanations)} açıklama kaydı var:")
            for exp in explanations:
                market = session.query(Market).filter(Market.id == exp.market_id).first()
                print(f"  - Market: {market.name if market else 'Bilinmiyor'}")
                print(f"    Model: {exp.llm_model or 'Bilinmiyor'}")
                print(f"    Açıklama: {exp.explanation_text[:200]}..." if len(exp.explanation_text) > 200 else f"    Açıklama: {exp.explanation_text}")
                print(f"    Tarih: {exp.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        else:
            print("❌ Açıklama bilgisi yok")
        print()
        
        # 8. ÖZET
        print("=" * 80)
        print("📊 VERİ ÖZETİ")
        print("=" * 80)
        print(f"✅ Temel Bilgiler: Var")
        print(f"{'✅' if match.home_support is not None else '❌'} Hype Bilgileri: {'Var' if match.home_support is not None else 'Yok'}")
        print(f"{'✅' if odds else '❌'} Odds Bilgileri: {'Var' if odds else 'Yok'}")
        print(f"{'✅' if stats else '❌'} İstatistikler: {'Var' if stats else 'Yok'} ({len(stats) if stats else 0} kayıt)")
        print(f"{'✅' if predictions else '❌'} Tahminler: {'Var' if predictions else 'Yok'} ({len(predictions) if predictions else 0} kayıt)")
        print(f"{'✅' if results else '❌'} Sonuçlar: {'Var' if results else 'Yok'} ({len(results) if results else 0} kayıt)")
        print(f"{'✅' if explanations else '❌'} Açıklamalar: {'Var' if explanations else 'Yok'} ({len(explanations) if explanations else 0} kayıt)")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    # Örnek maç göster (eğer ID belirtilmezse ilk bulunan maçı gösterir)
    show_match_details()

