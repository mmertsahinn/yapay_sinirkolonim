"""
Veritabanını JSON'a çevir - TÜM VERİLER
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Python path'i düzelt
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from src.db.connection import get_session
from src.db.schema import (
    League, Team, Match, MatchOdds, Stat, Prediction, 
    Result, Explanation, Experiment, ModelVersion, Market
)

def datetime_to_str(obj):
    """Datetime objelerini string'e çevir"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def serialize_match(match: Match) -> Dict[str, Any]:
    """Match objesini dict'e çevir - None değerleri None olarak bırak"""
    # Hype verileri: Eğer hype_updated_at None ise, hype verileri de None olmalı
    # (0.5 default değerleri gerçek veri değil)
    if match.hype_updated_at is None:
        # Hype verisi yok, hepsi None
        home_support = None
        away_support = None
        sentiment_score = None
        total_tweets = None
    else:
        # Hype verisi var, gerçek değerleri kullan
        home_support = match.home_support
        away_support = match.away_support
        sentiment_score = match.sentiment_score
        total_tweets = match.total_tweets
    
    return {
        "id": match.id,
        "match_id": match.match_id,
        "league_id": match.league_id,
        "home_team_id": match.home_team_id,
        "away_team_id": match.away_team_id,
        "match_date": datetime_to_str(match.match_date),
        "home_score": match.home_score,
        "away_score": match.away_score,
        "status": match.status,
        # Hype verileri - None olanlar None, gerçek veriler gerçek değer
        "home_support": home_support,
        "away_support": away_support,
        "sentiment_score": sentiment_score,
        "total_tweets": total_tweets,
        "hype_updated_at": datetime_to_str(match.hype_updated_at),
        "created_at": datetime_to_str(match.created_at),
        "updated_at": datetime_to_str(match.updated_at),
        # İlişkili veriler
        "league_name": match.league.name if match.league else None,
        "home_team_name": match.home_team.name if match.home_team else None,
        "away_team_name": match.away_team.name if match.away_team else None,
    }

def serialize_league(league: League) -> Dict[str, Any]:
    """League objesini dict'e çevir"""
    return {
        "id": league.id,
        "name": league.name,
        "country": league.country,
        "code": league.code,
        "created_at": datetime_to_str(league.created_at),
    }

def serialize_team(team: Team) -> Dict[str, Any]:
    """Team objesini dict'e çevir"""
    return {
        "id": team.id,
        "name": team.name,
        "league_id": team.league_id,
        "code": team.code,
        "created_at": datetime_to_str(team.created_at),
        "league_name": team.league.name if team.league else None,
    }

def serialize_match_odds(odds: MatchOdds) -> Dict[str, Any]:
    """MatchOdds objesini dict'e çevir"""
    return {
        "id": odds.id,
        "match_id": odds.match_id,
        "b365_h": odds.b365_h,
        "b365_d": odds.b365_d,
        "b365_a": odds.b365_a,
        "bf_h": odds.bf_h,
        "bf_d": odds.bf_d,
        "bf_a": odds.bf_a,
        "bfd_h": odds.bfd_h,
        "bfd_d": odds.bfd_d,
        "bfd_a": odds.bfd_a,
        "bv_h": odds.bv_h,
        "bv_d": odds.bv_d,
        "bv_a": odds.bv_a,
        "cl_h": odds.cl_h,
        "cl_d": odds.cl_d,
        "cl_a": odds.cl_a,
        "lb_h": odds.lb_h,
        "lb_d": odds.lb_d,
        "lb_a": odds.lb_a,
        "p_h": odds.p_h,
        "p_d": odds.p_d,
        "p_a": odds.p_a,
        "wh_h": odds.wh_h,
        "wh_d": odds.wh_d,
        "wh_a": odds.wh_a,
    }

def serialize_stat(stat: Stat) -> Dict[str, Any]:
    """Stat objesini dict'e çevir"""
    return {
        "id": stat.id,
        "match_id": stat.match_id,
        "team_id": stat.team_id,
        "stat_type": stat.stat_type,
        "value": stat.value,
    }

def serialize_prediction(pred: Prediction) -> Dict[str, Any]:
    """Prediction objesini dict'e çevir"""
    return {
        "id": pred.id,
        "match_id": pred.match_id,
        "market_id": pred.market_id,
        "predicted_outcome": getattr(pred, 'predicted_outcome', None),
        "p_hat": getattr(pred, 'p_hat', None),
        "model_version_id": pred.model_version_id,
        "timestamp": datetime_to_str(getattr(pred, 'timestamp', pred.created_at)),
        "created_at": datetime_to_str(pred.created_at),
    }

def serialize_result(result: Result) -> Dict[str, Any]:
    """Result objesini dict'e çevir"""
    return {
        "id": result.id,
        "match_id": result.match_id,
        "market_id": result.market_id,
        "actual_outcome": getattr(result, 'actual_outcome', None),
        "created_at": datetime_to_str(result.created_at),
        "updated_at": datetime_to_str(result.updated_at),
    }

def serialize_explanation(expl: Explanation) -> Dict[str, Any]:
    """Explanation objesini dict'e çevir"""
    return {
        "id": expl.id,
        "match_id": expl.match_id,
        "market_id": expl.market_id,
        "explanation_text": getattr(expl, 'explanation_text', None),
        "llm_model": getattr(expl, 'llm_model', None),
        "created_at": datetime_to_str(expl.created_at),
    }

def serialize_experiment(exp: Experiment) -> Dict[str, Any]:
    """Experiment objesini dict'e çevir"""
    return {
        "id": exp.id,
        "name": getattr(exp, 'name', None),
        "description": getattr(exp, 'description', None),
        "config": getattr(exp, 'config', None),
        "results": getattr(exp, 'results', None),
        "created_at": datetime_to_str(exp.created_at),
    }

def serialize_model_version(mv: ModelVersion) -> Dict[str, Any]:
    """ModelVersion objesini dict'e çevir"""
    return {
        "id": mv.id,
        "version": mv.version,
        "description": mv.description,
        "is_active": mv.is_active,
        "created_at": datetime_to_str(mv.created_at),
    }

def serialize_market(market: Market) -> Dict[str, Any]:
    """Market objesini dict'e çevir"""
    return {
        "id": market.id,
        "name": market.name,
        "description": market.description,
    }

def export_database_to_json(output_file: str = "football_brain_export.json"):
    """Tüm veritabanını JSON'a export et"""
    session = get_session()
    
    print("=" * 80)
    print("📊 VERİTABANI JSON EXPORT")
    print("=" * 80)
    print()
    
    try:
        # 1. Leagues
        print("📋 Ligler export ediliyor...")
        leagues = session.query(League).all()
        leagues_data = [serialize_league(league) for league in leagues]
        print(f"   ✅ {len(leagues_data)} lig")
        
        # 2. Teams
        print("👥 Takımlar export ediliyor...")
        teams = session.query(Team).all()
        teams_data = [serialize_team(team) for team in teams]
        print(f"   ✅ {len(teams_data)} takım")
        
        # 3. Matches (EN ÖNEMLİ - HYPE VERİLERİ BURADA)
        # EN GÜNCEL EN ALTA: .asc() kullan (en eski en üstte, en güncel en altta)
        print("⚽ Maçlar export ediliyor...")
        matches = session.query(Match).order_by(Match.match_date.asc()).all()
        matches_data = [serialize_match(match) for match in matches]
        
        # Hype istatistikleri
        with_hype = sum(1 for m in matches_data if m.get("hype_updated_at") is not None)
        total_mentions = sum(m.get("total_tweets", 0) or 0 for m in matches_data)
        
        print(f"   ✅ {len(matches_data)} maç")
        print(f"   📢 Hype'ı olan: {with_hype} maç")
        print(f"   📊 Toplam mentions: {total_mentions:,}")
        
        # 4. Match Odds
        print("💰 Bahis oranları export ediliyor...")
        odds = session.query(MatchOdds).all()
        odds_data = [serialize_match_odds(odd) for odd in odds]
        print(f"   ✅ {len(odds_data)} bahis oranı")
        
        # 5. Stats
        print("📈 İstatistikler export ediliyor...")
        stats = session.query(Stat).all()
        stats_data = [serialize_stat(stat) for stat in stats]
        print(f"   ✅ {len(stats_data)} istatistik")
        
        # 6. Predictions
        print("🔮 Tahminler export ediliyor...")
        predictions = session.query(Prediction).all()
        predictions_data = [serialize_prediction(pred) for pred in predictions]
        print(f"   ✅ {len(predictions_data)} tahmin")
        
        # 7. Results
        print("✅ Sonuçlar export ediliyor...")
        results = session.query(Result).all()
        results_data = [serialize_result(result) for result in results]
        print(f"   ✅ {len(results_data)} sonuç")
        
        # 8. Explanations (hata olursa atla)
        explanations_data = []
        try:
            print("📝 Açıklamalar export ediliyor...")
            explanations = session.query(Explanation).all()
            explanations_data = [serialize_explanation(expl) for expl in explanations]
            print(f"   ✅ {len(explanations_data)} açıklama")
        except Exception as e:
            print(f"   ⚠️ Açıklamalar tablosu yok veya hata: {e}")
        
        # 9. Experiments (hata olursa atla)
        experiments_data = []
        try:
            print("🧪 Deneyler export ediliyor...")
            experiments = session.query(Experiment).all()
            experiments_data = [serialize_experiment(exp) for exp in experiments]
            print(f"   ✅ {len(experiments_data)} deney")
        except Exception as e:
            print(f"   ⚠️ Deneyler tablosu yok veya hata: {e}")
        
        # 10. Model Versions (hata olursa atla)
        model_versions_data = []
        try:
            print("🤖 Model versiyonları export ediliyor...")
            model_versions = session.query(ModelVersion).all()
            model_versions_data = [serialize_model_version(mv) for mv in model_versions]
            print(f"   ✅ {len(model_versions_data)} model versiyonu")
        except Exception as e:
            print(f"   ⚠️ Model versiyonları tablosu yok veya hata: {e}")
        
        # 11. Markets (hata olursa atla)
        markets_data = []
        try:
            print("🏪 Marketler export ediliyor...")
            markets = session.query(Market).all()
            markets_data = [serialize_market(market) for market in markets]
            print(f"   ✅ {len(markets_data)} market")
        except Exception as e:
            print(f"   ⚠️ Marketler tablosu yok veya hata: {e}")
        
        # Tüm verileri birleştir
        export_data = {
            "export_date": datetime.now().isoformat(),
            "summary": {
                "leagues": len(leagues_data),
                "teams": len(teams_data),
                "matches": len(matches_data),
                "matches_with_hype": with_hype,
                "total_mentions": total_mentions,
                "match_odds": len(odds_data),
                "stats": len(stats_data),
                "predictions": len(predictions_data),
                "results": len(results_data),
                "explanations": len(explanations_data),
                "experiments": len(experiments_data),
                "model_versions": len(model_versions_data),
                "markets": len(markets_data),
            },
            "data": {
                "leagues": leagues_data,
                "teams": teams_data,
                "matches": matches_data,
                "match_odds": odds_data,
                "stats": stats_data,
                "predictions": predictions_data,
                "results": results_data,
                "explanations": explanations_data,
                "experiments": experiments_data,
                "model_versions": model_versions_data,
                "markets": markets_data,
            }
        }
        
        # JSON'a yaz
        output_path = Path(project_root) / output_file
        print()
        print(f"💾 JSON dosyasına yazılıyor: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=datetime_to_str)
        
        file_size = output_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print()
        print("=" * 80)
        print("✅ EXPORT TAMAMLANDI!")
        print("=" * 80)
        print(f"📁 Dosya: {output_path}")
        print(f"📊 Dosya boyutu: {file_size_mb:.2f} MB ({file_size:,} bytes)")
        print()
        print("📋 ÖZET:")
        print(f"   🏆 Ligler: {len(leagues_data)}")
        print(f"   👥 Takımlar: {len(teams_data)}")
        print(f"   ⚽ Maçlar: {len(matches_data)}")
        print(f"   📢 Hype'ı olan maçlar: {with_hype}")
        print(f"   📊 Toplam mentions: {total_mentions:,}")
        print(f"   💰 Bahis oranları: {len(odds_data)}")
        print()
        print("🎉 Tüm veriler JSON formatında hazır!")
        print("=" * 80)
        
        return output_path
        
    except Exception as e:
        print(f"❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        session.close()


if __name__ == "__main__":
    export_database_to_json()

