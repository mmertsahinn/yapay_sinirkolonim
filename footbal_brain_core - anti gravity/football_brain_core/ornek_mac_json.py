"""
Database'den bir örnek maç getirip JSON formatında gösterir
"""
import sys
from pathlib import Path
import json
from datetime import datetime

# Project root'u path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team, Stat, Market, Prediction, Result
from sqlalchemy import and_, extract

def datetime_serializer(obj):
    """Datetime objelerini string'e çevirir"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def get_match_dataset(match_id: int = None):
    """Bir maçın tüm verilerini dictionary olarak döner"""
    session = get_session()
    
    try:
        # Maç seç (hype bilgisi olan bir maç tercih edilir)
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
            return {"error": "Maç bulunamadı"}
        
        # Temel bilgiler
        dataset = {
            "match_id": match.id,
            "match_unique_id": match.match_id,
            "league": {
                "id": match.league.id if match.league else None,
                "name": match.league.name if match.league else None,
                "country": match.league.country if match.league else None,
                "code": match.league.code if match.league else None,
            },
            "match_date": match.match_date.isoformat() if match.match_date else None,
            "home_team": {
                "id": match.home_team.id if match.home_team else None,
                "name": match.home_team.name if match.home_team else None,
                "code": match.home_team.code if match.home_team else None,
            },
            "away_team": {
                "id": match.away_team.id if match.away_team else None,
                "name": match.away_team.name if match.away_team else None,
                "code": match.away_team.code if match.away_team else None,
            },
            "score": {
                "home": match.home_score,
                "away": match.away_score,
                "result": f"{match.home_score}-{match.away_score}" if match.home_score is not None and match.away_score is not None else None
            },
            "status": match.status,
            "hype_data": {
                "home_support": match.home_support,
                "away_support": match.away_support,
                "sentiment_score": match.sentiment_score,
                "total_tweets": match.total_tweets,
                "hype_updated_at": match.hype_updated_at.isoformat() if match.hype_updated_at else None,
            },
            "odds_data": None,
            "stats": [],
            "predictions": [],
            "results": [],
            "metadata": {
                "created_at": match.created_at.isoformat() if match.created_at else None,
                "updated_at": match.updated_at.isoformat() if match.updated_at else None,
            }
        }
        
        # Odds bilgileri
        try:
            odds = session.query(MatchOdds).filter(MatchOdds.match_id == match.id).first()
            if odds:
                dataset["odds_data"] = {
                    "bet365": {
                        "home": odds.b365_h,
                        "draw": odds.b365_d,
                        "away": odds.b365_a,
                        "over_25": odds.b365_over_25,
                        "under_25": odds.b365_under_25,
                    },
                    "pinnacle": {
                        "home": odds.p_h,
                        "draw": odds.p_d,
                        "away": odds.p_a,
                        "over_25": odds.p_over_25,
                        "under_25": odds.p_under_25,
                    },
                    "william_hill": {
                        "home": odds.wh_h,
                        "draw": odds.wh_d,
                        "away": odds.wh_a,
                    },
                    "market_averages": {
                        "home": odds.avg_h,
                        "draw": odds.avg_d,
                        "away": odds.avg_a,
                        "over_25": odds.avg_over_25,
                        "under_25": odds.avg_under_25,
                    },
                    "all_odds": odds.all_odds,
                }
        except:
            pass
        
        # İstatistikler
        try:
            stats = session.query(Stat).filter(Stat.match_id == match.id).all()
            for stat in stats:
                team = session.query(Team).filter(Team.id == stat.team_id).first()
                dataset["stats"].append({
                    "team_id": stat.team_id,
                    "team_name": team.name if team else None,
                    "stat_type": stat.stat_type,
                    "stat_value": stat.stat_value,
                })
        except:
            pass
        
        # Tahminler
        try:
            predictions = session.query(Prediction).filter(Prediction.match_id == match.id).all()
            for pred in predictions:
                market = session.query(Market).filter(Market.id == pred.market_id).first()
                dataset["predictions"].append({
                    "market": market.name if market else None,
                    "predicted_outcome": pred.predicted_outcome,
                    "probability": pred.p_hat,
                    "timestamp": pred.timestamp.isoformat() if pred.timestamp else None,
                })
        except:
            pass
        
        # Sonuçlar
        try:
            results = session.query(Result).filter(Result.match_id == match.id).all()
            for result in results:
                market = session.query(Market).filter(Market.id == result.market_id).first()
                dataset["results"].append({
                    "market": market.name if market else None,
                    "actual_outcome": result.actual_outcome,
                    "created_at": result.created_at.isoformat() if result.created_at else None,
                })
        except:
            pass
        
        return dataset
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        session.close()

if __name__ == "__main__":
    dataset = get_match_dataset()
    print(json.dumps(dataset, indent=2, ensure_ascii=False, default=datetime_serializer))

