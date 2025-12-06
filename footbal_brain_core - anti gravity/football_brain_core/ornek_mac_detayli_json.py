"""
Database'den rastgele bir maç seçip tüm bilgilerini detaylı JSON dosyasına yazar
"""
import sys
import json
from pathlib import Path
from datetime import datetime
import random

# Project root'u path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.db.connection import get_session
from src.db.schema import Match, MatchOdds, League, Team, Stat, Market, Prediction, Result
from sqlalchemy import and_, extract

# Windows encoding sorunu için
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def datetime_serializer(obj):
    """Datetime objelerini string'e çevirir"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def get_detailed_match_dataset(match_id: int = None):
    """Bir maçın tüm verilerini detaylı dictionary olarak döner"""
    session = get_session()
    
    try:
        # Maç seç (rastgele veya belirtilen ID)
        if match_id:
            match = session.query(Match).filter(Match.id == match_id).first()
        else:
            # 2020-2025 arası, skoru olan tüm maçları al
            all_matches = session.query(Match).filter(
                and_(
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None),
                    extract('year', Match.match_date) >= 2020,
                    extract('year', Match.match_date) <= 2025
                )
            ).all()
            
            if all_matches:
                # Rastgele bir maç seç
                match = random.choice(all_matches)
            else:
                return {"error": "2020-2025 arası maç bulunamadı"}
        
        if not match:
            return {"error": "Maç bulunamadı"}
        
        # Lig bilgisi
        league = session.query(League).filter(League.id == match.league_id).first()
        
        # Takım bilgileri
        home_team = session.query(Team).filter(Team.id == match.home_team_id).first()
        away_team = session.query(Team).filter(Team.id == match.away_team_id).first()
        
        # Temel bilgiler
        dataset = {
            "match_info": {
                "match_id": match.id,
                "match_unique_id": match.match_id,
                "match_date": match.match_date.isoformat() if match.match_date else None,
                "status": match.status,
                "created_at": match.created_at.isoformat() if match.created_at else None,
                "updated_at": match.updated_at.isoformat() if match.updated_at else None,
            },
            "league": {
                "id": league.id if league else None,
                "name": league.name if league else None,
                "country": league.country if league else None,
                "code": league.code if league else None,
            },
            "teams": {
                "home": {
                    "id": home_team.id if home_team else None,
                    "name": home_team.name if home_team else None,
                    "code": home_team.code if home_team else None,
                },
                "away": {
                    "id": away_team.id if away_team else None,
                    "name": away_team.name if away_team else None,
                    "code": away_team.code if away_team else None,
                }
            },
            "score": {
                "home": match.home_score,
                "away": match.away_score,
                "result": f"{match.home_score}-{match.away_score}" if match.home_score is not None and match.away_score is not None else None,
                "home_win": match.home_score > match.away_score if match.home_score is not None and match.away_score is not None else None,
                "draw": match.home_score == match.away_score if match.home_score is not None and match.away_score is not None else None,
                "away_win": match.home_score < match.away_score if match.home_score is not None and match.away_score is not None else None,
            },
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
    print("=" * 80)
    print("📊 RASTGELE MAÇ DETAYLI JSON OLUŞTURULUYOR")
    print("=" * 80)
    print()
    
    dataset = get_detailed_match_dataset()
    
    if "error" in dataset:
        print(f"❌ Hata: {dataset['error']}")
    else:
        # JSON dosyasına kaydet
        output_file = project_root / "ornek_mac_detayli.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False, default=datetime_serializer)
        
        print(f"✅ JSON dosyası oluşturuldu: {output_file}")
        print()
        print("=" * 80)
        print("📋 ÖZET BİLGİLER")
        print("=" * 80)
        print(f"Maç ID: {dataset['match_info']['match_id']}")
        print(f"Lig: {dataset['league']['name']}")
        print(f"Tarih: {dataset['match_info']['match_date']}")
        print(f"Ev Sahibi: {dataset['teams']['home']['name']}")
        print(f"Deplasman: {dataset['teams']['away']['name']}")
        print(f"Skor: {dataset['score']['result']}")
        print(f"Hype Bilgisi: {'✅ Var' if dataset['hype_data']['hype_updated_at'] else '❌ Yok'}")
        print(f"Odds Bilgisi: {'✅ Var' if dataset['odds_data'] else '❌ Yok'}")
        print(f"İstatistik: {len(dataset['stats'])} kayıt")
        print(f"Tahmin: {len(dataset['predictions'])} kayıt")
        print(f"Sonuç: {len(dataset['results'])} kayıt")
        print("=" * 80)
        print()
        print("📄 JSON içeriği:")
        print(json.dumps(dataset, indent=2, ensure_ascii=False, default=datetime_serializer))






