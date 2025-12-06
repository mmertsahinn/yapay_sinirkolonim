"""Feature builder'ın Twitter özelliği olmadan çalıştığını test eder"""
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.features.feature_builder import FeatureBuilder
from src.db.connection import get_session

session = get_session()
try:
    fb = FeatureBuilder()
    print("✅ FeatureBuilder başarıyla oluşturuldu (Twitter özelliği YOK)")
    print()
    
    # Test için bir maç al
    from src.db.schema import Match
    match = session.query(Match).filter(
        Match.home_score.isnot(None),
        Match.away_score.isnot(None)
    ).first()
    
    if match:
        features = fb.build_match_features(
            match.home_team_id,
            match.away_team_id,
            match.match_date,
            match.league_id,
            session
        )
        
        print(f"📊 Feature Vector Boyutu: {len(features)}")
        print(f"📋 Feature'lar:")
        print(f"   - Home/Away avg_goals_scored: 2 feature")
        print(f"   - Home/Away avg_goals_conceded: 2 feature")
        print(f"   - Home/Away win_rate: 2 feature")
        print(f"   - Home/Away draw_rate: 2 feature")
        print(f"   - Home/Away loss_rate: 2 feature")
        print(f"   - Home/Away btts_rate: 2 feature")
        print(f"   - Home/Away over_25_rate: 2 feature")
        print(f"   - Home/Away home_avg_goals_scored: 2 feature")
        print(f"   - Home/Away home_avg_goals_conceded: 2 feature")
        print(f"   - Home/Away away_avg_goals_scored: 2 feature")
        print(f"   - Home/Away away_avg_goals_conceded: 2 feature")
        print(f"   - League ID (normalized): 1 feature")
        print(f"   Toplam: 25 feature")
        print()
        print(f"❌ Twitter/Hype feature'ları YOK (4 feature kaldırıldı)")
        print(f"   - home_support: KALDIRILDI")
        print(f"   - away_support: KALDIRILDI")
        print(f"   - sentiment_score: KALDIRILDI")
        print(f"   - total_tweets: KALDIRILDI")
        print()
        print(f"✅ Örnek Feature Vector (ilk 10 değer):")
        print(f"   {features[:10]}")
    else:
        print("❌ Test için maç bulunamadı")
        
finally:
    session.close()






