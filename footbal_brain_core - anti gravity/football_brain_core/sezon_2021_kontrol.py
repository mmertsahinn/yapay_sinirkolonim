"""2021 sezonu kontrol"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_session
from src.db.schema import Match, League
from src.config import Config

session = get_session()
config = Config()

print("2021 SEZONU KONTROLU:")
print("="*60)

season_start = datetime(2021, 8, 1)
season_end = datetime(2022, 7, 31)

for league_config in config.TARGET_LEAGUES:
    league = session.query(League).filter(League.name == league_config.name).first()
    if league:
        count = session.query(Match).filter(
            Match.league_id == league.id,
            Match.match_date >= season_start,
            Match.match_date <= season_end
        ).count()
        print(f"{league_config.name}: {count} mac")

session.close()






