from datetime import datetime, date, timedelta
from typing import List, Optional
import logging

from football_brain_core.src.ingestion.api_client import APIFootballClient
from football_brain_core.src.ingestion.historical_loader import HistoricalLoader
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository, ResultRepository
from football_brain_core.src.features.market_targets import calculate_all_market_outcomes
from football_brain_core.src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyJobs:
    def __init__(self, api_client: Optional[APIFootballClient] = None):
        self.api_client = api_client or APIFootballClient()
        self.loader = HistoricalLoader(self.api_client)
        self.config = Config()
    
    def update_fixtures(self, days_ahead: int = 7) -> None:
        session = get_session()
        try:
            today = date.today()
            date_to = today + timedelta(days=days_ahead)
            
            logger.info(f"Updating fixtures from {today} to {date_to}")
            
            current_year = datetime.now().year
            season = current_year if datetime.now().month >= 8 else current_year - 1
            
            # Config'deki sabit league ID'lerini direkt kullan
            for league_config in self.config.TARGET_LEAGUES:
                league_name = league_config.name
                api_league_id = league_config.api_league_id
                
                fixtures = self.api_client.get_fixtures(
                    league_id=api_league_id,
                    season=season,
                    date_from=today,
                    date_to=date_to
                )
                
                # Tarih sırasına göre sırala (erken tarihler önce)
                # Böylece maçlar sırayla çekilir ve hiçbir maç kaçmaz
                fixtures_sorted = sorted(
                    fixtures,
                    key=lambda x: x.get("fixture", {}).get("date", ""),
                    reverse=False  # Eski tarihler önce
                )
                
                from football_brain_core.src.db.repositories import LeagueRepository, TeamRepository
                
                league = LeagueRepository.get_or_create(session, league_name)
                
                for fixture_data in fixtures_sorted:
                    fixture = fixture_data.get("fixture", {})
                    teams = fixture_data.get("teams", {})
                    
                    fixture_id = str(fixture.get("id", ""))
                    match_date = datetime.fromisoformat(
                        fixture.get("date", "").replace("Z", "+00:00")
                    )
                    
                    home_team_name = teams.get("home", {}).get("name", "")
                    away_team_name = teams.get("away", {}).get("name", "")
                    
                    home_team = TeamRepository.get_or_create(
                        session, name=home_team_name, league_id=league.id
                    )
                    away_team = TeamRepository.get_or_create(
                        session, name=away_team_name, league_id=league.id
                    )
                    
                    MatchRepository.get_or_create(
                        session,
                        match_id=fixture_id,
                        league_id=league.id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        match_date=match_date,
                        status=fixture.get("status", {}).get("short", "")
                    )
                
                logger.info(f"Updated {len(fixtures)} fixtures for {league_name}")
            
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating fixtures: {e}")
            raise
        finally:
            session.close()
    
    def update_results(self, days_back: int = 7) -> None:
        session = get_session()
        try:
            today = date.today()
            date_from = today - timedelta(days=days_back)
            
            logger.info(f"Updating results from {date_from} to {today}")
            
            current_year = datetime.now().year
            season = current_year if datetime.now().month >= 8 else current_year - 1
            
            # Config'deki sabit league ID'lerini direkt kullan
            for league_config in self.config.TARGET_LEAGUES:
                league_name = league_config.name
                api_league_id = league_config.api_league_id
                
                fixtures = self.api_client.get_fixtures(
                    league_id=api_league_id,
                    season=season,
                    date_from=date_from,
                    date_to=today
                )
                
                # Tarih sırasına göre sırala (erken tarihler önce)
                # Böylece maçlar sırayla çekilir ve hiçbir maç kaçmaz
                fixtures_sorted = sorted(
                    fixtures,
                    key=lambda x: x.get("fixture", {}).get("date", ""),
                    reverse=False  # Eski tarihler önce
                )
                
                from football_brain_core.src.db.repositories import LeagueRepository, MarketRepository
                from football_brain_core.src.features.market_targets import MarketType
                
                league = LeagueRepository.get_or_create(session, league_name)
                
                for fixture_data in fixtures_sorted:
                    fixture = fixture_data.get("fixture", {})
                    goals = fixture_data.get("goals", {})
                    
                    fixture_id = str(fixture.get("id", ""))
                    match = MatchRepository.get_by_id(
                        session,
                        MatchRepository.get_or_create(
                            session,
                            match_id=fixture_id,
                            league_id=league.id,
                            home_team_id=1,
                            away_team_id=1,
                            match_date=datetime.now()
                        ).id
                    )
                    
                    if not match:
                        continue
                    
                    home_score = goals.get("home")
                    away_score = goals.get("away")
                    
                    if home_score is None or away_score is None:
                        continue
                    
                    match.home_score = home_score
                    match.away_score = away_score
                    match.status = fixture.get("status", {}).get("short", "")
                    
                    outcomes = calculate_all_market_outcomes(home_score, away_score)
                    
                    for market_type, outcome in outcomes.items():
                        market = MarketRepository.get_or_create(
                            session, name=market_type.value
                        )
                        ResultRepository.get_or_create(
                            session,
                            match_id=match.id,
                            market_id=market.id,
                            actual_outcome=outcome
                        )
                
                logger.info(f"Updated results for {league_name}")
            
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating results: {e}")
            raise
        finally:
            session.close()
    
    def run_daily_update(self) -> None:
        logger.info("Running daily update job")
        self.update_fixtures(days_ahead=7)
        self.update_results(days_back=7)
        logger.info("Daily update completed")

