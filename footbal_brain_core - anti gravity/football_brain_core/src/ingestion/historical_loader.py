from datetime import datetime, date, timedelta
from typing import List, Optional
import logging
from pathlib import Path
import sys

from football_brain_core.src.ingestion.api_client import APIFootballClient
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    LeagueRepository, TeamRepository, MatchRepository, StatRepository
)
from football_brain_core.src.db.schema import Match
from football_brain_core.src.config import Config

# Checkpoint manager'ı import et
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from checkpoint_loader import CheckpointManager
    CHECKPOINT_ENABLED = True
except ImportError:
    CHECKPOINT_ENABLED = False
    CheckpointManager = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalLoader:
    def __init__(self, api_client: Optional[APIFootballClient] = None):
        self.api_client = api_client or APIFootballClient()
        self.config = Config()
        # Checkpoint manager
        if CHECKPOINT_ENABLED:
            self.checkpoint_manager = CheckpointManager()
        else:
            self.checkpoint_manager = None
    
    def load_leagues(self, season: int) -> None:
        session = get_session()
        try:
            logger.info(f"Loading leagues for season {season}")
            
            # Config'deki sabit league ID'lerini kullan
            for league_config in self.config.TARGET_LEAGUES:
                league_name = league_config.name
                api_league_id = league_config.api_league_id
                
                try:
                    league_data = self.api_client.get_leagues(league_id=api_league_id, season=season)
                    if league_data:
                        league_info = league_data[0].get("league", {})
                        country = league_info.get("country", league_config.country)
                        code = league_info.get("code", "")
                        
                        LeagueRepository.get_or_create(
                            session, name=league_name, country=country, code=code
                        )
                        logger.info(f"Loaded league: {league_name} (ID: {api_league_id})")
                    else:
                        logger.warning(f"No data found for {league_name} (ID: {api_league_id})")
                except Exception as e:
                    logger.error(f"Error loading league {league_name}: {e}")
                    continue
            
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading leagues: {e}")
            raise
        finally:
            session.close()
    
    def load_teams_for_league(self, league_name: str, season: int) -> None:
        session = get_session()
        try:
            league = LeagueRepository.get_by_id(
                session, 
                LeagueRepository.get_or_create(session, league_name).id
            )
            if not league:
                logger.warning(f"League {league_name} not found")
                return
            
            # Config'den direkt league ID'yi al
            league_config = next(
                (l for l in self.config.TARGET_LEAGUES if l.name == league_name),
                None
            )
            if not league_config:
                logger.warning(f"League config not found for {league_name}")
                return
            
            api_league_id = league_config.api_league_id
            teams_data = self.api_client.get_teams(league_id=api_league_id, season=season)
            
            for team_info in teams_data:
                team = team_info.get("team", {})
                name = team.get("name", "")
                code = team.get("code", "")
                
                if name:
                    TeamRepository.get_or_create(
                        session, name=name, league_id=league.id, code=code
                    )
            
            session.commit()
            logger.info(f"Loaded teams for {league_name}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading teams for {league_name}: {e}")
            raise
        finally:
            session.close()
    
    def load_matches_for_league(
        self,
        league_name: str,
        season: int,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None
    ) -> None:
        session = get_session()
        try:
            league = LeagueRepository.get_by_id(
                session,
                LeagueRepository.get_or_create(session, league_name).id
            )
            if not league:
                logger.warning(f"League {league_name} not found")
                return
            
            # Config'den direkt league ID'yi al
            league_config = next(
                (l for l in self.config.TARGET_LEAGUES if l.name == league_name),
                None
            )
            if not league_config:
                logger.warning(f"League config not found for {league_name}")
                return
            
            api_league_id = league_config.api_league_id
            
            if not date_from:
                date_from = date(season, 8, 1)
            if not date_to:
                date_to = date(season + 1, 7, 31)
            
            fixtures = self.api_client.get_fixtures(
                league_id=api_league_id,
                season=season,
                date_from=date_from,
                date_to=date_to
            )
            
            # Tarih sırasına göre sırala (erken tarihler önce)
            # Böylece maçlar sırayla çekilir ve hiçbir maç kaçmaz
            fixtures_sorted = sorted(
                fixtures,
                key=lambda x: x.get("fixture", {}).get("date", ""),
                reverse=False  # Eski tarihler önce
            )
            
            logger.info(f"Found {len(fixtures_sorted)} fixtures for {league_name} season {season}, processing in date order...")
            
            # Duplicate kontrolü için set
            processed_match_ids = set()
            skipped_duplicates = 0
            successful_loads = 0
            
            for fixture_data in fixtures_sorted:
                fixture = fixture_data.get("fixture", {})
                teams = fixture_data.get("teams", {})
                goals = fixture_data.get("goals", {})
                score = fixture_data.get("score", {})
                
                fixture_id = str(fixture.get("id", ""))
                
                # Duplicate kontrolü - aynı maç iki kez eklenmesin
                if not fixture_id or fixture_id in processed_match_ids:
                    skipped_duplicates += 1
                    continue
                
                processed_match_ids.add(fixture_id)
                
                try:
                    match_date = datetime.fromisoformat(
                        fixture.get("date", "").replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Invalid date for fixture {fixture_id}: {e}")
                    continue
                
                home_team_name = teams.get("home", {}).get("name", "")
                away_team_name = teams.get("away", {}).get("name", "")
                
                # Takım isimleri boşsa atla
                if not home_team_name or not away_team_name:
                    logger.warning(f"Skipping fixture {fixture_id}: missing team names")
                    continue
                
                home_team = TeamRepository.get_or_create(
                    session, name=home_team_name, league_id=league.id
                )
                away_team = TeamRepository.get_or_create(
                    session, name=away_team_name, league_id=league.id
                )
                
                home_score = goals.get("home")
                away_score = goals.get("away")
                status = fixture.get("status", {}).get("short", "")
                
                # Veritabanında zaten var mı kontrol et (match_id unique)
                existing_match = session.query(Match).filter(Match.match_id == fixture_id).first()
                if existing_match:
                    # Mevcut maçı güncelle (skor değişmiş olabilir)
                    if home_score is not None:
                        existing_match.home_score = home_score
                    if away_score is not None:
                        existing_match.away_score = away_score
                    if status:
                        existing_match.status = status
                    existing_match.updated_at = datetime.utcnow()
                else:
                    # Yeni maç ekle
                    MatchRepository.get_or_create(
                        session,
                        match_id=fixture_id,
                        league_id=league.id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        match_date=match_date,
                        home_score=home_score,
                        away_score=away_score,
                        status=status
                    )
                    successful_loads += 1
            
            session.commit()
            
            # Detaylı log
            logger.info(f"Loaded matches for {league_name} season {season}:")
            logger.info(f"  - Total fixtures from API: {len(fixtures_sorted)}")
            logger.info(f"  - Successfully loaded: {successful_loads}")
            logger.info(f"  - Updated existing: {len(processed_match_ids) - successful_loads - skipped_duplicates}")
            logger.info(f"  - Skipped duplicates: {skipped_duplicates}")
            
            # API limit bilgisi
            if hasattr(self.api_client, 'requests_today'):
                remaining = self.api_client.daily_limit - self.api_client.requests_today
                logger.info(f"API requests remaining today: {remaining}/{self.api_client.daily_limit}")
                
                if remaining <= 0:
                    logger.error("API LIMITI DOLDU! Yeni API key gerekiyor!")
                elif remaining < 10:
                    logger.warning(f"API limiti azaliyor: {remaining} kalan")
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading matches for {league_name}: {e}")
            raise
        finally:
            session.close()
    
    def load_all_historical_data(self, seasons: Optional[List[int]] = None) -> None:
        if not seasons:
            current_year = datetime.now().year
            seasons = list(range(current_year - self.config.HISTORICAL_SEASONS, current_year))
        
        logger.info(f"Loading historical data for seasons: {seasons}")
        
        for season in seasons:
            logger.info(f"Processing season {season}")
            self.load_leagues(season)
            
            for league_config in self.config.TARGET_LEAGUES:
                logger.info(f"Loading data for {league_config.name}")
                self.load_teams_for_league(league_config.name, season)
                self.load_matches_for_league(league_config.name, season)
        
        logger.info("Historical data loading completed")

