"""
Takım Analizi ve İlişki Takibi
- Her takımın kendi pattern'lerini öğrenir
- Takım çiftlerinin ikili ilişkilerini analiz eder
- Her hatadan sonra bu bilgileri günceller
"""
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from collections import defaultdict

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    MatchRepository, TeamRepository, ResultRepository, MarketRepository
)
from football_brain_core.src.features.market_targets import MarketType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamAnalyzer:
    """
    Takım bazlı analiz ve öğrenme.
    Her takımın pattern'lerini ve takım çiftlerinin ilişkilerini öğrenir.
    """
    
    def analyze_team_patterns(
        self,
        team_id: int,
        matches: List,
        market_type: MarketType
    ) -> Dict[str, Any]:
        """
        Bir takımın belirli bir market için pattern'lerini analiz eder.
        """
        session = get_session()
        try:
            team = TeamRepository.get_by_id(session, team_id)
            if not team:
                return {}
            
            # Ev sahibi ve deplasman pattern'leri
            home_matches = [m for m in matches if m.home_team_id == team_id]
            away_matches = [m for m in matches if m.away_team_id == team_id]
            
            market = MarketRepository.get_or_create(session, name=market_type.value)
            
            pattern = {
                "team_id": team_id,
                "team_name": team.name,
                "market": market_type.value,
                "home_patterns": {},
                "away_patterns": {},
                "overall_patterns": {},
                "strengths": [],
                "weaknesses": []
            }
            
            # Ev sahibi pattern'leri
            if home_matches:
                home_outcomes = []
                for match in home_matches:
                    results = ResultRepository.get_by_match(session, match.id)
                    result = next((r for r in results if r.market_id == market.id), None)
                    if result:
                        home_outcomes.append(result.actual_outcome)
                
                if home_outcomes:
                    from collections import Counter
                    outcome_counts = Counter(home_outcomes)
                    total = len(home_outcomes)
                    
                    pattern["home_patterns"] = {
                        outcome: count / total
                        for outcome, count in outcome_counts.items()
                    }
            
            # Deplasman pattern'leri
            if away_matches:
                away_outcomes = []
                for match in away_matches:
                    results = ResultRepository.get_by_match(session, match.id)
                    result = next((r for r in results if r.market_id == market.id), None)
                    if result:
                        away_outcomes.append(result.actual_outcome)
                
                if away_outcomes:
                    from collections import Counter
                    outcome_counts = Counter(away_outcomes)
                    total = len(away_outcomes)
                    
                    pattern["away_patterns"] = {
                        outcome: count / total
                        for outcome, count in outcome_counts.items()
                    }
            
            # Güçlü yönler ve zayıf yönler
            if pattern["home_patterns"]:
                max_home = max(pattern["home_patterns"].items(), key=lambda x: x[1])
                pattern["strengths"].append(f"Ev sahibi: {max_home[0]} ({max_home[1]:.1%})")
            
            if pattern["away_patterns"]:
                max_away = max(pattern["away_patterns"].items(), key=lambda x: x[1])
                if max_away[1] < 0.3:  # Düşük başarı
                    pattern["weaknesses"].append(f"Deplasman: {max_away[0]} zayıf ({max_away[1]:.1%})")
            
            return pattern
        
        finally:
            session.close()
    
    def analyze_team_pair_relationship(
        self,
        team_a_id: int,
        team_b_id: int,
        matches: List,
        market_type: MarketType
    ) -> Dict[str, Any]:
        """
        İki takım arasındaki ikili ilişkiyi analiz eder.
        Hangi takım hangi takıma karşı nasıl performans gösteriyor.
        """
        session = get_session()
        try:
            team_a = TeamRepository.get_by_id(session, team_a_id)
            team_b = TeamRepository.get_by_id(session, team_b_id)
            
            if not team_a or not team_b:
                return {}
            
            # Bu iki takım arasındaki maçları bul
            pair_matches = [
                m for m in matches
                if (m.home_team_id == team_a_id and m.away_team_id == team_b_id) or
                   (m.home_team_id == team_b_id and m.away_team_id == team_a_id)
            ]
            
            if not pair_matches:
                return {}
            
            market = MarketRepository.get_or_create(session, name=market_type.value)
            
            relationship = {
                "team_a_id": team_a_id,
                "team_a_name": team_a.name,
                "team_b_id": team_b_id,
                "team_b_name": team_b.name,
                "market": market_type.value,
                "total_matches": len(pair_matches),
                "team_a_as_home": {},
                "team_a_as_away": {},
                "team_b_as_home": {},
                "team_b_as_away": {},
                "dominance": None,  # Hangi takım daha dominant
                "relationship_type": None  # "rivalry", "dominance", "balanced"
            }
            
            # Team A ev sahibi olduğunda
            a_home_matches = [m for m in pair_matches if m.home_team_id == team_a_id]
            a_home_outcomes = []
            for match in a_home_matches:
                results = ResultRepository.get_by_match(session, match.id)
                result = next((r for r in results if r.market_id == market.id), None)
                if result:
                    a_home_outcomes.append(result.actual_outcome)
            
            if a_home_outcomes:
                from collections import Counter
                relationship["team_a_as_home"] = dict(Counter(a_home_outcomes))
            
            # Team A deplasman olduğunda
            a_away_matches = [m for m in pair_matches if m.away_team_id == team_a_id]
            a_away_outcomes = []
            for match in a_away_matches:
                results = ResultRepository.get_by_match(session, match.id)
                result = next((r for r in results if r.market_id == market.id), None)
                if result:
                    a_away_outcomes.append(result.actual_outcome)
            
            if a_away_outcomes:
                from collections import Counter
                relationship["team_a_as_away"] = dict(Counter(a_away_outcomes))
            
            # Dominance analizi
            if a_home_outcomes and a_away_outcomes:
                a_home_wins = sum(1 for o in a_home_outcomes if o in ["1", "Home", "Yes"])
                a_away_wins = sum(1 for o in a_away_outcomes if o in ["2", "Away", "Yes"])
                
                if a_home_wins > len(a_home_outcomes) * 0.6:
                    relationship["dominance"] = f"{team_a.name} ev sahibi avantajı var"
                    relationship["relationship_type"] = "home_advantage"
                elif a_away_wins > len(a_away_outcomes) * 0.6:
                    relationship["dominance"] = f"{team_a.name} deplasman güçlü"
                    relationship["relationship_type"] = "away_strength"
                else:
                    relationship["relationship_type"] = "balanced"
            
            return relationship
        
        finally:
            session.close()
    
    def get_all_team_patterns(
        self,
        season: int,
        market_types: List[MarketType]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Tüm takımların pattern'lerini analiz eder.
        """
        session = get_session()
        try:
            all_matches = []
            leagues = LeagueRepository.get_all(session)
            
            for league in leagues:
                matches = MatchRepository.get_by_league_and_season(
                    session, league.id, season
                )
                matches = [m for m in matches if m.home_score is not None and m.away_score is not None]
                all_matches.extend(matches)
            
            teams = TeamRepository.get_all(session)
            all_patterns = {}
            
            for team in teams:
                team_patterns = {}
                for market_type in market_types:
                    pattern = self.analyze_team_patterns(team.id, all_matches, market_type)
                    if pattern:
                        team_patterns[market_type.value] = pattern
                
                if team_patterns:
                    all_patterns[team.id] = {
                        "team_id": team.id,
                        "team_name": team.name,
                        "patterns": team_patterns
                    }
            
            return all_patterns
        
        finally:
            session.close()
    
    def get_all_team_relationships(
        self,
        season: int,
        market_types: List[MarketType]
    ) -> List[Dict[str, Any]]:
        """
        Tüm takım çiftlerinin ilişkilerini analiz eder.
        """
        session = get_session()
        try:
            all_matches = []
            leagues = LeagueRepository.get_all(session)
            
            for league in leagues:
                matches = MatchRepository.get_by_league_and_season(
                    session, league.id, season
                )
                matches = [m for m in matches if m.home_score is not None and m.away_score is not None]
                all_matches.extend(matches)
            
            teams = TeamRepository.get_all(session)
            relationships = []
            
            # Her takım çifti için
            for i, team_a in enumerate(teams):
                for team_b in teams[i+1:]:
                    for market_type in market_types:
                        relationship = self.analyze_team_pair_relationship(
                            team_a.id, team_b.id, all_matches, market_type
                        )
                        if relationship and relationship.get("total_matches", 0) > 0:
                            relationships.append(relationship)
            
            return relationships
        
        finally:
            session.close()







