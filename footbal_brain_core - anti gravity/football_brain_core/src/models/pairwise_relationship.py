"""
Takım Çiftleri İlişki Modeli
Her takım çifti için ayrı algoritma/düşünce:
- H2H (head-to-head) analizi
- Ev sahibi avantajı
- Deplasman gücü
- İkili dinamikler
"""
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import logging

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    MatchRepository, TeamRepository, ResultRepository, MarketRepository, LeagueRepository
)
from football_brain_core.src.features.market_targets import MarketType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairwiseRelationship:
    """
    İki takım arasındaki özel ilişki modeli.
    Her takım çifti için ayrı algoritma.
    """
    
    def __init__(self, team_a_id: int, team_b_id: int):
        self.team_a_id = team_a_id
        self.team_b_id = team_b_id
        self.relationship = {
            "team_a_id": team_a_id,
            "team_b_id": team_b_id,
            "team_a_name": None,
            "team_b_name": None,
            "total_matches": 0,
            "h2h_patterns": {},  # Head-to-head pattern'leri
            "home_advantage": {},  # Ev sahibi avantajı analizi
            "away_strength": {},  # Deplasman gücü analizi
            "market_relationships": {},  # Her market için ilişki
            "dominance": None,  # Hangi takım dominant
            "relationship_type": None,  # İlişki tipi
            "prediction_algorithm": None  # Bu çift için özel algoritma
        }
    
    def build_relationship_model(
        self,
        matches: List,
        market_types: List[MarketType]
    ) -> Dict[str, Any]:
        """
        İki takım arasındaki ilişki modelini oluştur.
        Ayrı bir algoritma/düşünce sistemi.
        """
        session = get_session()
        try:
            team_a = TeamRepository.get_by_id(session, self.team_a_id)
            team_b = TeamRepository.get_by_id(session, self.team_b_id)
            
            if not team_a or not team_b:
                return {}
            
            self.relationship["team_a_name"] = team_a.name
            self.relationship["team_b_name"] = team_b.name
            
            # Bu iki takım arasındaki maçları bul
            pair_matches = [
                m for m in matches
                if (m.home_team_id == self.team_a_id and m.away_team_id == self.team_b_id) or
                   (m.home_team_id == self.team_b_id and m.away_team_id == self.team_a_id)
            ]
            
            self.relationship["total_matches"] = len(pair_matches)
            
            if not pair_matches:
                return self.relationship
            
            # H2H pattern'leri
            self.relationship["h2h_patterns"] = self._analyze_h2h_patterns(pair_matches, session)
            
            # Ev sahibi avantajı analizi
            self.relationship["home_advantage"] = self._analyze_home_advantage(pair_matches, session)
            
            # Deplasman gücü analizi
            self.relationship["away_strength"] = self._analyze_away_strength(pair_matches, session)
            
            # Her market için ilişki
            for market_type in market_types:
                market_rel = self._analyze_market_relationship(pair_matches, market_type, session)
                self.relationship["market_relationships"][market_type.value] = market_rel
            
            # Dominance belirleme
            self.relationship["dominance"] = self._determine_dominance()
            
            # İlişki tipi
            self.relationship["relationship_type"] = self._classify_relationship()
            
            # Özel algoritma oluştur
            self.relationship["prediction_algorithm"] = self._create_prediction_algorithm()
            
            return self.relationship
        
        finally:
            session.close()
    
    def _analyze_h2h_patterns(self, matches: List, session) -> Dict[str, Any]:
        """Head-to-head pattern'leri"""
        patterns = {
            "team_a_wins": 0,
            "team_b_wins": 0,
            "draws": 0,
            "team_a_goals": 0,
            "team_b_goals": 0,
            "recent_form": []  # Son 5 maç
        }
        
        for match in sorted(matches, key=lambda m: m.match_date, reverse=True)[:10]:
            if match.home_score is None:
                continue
            
            if match.home_team_id == self.team_a_id:
                team_a_score = match.home_score
                team_b_score = match.away_score
            else:
                team_a_score = match.away_score
                team_b_score = match.home_score
            
            patterns["team_a_goals"] += team_a_score
            patterns["team_b_goals"] += team_b_score
            
            if team_a_score > team_b_score:
                patterns["team_a_wins"] += 1
                patterns["recent_form"].append("A")
            elif team_b_score > team_a_score:
                patterns["team_b_wins"] += 1
                patterns["recent_form"].append("B")
            else:
                patterns["draws"] += 1
                patterns["recent_form"].append("D")
        
        return patterns
    
    def _analyze_home_advantage(self, matches: List, session) -> Dict[str, Any]:
        """Ev sahibi avantajı analizi"""
        advantage = {
            "team_a_home": {"wins": 0, "draws": 0, "losses": 0},
            "team_b_home": {"wins": 0, "draws": 0, "losses": 0}
        }
        
        for match in matches:
            if match.home_score is None:
                continue
            
            if match.home_team_id == self.team_a_id:
                if match.home_score > match.away_score:
                    advantage["team_a_home"]["wins"] += 1
                elif match.home_score < match.away_score:
                    advantage["team_a_home"]["losses"] += 1
                else:
                    advantage["team_a_home"]["draws"] += 1
            elif match.home_team_id == self.team_b_id:
                if match.home_score > match.away_score:
                    advantage["team_b_home"]["wins"] += 1
                elif match.home_score < match.away_score:
                    advantage["team_b_home"]["losses"] += 1
                else:
                    advantage["team_b_home"]["draws"] += 1
        
        return advantage
    
    def _analyze_away_strength(self, matches: List, session) -> Dict[str, Any]:
        """Deplasman gücü analizi"""
        strength = {
            "team_a_away": {"wins": 0, "draws": 0, "losses": 0},
            "team_b_away": {"wins": 0, "draws": 0, "losses": 0}
        }
        
        for match in matches:
            if match.home_score is None:
                continue
            
            if match.away_team_id == self.team_a_id:
                if match.away_score > match.home_score:
                    strength["team_a_away"]["wins"] += 1
                elif match.away_score < match.home_score:
                    strength["team_a_away"]["losses"] += 1
                else:
                    strength["team_a_away"]["draws"] += 1
            elif match.away_team_id == self.team_b_id:
                if match.away_score > match.home_score:
                    strength["team_b_away"]["wins"] += 1
                elif match.away_score < match.home_score:
                    strength["team_b_away"]["losses"] += 1
                else:
                    strength["team_b_away"]["draws"] += 1
        
        return strength
    
    def _analyze_market_relationship(
        self,
        matches: List,
        market_type: MarketType,
        session
    ) -> Dict[str, Any]:
        """Bir market için ilişki analizi"""
        market = MarketRepository.get_or_create(session, name=market_type.value)
        
        relationship = {
            "outcomes": Counter(),
            "team_a_favored": 0,
            "team_b_favored": 0,
            "balanced": 0
        }
        
        for match in matches:
            results = ResultRepository.get_by_match(session, match.id)
            result = next((r for r in results if r.market_id == market.id), None)
            
            if result:
                relationship["outcomes"][result.actual_outcome] += 1
        
        # Hangi takım daha avantajlı?
        total = sum(relationship["outcomes"].values())
        if total > 0:
            # Market'e göre analiz
            if market_type == MarketType.MATCH_RESULT:
                team_a_wins = relationship["outcomes"].get("1", 0) + relationship["outcomes"].get("Home", 0)
                team_b_wins = relationship["outcomes"].get("2", 0) + relationship["outcomes"].get("Away", 0)
                
                if team_a_wins > team_b_wins * 1.5:
                    relationship["team_a_favored"] = 1
                elif team_b_wins > team_a_wins * 1.5:
                    relationship["team_b_favored"] = 1
                else:
                    relationship["balanced"] = 1
        
        return relationship
    
    def _determine_dominance(self) -> str:
        """Hangi takım dominant?"""
        h2h = self.relationship["h2h_patterns"]
        total = h2h.get("team_a_wins", 0) + h2h.get("team_b_wins", 0) + h2h.get("draws", 0)
        
        if total == 0:
            return "unknown"
        
        a_wins = h2h.get("team_a_wins", 0)
        b_wins = h2h.get("team_b_wins", 0)
        
        if a_wins > b_wins * 1.5:
            return f"{self.relationship['team_a_name']} dominant"
        elif b_wins > a_wins * 1.5:
            return f"{self.relationship['team_b_name']} dominant"
        else:
            return "balanced"
    
    def _classify_relationship(self) -> str:
        """İlişki tipini sınıflandır"""
        h2h = self.relationship["h2h_patterns"]
        draws = h2h.get("draws", 0)
        total = self.relationship["total_matches"]
        
        if total == 0:
            return "unknown"
        
        draw_rate = draws / total
        
        if draw_rate > 0.4:
            return "rivalry"  # Çok beraberlik
        elif self.relationship["dominance"] != "balanced":
            return "dominance"  # Bir takım dominant
        else:
            return "competitive"  # Rekabetçi
    
    def _create_prediction_algorithm(self) -> Dict[str, Any]:
        """
        Bu takım çifti için özel tahmin algoritması oluştur.
        """
        algorithm = {
            "type": self.relationship["relationship_type"],
            "weights": {},
            "rules": []
        }
        
        # İlişki tipine göre ağırlıklar
        if self.relationship["relationship_type"] == "rivalry":
            algorithm["weights"]["h2h"] = 0.4
            algorithm["weights"]["home_advantage"] = 0.3
            algorithm["weights"]["recent_form"] = 0.3
            algorithm["rules"].append("Beraberlik olasılığı yüksek")
        
        elif self.relationship["relationship_type"] == "dominance":
            algorithm["weights"]["h2h"] = 0.5
            algorithm["weights"]["home_advantage"] = 0.3
            algorithm["weights"]["recent_form"] = 0.2
            algorithm["rules"].append("Dominant takım lehine tahmin")
        
        else:  # competitive
            algorithm["weights"]["h2h"] = 0.3
            algorithm["weights"]["home_advantage"] = 0.4
            algorithm["weights"]["recent_form"] = 0.3
            algorithm["rules"].append("Ev sahibi avantajı önemli")
        
        return algorithm


class PairwiseRelationshipManager:
    """
    Tüm takım çiftleri ilişkilerini yönetir.
    """
    
    def __init__(self):
        self.relationships: Dict[Tuple[int, int], PairwiseRelationship] = {}
    
    def get_or_create_relationship(
        self,
        team_a_id: int,
        team_b_id: int
    ) -> PairwiseRelationship:
        """İlişki al veya oluştur"""
        key = tuple(sorted([team_a_id, team_b_id]))
        if key not in self.relationships:
            self.relationships[key] = PairwiseRelationship(team_a_id, team_b_id)
        return self.relationships[key]
    
    def build_all_relationships(
        self,
        season: int,
        market_types: List[MarketType]
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Tüm takım çiftleri ilişkilerini oluştur"""
        session = get_session()
        try:
            teams = TeamRepository.get_all(session)
            leagues = LeagueRepository.get_all(session)
            
            all_matches = []
            for league in leagues:
                matches = MatchRepository.get_by_league_and_season(session, league.id, season)
                matches = [m for m in matches if m.home_score is not None and m.away_score is not None]
                all_matches.extend(matches)
            
            all_relationships = {}
            
            # Her takım çifti için
            for i, team_a in enumerate(teams):
                for team_b in teams[i+1:]:
                    pair_matches = [
                        m for m in all_matches
                        if (m.home_team_id == team_a.id and m.away_team_id == team_b.id) or
                           (m.home_team_id == team_b.id and m.away_team_id == team_a.id)
                    ]
                    
                    if len(pair_matches) > 0:
                        logger.info(f"🤝 {team_a.name} vs {team_b.name} ilişkisi analiz ediliyor...")
                        
                        relationship = self.get_or_create_relationship(team_a.id, team_b.id)
                        rel_model = relationship.build_relationship_model(pair_matches, market_types)
                        
                        if rel_model:
                            key = tuple(sorted([team_a.id, team_b.id]))
                            all_relationships[key] = rel_model
            
            logger.info(f"✅ {len(all_relationships)} takım çifti ilişkisi oluşturuldu")
            return all_relationships
        
        finally:
            session.close()

