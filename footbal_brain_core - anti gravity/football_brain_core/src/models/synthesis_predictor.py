"""
Sentez Tahmin Sistemi
Takım profili + İkili ilişki → Sentezlenmiş tahmin
Her iki bilgiyi birleştirerek düşük hata oranı için optimize edilmiş tahmin.
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

from football_brain_core.src.models.team_profile import TeamProfileManager, TeamProfile
from football_brain_core.src.models.pairwise_relationship import PairwiseRelationshipManager, PairwiseRelationship
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import MatchRepository, TeamRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynthesisPredictor:
    """
    Takım profili ve ikili ilişkiyi sentezleyerek tahmin yapar.
    Düşük hata oranı için optimize edilmiş.
    """
    
    def __init__(
        self,
        team_profile_manager: TeamProfileManager,
        pairwise_manager: PairwiseRelationshipManager
    ):
        self.team_profiles = team_profile_manager
        self.pairwise_relationships = pairwise_manager
    
    def predict_with_synthesis(
        self,
        match_id: int,
        market_type: MarketType,
        base_prediction: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Sentezlenmiş tahmin yap:
        1. Takım profillerinden bilgi al
        2. İkili ilişkiden bilgi al
        3. İkisini sentezle
        4. Optimize edilmiş tahmin döndür
        """
        session = get_session()
        try:
            match = MatchRepository.get_by_id(session, match_id)
            if not match:
                return {}
            
            home_team = TeamRepository.get_by_id(session, match.home_team_id)
            away_team = TeamRepository.get_by_id(session, match.away_team_id)
            
            if not home_team or not away_team:
                return {}
            
            # 1. Takım profillerinden bilgi al
            home_profile = self.team_profiles.get_or_create_profile(match.home_team_id)
            away_profile = self.team_profiles.get_or_create_profile(match.away_team_id)
            
            home_info = self._extract_profile_info(home_profile, market_type, "home")
            away_info = self._extract_profile_info(away_profile, market_type, "away")
            
            # 2. İkili ilişkiden bilgi al
            relationship = self.pairwise_relationships.get_or_create_relationship(
                match.home_team_id, match.away_team_id
            )
            relationship_info = self._extract_relationship_info(relationship, market_type)
            
            # 3. Sentezle
            synthesis = self._synthesize_predictions(
                home_info,
                away_info,
                relationship_info,
                market_type
            )
            
            # 4. Base prediction ile birleştir (eğer varsa)
            if base_prediction:
                final_prediction = self._combine_with_base(synthesis, base_prediction)
            else:
                final_prediction = synthesis
            
            return {
                "prediction": final_prediction,
                "confidence": self._calculate_confidence(home_info, away_info, relationship_info),
                "reasoning": self._generate_reasoning(home_info, away_info, relationship_info),
                "sources": {
                    "home_profile": home_info,
                    "away_profile": away_info,
                    "relationship": relationship_info
                }
            }
        
        finally:
            session.close()
    
    def _extract_profile_info(
        self,
        profile: TeamProfile,
        market_type: MarketType,
        venue: str
    ) -> Dict[str, Any]:
        """Takım profilinden bilgi çıkar"""
        info = {
            "team_id": profile.team_id,
            "team_name": profile.profile.get("team_name", ""),
            "market_probabilities": {},
            "venue_patterns": {},
            "trends": {},
            "strengths": [],
            "weaknesses": []
        }
        
        # Market profili
        market_profile = profile.profile.get("market_profiles", {}).get(market_type.value, {})
        info["market_probabilities"] = market_profile.get("probability_estimates", {})
        info["confidence_levels"] = market_profile.get("confidence_levels", {})
        
        # Venue pattern'leri
        if venue == "home":
            info["venue_patterns"] = profile.profile.get("home_patterns", {})
        else:
            info["venue_patterns"] = profile.profile.get("away_patterns", {})
        
        # Trendler
        info["trends"] = profile.profile.get("trends", {})
        
        # Güçlü/zayıf yönler
        info["strengths"] = profile.profile.get("strengths", [])
        info["weaknesses"] = profile.profile.get("weaknesses", [])
        
        return info
    
    def _extract_relationship_info(
        self,
        relationship: PairwiseRelationship,
        market_type: MarketType
    ) -> Dict[str, Any]:
        """İkili ilişkiden bilgi çıkar"""
        info = {
            "h2h_patterns": relationship.relationship.get("h2h_patterns", {}),
            "home_advantage": relationship.relationship.get("home_advantage", {}),
            "away_strength": relationship.relationship.get("away_strength", {}),
            "market_relationship": relationship.relationship.get("market_relationships", {}).get(market_type.value, {}),
            "dominance": relationship.relationship.get("dominance", ""),
            "relationship_type": relationship.relationship.get("relationship_type", ""),
            "algorithm": relationship.relationship.get("prediction_algorithm", {})
        }
        
        return info
    
    def _synthesize_predictions(
        self,
        home_info: Dict[str, Any],
        away_info: Dict[str, Any],
        relationship_info: Dict[str, Any],
        market_type: MarketType
    ) -> Dict[str, Any]:
        """
        İki bilgiyi sentezle:
        - Takım profilleri: Genel davranış
        - İkili ilişki: Özel dinamikler
        """
        synthesis = {
            "outcome_probabilities": {},
            "primary_outcome": None,
            "secondary_outcome": None
        }
        
        # Ağırlıklar (ilişki tipine göre)
        algorithm = relationship_info.get("algorithm", {})
        weights = algorithm.get("weights", {
            "profile": 0.5,
            "relationship": 0.5
        })
        
        # Profil bazlı olasılıklar
        home_probs = home_info.get("market_probabilities", {})
        away_probs = away_info.get("market_probabilities", {})
        
        # İlişki bazlı olasılıklar
        rel_probs = relationship_info.get("market_relationship", {}).get("outcomes", {})
        total_rel = sum(rel_probs.values()) if isinstance(rel_probs, dict) else 0
        
        # Market'e göre sentez
        if market_type == MarketType.MATCH_RESULT:
            # Profil olasılıkları
            home_win_prob = home_info.get("venue_patterns", {}).get("win_rate", 0.0) * weights["profile"]
            away_win_prob = away_info.get("venue_patterns", {}).get("win_rate", 0.0) * weights["profile"]
            draw_prob = (home_info.get("venue_patterns", {}).get("draw_rate", 0.0) + 
                        away_info.get("venue_patterns", {}).get("draw_rate", 0.0)) / 2 * weights["profile"]
            
            # İlişki olasılıkları
            if total_rel > 0:
                rel_home_wins = (rel_probs.get("1", 0) + rel_probs.get("Home", 0)) / total_rel
                rel_away_wins = (rel_probs.get("2", 0) + rel_probs.get("Away", 0)) / total_rel
                rel_draws = rel_probs.get("X", 0) / total_rel
                
                home_win_prob += rel_home_wins * weights["relationship"]
                away_win_prob += rel_away_wins * weights["relationship"]
                draw_prob += rel_draws * weights["relationship"]
            
            # Normalize et
            total = home_win_prob + draw_prob + away_win_prob
            if total > 0:
                synthesis["outcome_probabilities"] = {
                    "1": home_win_prob / total,
                    "X": draw_prob / total,
                    "2": away_win_prob / total
                }
                
                # En yüksek olasılıklı outcome
                sorted_outcomes = sorted(
                    synthesis["outcome_probabilities"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                synthesis["primary_outcome"] = sorted_outcomes[0][0]
                if len(sorted_outcomes) > 1:
                    synthesis["secondary_outcome"] = sorted_outcomes[1][0]
        
        return synthesis
    
    def _combine_with_base(
        self,
        synthesis: Dict[str, Any],
        base_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Base prediction ile birleştir (ensemble)"""
        # Base prediction ağırlığı: 0.3
        # Synthesis ağırlığı: 0.7
        
        combined = {
            "outcome_probabilities": {},
            "primary_outcome": synthesis.get("primary_outcome"),
            "confidence": synthesis.get("confidence", 0.0)
        }
        
        # Olasılıkları birleştir
        base_probs = base_prediction.get("probabilities", {})
        synth_probs = synthesis.get("outcome_probabilities", {})
        
        all_outcomes = set(list(base_probs.keys()) + list(synth_probs.keys()))
        
        for outcome in all_outcomes:
            base_prob = base_probs.get(outcome, 0.0) * 0.3
            synth_prob = synth_probs.get(outcome, 0.0) * 0.7
            combined["outcome_probabilities"][outcome] = base_prob + synth_prob
        
        # Normalize
        total = sum(combined["outcome_probabilities"].values())
        if total > 0:
            combined["outcome_probabilities"] = {
                k: v / total
                for k, v in combined["outcome_probabilities"].items()
            }
            
            # En yüksek olasılıklı outcome
            sorted_outcomes = sorted(
                combined["outcome_probabilities"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            combined["primary_outcome"] = sorted_outcomes[0][0]
        
        return combined
    
    def _calculate_confidence(
        self,
        home_info: Dict[str, Any],
        away_info: Dict[str, Any],
        relationship_info: Dict[str, Any]
    ) -> float:
        """Güven seviyesi hesapla"""
        confidence = 0.5  # Base
        
        # Profil güveni
        home_conf = home_info.get("confidence_levels", {})
        away_conf = away_info.get("confidence_levels", {})
        
        if "high" in home_conf.values():
            confidence += 0.1
        if "high" in away_conf.values():
            confidence += 0.1
        
        # İlişki güveni
        total_matches = relationship_info.get("h2h_patterns", {}).get("total_matches", 0)
        if total_matches >= 5:
            confidence += 0.2
        elif total_matches >= 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(
        self,
        home_info: Dict[str, Any],
        away_info: Dict[str, Any],
        relationship_info: Dict[str, Any]
    ) -> str:
        """Açıklama oluştur"""
        reasoning_parts = []
        
        # Takım profili bilgileri
        home_name = home_info.get("team_name", "Home Team")
        away_name = away_info.get("team_name", "Away Team")
        
        home_trend = home_info.get("trends", {}).get("recent_form", "unknown")
        away_trend = away_info.get("trends", {}).get("recent_form", "unknown")
        
        if home_trend == "good":
            reasoning_parts.append(f"{home_name} son dönemde iyi formda")
        elif home_trend == "poor":
            reasoning_parts.append(f"{home_name} son dönemde zayıf formda")
        
        if away_trend == "good":
            reasoning_parts.append(f"{away_name} son dönemde iyi formda")
        elif away_trend == "poor":
            reasoning_parts.append(f"{away_name} son dönemde zayıf formda")
        
        # İlişki bilgileri
        dominance = relationship_info.get("dominance", "")
        if dominance and dominance != "balanced":
            reasoning_parts.append(f"Geçmiş maçlarda {dominance}")
        
        rel_type = relationship_info.get("relationship_type", "")
        if rel_type == "rivalry":
            reasoning_parts.append("Bu iki takım arasında rekabetçi bir ilişki var")
        
        return ". ".join(reasoning_parts) if reasoning_parts else "Genel analiz"







