from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

from src.db.connection import get_session
from src.db.repositories import MatchRepository, TeamRepository
from src.db.schema import Match, Team

logger = logging.getLogger(__name__)


class FeatureBuilder:
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
    
    def build_team_features(self, team_id: int, match_date: datetime, session) -> Dict[str, float]:
        matches = MatchRepository.get_team_matches(session, team_id, limit=self.sequence_length)
        
        if not matches:
            return self._empty_features()
        
        home_matches = [m for m in matches if m.home_team_id == team_id and m.match_date < match_date]
        away_matches = [m for m in matches if m.away_team_id == team_id and m.match_date < match_date]
        all_matches = sorted(
            [m for m in matches if m.match_date < match_date],
            key=lambda x: x.match_date,
            reverse=True
        )[:self.sequence_length]
        
        features = {}
        
        if all_matches:
            goals_scored = []
            goals_conceded = []
            results = []
            
            for match in all_matches:
                if match.home_score is None or match.away_score is None:
                    continue
                
                if match.home_team_id == team_id:
                    goals_scored.append(match.home_score)
                    goals_conceded.append(match.away_score)
                    if match.home_score > match.away_score:
                        results.append(1)
                    elif match.home_score < match.away_score:
                        results.append(-1)
                    else:
                        results.append(0)
                else:
                    goals_scored.append(match.away_score)
                    goals_conceded.append(match.home_score)
                    if match.away_score > match.home_score:
                        results.append(1)
                    elif match.away_score < match.home_score:
                        results.append(-1)
                    else:
                        results.append(0)
            
            if goals_scored:
                features["avg_goals_scored"] = np.mean(goals_scored)
                features["avg_goals_conceded"] = np.mean(goals_conceded)
                features["total_goals_scored"] = sum(goals_scored)
                features["total_goals_conceded"] = sum(goals_conceded)
                features["win_rate"] = sum(1 for r in results if r == 1) / len(results)
                features["draw_rate"] = sum(1 for r in results if r == 0) / len(results)
                features["loss_rate"] = sum(1 for r in results if r == -1) / len(results)
                features["btts_rate"] = sum(
                    1 for i in range(len(goals_scored))
                    if goals_scored[i] > 0 and goals_conceded[i] > 0
                ) / len(goals_scored) if goals_scored else 0.0
                features["over_25_rate"] = sum(
                    1 for i in range(len(goals_scored))
                    if goals_scored[i] + goals_conceded[i] > 2.5
                ) / len(goals_scored) if goals_scored else 0.0
            else:
                features.update(self._empty_features())
        else:
            features.update(self._empty_features())
        
        if home_matches:
            home_goals_scored = [
                m.home_score for m in home_matches
                if m.home_score is not None
            ]
            home_goals_conceded = [
                m.away_score for m in home_matches
                if m.away_score is not None
            ]
            
            if home_goals_scored:
                features["home_avg_goals_scored"] = np.mean(home_goals_scored)
                features["home_avg_goals_conceded"] = np.mean(home_goals_conceded)
            else:
                features["home_avg_goals_scored"] = 0.0
                features["home_avg_goals_conceded"] = 0.0
        else:
            features["home_avg_goals_scored"] = 0.0
            features["home_avg_goals_conceded"] = 0.0
        
        if away_matches:
            away_goals_scored = [
                m.away_score for m in away_matches
                if m.away_score is not None
            ]
            away_goals_conceded = [
                m.home_score for m in away_matches
                if m.home_score is not None
            ]
            
            if away_goals_scored:
                features["away_avg_goals_scored"] = np.mean(away_goals_scored)
                features["away_avg_goals_conceded"] = np.mean(away_goals_conceded)
            else:
                features["away_avg_goals_scored"] = 0.0
                features["away_avg_goals_conceded"] = 0.0
        else:
            features["away_avg_goals_scored"] = 0.0
            features["away_avg_goals_conceded"] = 0.0
        
        return features
    
    def build_match_features(
        self,
        home_team_id: int,
        away_team_id: int,
        match_date: datetime,
        league_id: int,
        session
    ) -> np.ndarray:
        home_features = self.build_team_features(home_team_id, match_date, session)
        away_features = self.build_team_features(away_team_id, match_date, session)
        
        feature_vector = []
        
        feature_order = [
            "avg_goals_scored", "avg_goals_conceded",
            "win_rate", "draw_rate", "loss_rate",
            "btts_rate", "over_25_rate",
            "home_avg_goals_scored", "home_avg_goals_conceded",
            "away_avg_goals_scored", "away_avg_goals_conceded"
        ]
        
        for feat in feature_order:
            feature_vector.append(home_features.get(feat, 0.0))
            feature_vector.append(away_features.get(feat, 0.0))
        
        feature_vector.append(league_id / 1000.0)
        
        return np.array(feature_vector, dtype=np.float32)
    
    def _empty_features(self) -> Dict[str, float]:
        return {
            "avg_goals_scored": 0.0,
            "avg_goals_conceded": 0.0,
            "total_goals_scored": 0.0,
            "total_goals_conceded": 0.0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "btts_rate": 0.0,
            "over_25_rate": 0.0,
        }


