"""
Takım Profili Sistemi
Her takımın en ince ayrıntısına kadar öğrenilmesi:
- Form döngüleri, güçlü/zayıf yönler
- Ev sahibi/deplasman pattern'leri
- Market bazlı davranışlar
- Zaman bazlı trendler
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from collections import defaultdict, Counter
import logging

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    MatchRepository, TeamRepository, ResultRepository, MarketRepository, LeagueRepository
)
from football_brain_core.src.features.market_targets import MarketType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamProfile:
    """
    Bir takımın detaylı profili - her şeyi ezberinde tutar.
    """
    
    def __init__(self, team_id: int):
        self.team_id = team_id
        self.profile = {
            "team_id": team_id,
            "team_name": None,
            "market_profiles": {},  # Her market için ayrı profil
            "home_patterns": {},
            "away_patterns": {},
            "form_cycles": [],  # Form döngüleri
            "strengths": [],
            "weaknesses": [],
            "trends": {},  # Zaman bazlı trendler
            "detailed_stats": {}  # En ince ayrıntılar
        }
    
    def build_comprehensive_profile(
        self,
        matches: List,
        market_types: List[MarketType]
    ) -> Dict[str, Any]:
        """
        Takımın kapsamlı profilini oluştur - en ince ayrıntısına kadar.
        """
        session = get_session()
        try:
            team = TeamRepository.get_by_id(session, self.team_id)
            if not team:
                return {}
            
            self.profile["team_name"] = team.name
            
            # Her market için detaylı profil
            for market_type in market_types:
                market_profile = self._build_market_profile(matches, market_type, session)
                self.profile["market_profiles"][market_type.value] = market_profile
            
            # Ev sahibi pattern'leri (çok detaylı)
            home_matches = [m for m in matches if m.home_team_id == self.team_id]
            self.profile["home_patterns"] = self._analyze_detailed_patterns(
                home_matches, "home", session
            )
            
            # Deplasman pattern'leri (çok detaylı)
            away_matches = [m for m in matches if m.away_team_id == self.team_id]
            self.profile["away_patterns"] = self._analyze_detailed_patterns(
                away_matches, "away", session
            )
            
            # Form döngüleri analizi
            self.profile["form_cycles"] = self._analyze_form_cycles(matches, session)
            
            # Güçlü/zayıf yönler
            self.profile["strengths"], self.profile["weaknesses"] = self._identify_strengths_weaknesses()
            
            # Zaman bazlı trendler
            self.profile["trends"] = self._analyze_trends(matches, session)
            
            # En ince ayrıntılar
            self.profile["detailed_stats"] = self._calculate_detailed_stats(matches, session)
            
            return self.profile
        
        finally:
            session.close()
    
    def _build_market_profile(
        self,
        matches: List,
        market_type: MarketType,
        session
    ) -> Dict[str, Any]:
        """Bir market için detaylı profil"""
        market = MarketRepository.get_or_create(session, name=market_type.value)
        
        profile = {
            "market": market_type.value,
            "outcome_distribution": {},
            "home_outcomes": {},
            "away_outcomes": {},
            "probability_estimates": {},  # Her outcome için olasılık
            "confidence_levels": {},  # Güven seviyeleri
            "contextual_patterns": {}  # Bağlamsal pattern'ler
        }
        
        all_outcomes = []
        home_outcomes = []
        away_outcomes = []
        
        for match in matches:
            results = ResultRepository.get_by_match(session, match.id)
            result = next((r for r in results if r.market_id == market.id), None)
            
            if result:
                outcome = result.actual_outcome
                all_outcomes.append(outcome)
                
                if match.home_team_id == self.team_id:
                    home_outcomes.append(outcome)
                elif match.away_team_id == self.team_id:
                    away_outcomes.append(outcome)
        
        # Olasılık dağılımları
        if all_outcomes:
            outcome_counts = Counter(all_outcomes)
            total = len(all_outcomes)
            profile["outcome_distribution"] = {
                outcome: count / total
                for outcome, count in outcome_counts.items()
            }
            profile["probability_estimates"] = profile["outcome_distribution"]
        
        if home_outcomes:
            home_counts = Counter(home_outcomes)
            home_total = len(home_outcomes)
            profile["home_outcomes"] = {
                outcome: count / home_total
                for outcome, count in home_counts.items()
            }
        
        if away_outcomes:
            away_counts = Counter(away_outcomes)
            away_total = len(away_outcomes)
            profile["away_outcomes"] = {
                outcome: count / away_total
                for outcome, count in away_counts.items()
            }
        
        # Güven seviyeleri (yeterli veri varsa yüksek güven)
        for outcome, prob in profile["probability_estimates"].items():
            sample_size = len(all_outcomes)
            if sample_size >= 20:
                confidence = "high"
            elif sample_size >= 10:
                confidence = "medium"
            else:
                confidence = "low"
            profile["confidence_levels"][outcome] = confidence
        
        return profile
    
    def _analyze_detailed_patterns(
        self,
        matches: List,
        venue: str,
        session
    ) -> Dict[str, Any]:
        """Çok detaylı pattern analizi"""
        patterns = {
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "avg_goals_scored": 0.0,
            "avg_goals_conceded": 0.0,
            "clean_sheet_rate": 0.0,
            "btts_rate": 0.0,
            "over_25_rate": 0.0,
            "streak_patterns": [],
            "time_based_patterns": {}  # Hafta içi/hafta sonu, saat bazlı
        }
        
        if not matches:
            return patterns
        
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        clean_sheets = 0
        btts_count = 0
        over_25_count = 0
        
        for match in matches:
            if match.home_score is None or match.away_score is None:
                continue
            
            if venue == "home":
                team_score = match.home_score
                opponent_score = match.away_score
            else:
                team_score = match.away_score
                opponent_score = match.home_score
            
            goals_scored += team_score
            goals_conceded += opponent_score
            
            if team_score > opponent_score:
                wins += 1
            elif team_score == opponent_score:
                draws += 1
            else:
                losses += 1
            
            if opponent_score == 0:
                clean_sheets += 1
            
            if team_score > 0 and opponent_score > 0:
                btts_count += 1
            
            if team_score + opponent_score > 2.5:
                over_25_count += 1
        
        total = len([m for m in matches if m.home_score is not None])
        if total > 0:
            patterns["win_rate"] = wins / total
            patterns["draw_rate"] = draws / total
            patterns["loss_rate"] = losses / total
            patterns["avg_goals_scored"] = goals_scored / total
            patterns["avg_goals_conceded"] = goals_conceded / total
            patterns["clean_sheet_rate"] = clean_sheets / total
            patterns["btts_rate"] = btts_count / total
            patterns["over_25_rate"] = over_25_count / total
        
        return patterns
    
    def _analyze_form_cycles(self, matches: List, session) -> List[Dict[str, Any]]:
        """Form döngülerini analiz et"""
        cycles = []
        
        # Son 10 maçı grupla
        recent_matches = sorted(matches, key=lambda m: m.match_date, reverse=True)[:30]
        
        if len(recent_matches) < 10:
            return cycles
        
        # 10'ar maçlık dönemler
        for i in range(0, len(recent_matches) - 9, 5):
            period_matches = recent_matches[i:i+10]
            
            wins = 0
            draws = 0
            losses = 0
            
            for match in period_matches:
                if match.home_score is None:
                    continue
                
                if match.home_team_id == self.team_id:
                    if match.home_score > match.away_score:
                        wins += 1
                    elif match.home_score < match.away_score:
                        losses += 1
                    else:
                        draws += 1
                else:
                    if match.away_score > match.home_score:
                        wins += 1
                    elif match.away_score < match.home_score:
                        losses += 1
                    else:
                        draws += 1
            
            cycle = {
                "period": f"{period_matches[-1].match_date.date()} to {period_matches[0].match_date.date()}",
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "form": "good" if wins >= 6 else "bad" if losses >= 6 else "average"
            }
            cycles.append(cycle)
        
        return cycles
    
    def _identify_strengths_weaknesses(self) -> tuple:
        """Güçlü ve zayıf yönleri belirle"""
        strengths = []
        weaknesses = []
        
        # Ev sahibi güçlü yönler
        if self.profile["home_patterns"].get("win_rate", 0) > 0.6:
            strengths.append(f"Ev sahibi güçlü: {self.profile['home_patterns']['win_rate']:.1%} kazanma oranı")
        
        if self.profile["home_patterns"].get("clean_sheet_rate", 0) > 0.4:
            strengths.append(f"Ev sahibi defans güçlü: {self.profile['home_patterns']['clean_sheet_rate']:.1%} temiz sayfa")
        
        # Deplasman zayıf yönler
        if self.profile["away_patterns"].get("loss_rate", 0) > 0.5:
            weaknesses.append(f"Deplasman zayıf: {self.profile['away_patterns']['loss_rate']:.1%} kayıp oranı")
        
        if self.profile["away_patterns"].get("avg_goals_conceded", 0) > 2.0:
            weaknesses.append(f"Deplasman defans zayıf: {self.profile['away_patterns']['avg_goals_conceded']:.1f} gol yiyor")
        
        return strengths, weaknesses
    
    def _analyze_trends(self, matches: List, session) -> Dict[str, Any]:
        """Zaman bazlı trendler"""
        trends = {
            "recent_form": "unknown",
            "momentum": "neutral",
            "goal_trend": "stable"
        }
        
        if len(matches) < 5:
            return trends
        
        recent = sorted(matches, key=lambda m: m.match_date, reverse=True)[:5]
        older = sorted(matches, key=lambda m: m.match_date, reverse=True)[5:10] if len(matches) >= 10 else []
        
        # Son 5 maç formu
        recent_wins = sum(1 for m in recent if self._team_won(m))
        if recent_wins >= 4:
            trends["recent_form"] = "excellent"
        elif recent_wins >= 3:
            trends["recent_form"] = "good"
        elif recent_wins <= 1:
            trends["recent_form"] = "poor"
        
        # Momentum (son 5 vs önceki 5)
        if older:
            older_wins = sum(1 for m in older if self._team_won(m))
            if recent_wins > older_wins:
                trends["momentum"] = "improving"
            elif recent_wins < older_wins:
                trends["momentum"] = "declining"
        
        return trends
    
    def _team_won(self, match) -> bool:
        """Takım kazandı mı?"""
        if match.home_score is None:
            return False
        
        if match.home_team_id == self.team_id:
            return match.home_score > match.away_score
        else:
            return match.away_score > match.home_score
    
    def _calculate_detailed_stats(self, matches: List, session) -> Dict[str, Any]:
        """En ince ayrıntılar"""
        stats = {
            "total_matches": len(matches),
            "home_matches": len([m for m in matches if m.home_team_id == self.team_id]),
            "away_matches": len([m for m in matches if m.away_team_id == self.team_id]),
            "avg_goals_per_match": 0.0,
            "avg_conceded_per_match": 0.0,
            "goal_difference": 0,
            "most_common_scoreline": None
        }
        
        scorelines = Counter()
        total_goals_scored = 0
        total_goals_conceded = 0
        
        for match in matches:
            if match.home_score is None:
                continue
            
            if match.home_team_id == self.team_id:
                score = match.home_score
                conceded = match.away_score
            else:
                score = match.away_score
                conceded = match.home_score
            
            total_goals_scored += score
            total_goals_conceded += conceded
            scorelines[f"{score}-{conceded}"] += 1
        
        if stats["total_matches"] > 0:
            stats["avg_goals_per_match"] = total_goals_scored / stats["total_matches"]
            stats["avg_conceded_per_match"] = total_goals_conceded / stats["total_matches"]
            stats["goal_difference"] = total_goals_scored - total_goals_conceded
        
        if scorelines:
            stats["most_common_scoreline"] = scorelines.most_common(1)[0][0]
        
        return stats


class TeamProfileManager:
    """
    Tüm takım profillerini yönetir - her takım için detaylı profil tutar.
    """
    
    def __init__(self):
        self.profiles: Dict[int, TeamProfile] = {}
    
    def get_or_create_profile(self, team_id: int) -> TeamProfile:
        """Takım profili al veya oluştur"""
        if team_id not in self.profiles:
            self.profiles[team_id] = TeamProfile(team_id)
        return self.profiles[team_id]
    
    def build_all_profiles(
        self,
        season: int,
        market_types: List[MarketType]
    ) -> Dict[int, Dict[str, Any]]:
        """Tüm takımların profillerini oluştur"""
        session = get_session()
        try:
            teams = TeamRepository.get_all(session)
            leagues = LeagueRepository.get_all(session)
            
            all_matches = []
            for league in leagues:
                matches = MatchRepository.get_by_league_and_season(session, league.id, season)
                matches = [m for m in matches if m.home_score is not None and m.away_score is not None]
                all_matches.extend(matches)
            
            all_profiles = {}
            
            for team in teams:
                logger.info(f"📊 {team.name} profili oluşturuluyor...")
                
                team_matches = [
                    m for m in all_matches
                    if m.home_team_id == team.id or m.away_team_id == team.id
                ]
                
                profile = self.get_or_create_profile(team.id)
                team_profile = profile.build_comprehensive_profile(team_matches, market_types)
                
                if team_profile:
                    all_profiles[team.id] = team_profile
            
            logger.info(f"✅ {len(all_profiles)} takım profili oluşturuldu")
            return all_profiles
        
        finally:
            session.close()







