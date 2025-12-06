"""
Hata Analizi ve Öğrenme Mekanizması
- Her hatayı kategorize eder (bias, variance, feature eksikliği, vb.)
- Sapma payı, bias, feature importance öğrenir
- Formülüne/feature'larına eklemeler yapar
- Kendini evrimleştirir
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from collections import defaultdict

from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    MatchRepository, ResultRepository, TeamRepository, MarketRepository
)
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.features.feature_builder import FeatureBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Hataları analiz eder, kategorize eder ve öğrenme için bilgi çıkarır.
    """
    
    def __init__(self, feature_builder: FeatureBuilder):
        self.feature_builder = feature_builder
        self.error_patterns = defaultdict(list)
        self.feature_importance = defaultdict(float)
        self.bias_detection = defaultdict(float)
        self.variance_analysis = defaultdict(list)
    
    def analyze_error(
        self,
        match_id: int,
        market_type: MarketType,
        predicted: str,
        actual: str,
        predicted_proba: Dict[str, float],
        session
    ) -> Dict[str, Any]:
        """
        Bir hatayı detaylı analiz eder ve kategorize eder.
        
        Returns:
            Hata analizi sonuçları (hata türü, nedeni, çözüm önerisi)
        """
        match = MatchRepository.get_by_id(session, match_id)
        if not match:
            return {}
        
        home_team = TeamRepository.get_by_id(session, match.home_team_id)
        away_team = TeamRepository.get_by_id(session, match.away_team_id)
        
        # Feature'ları al
        features = self.feature_builder.build_match_features(
            match.home_team_id,
            match.away_team_id,
            match.match_date,
            match.league_id,
            session
        )
        
        error_analysis = {
            "error_type": None,
            "error_category": None,
            "confidence": predicted_proba.get(predicted, 0.0),
            "actual_probability": predicted_proba.get(actual, 0.0),
            "bias_detected": False,
            "variance_issue": False,
            "missing_features": [],
            "suggested_corrections": [],
            "feature_contributions": {}
        }
        
        # 1. Confidence analizi - Düşük güven ile yanlış tahmin = variance problemi
        if error_analysis["confidence"] < 0.4:
            error_analysis["error_category"] = "low_confidence"
            error_analysis["variance_issue"] = True
            error_analysis["suggested_corrections"].append("Model belirsiz, daha fazla feature gerekli")
        
        # 2. Bias analizi - Yüksek güven ile yanlış tahmin = bias problemi
        elif error_analysis["confidence"] > 0.7:
            error_analysis["error_category"] = "high_confidence_error"
            error_analysis["bias_detected"] = True
            error_analysis["suggested_corrections"].append("Model yanlış pattern öğrenmiş, bias düzeltmesi gerekli")
        
        # 3. Takım bazlı bias
        team_bias = self._detect_team_bias(home_team.id if home_team else None, away_team.id if away_team else None, market_type, session)
        if team_bias:
            error_analysis["bias_detected"] = True
            error_analysis["team_bias"] = team_bias
            error_analysis["suggested_corrections"].append(f"Takım bazlı bias tespit edildi: {team_bias}")
        
        # 4. Feature importance analizi
        feature_contrib = self._analyze_feature_contribution(features, predicted, actual, market_type)
        error_analysis["feature_contributions"] = feature_contrib
        
        # Eksik feature tespiti
        missing = self._detect_missing_features(match, features, predicted, actual, session)
        if missing:
            error_analysis["missing_features"] = missing
            error_analysis["suggested_corrections"].append(f"Eksik feature'lar: {', '.join(missing)}")
        
        # 5. Pattern analizi
        pattern_error = self._analyze_pattern_error(match, predicted, actual, market_type, session)
        if pattern_error:
            error_analysis["pattern_error"] = pattern_error
            error_analysis["suggested_corrections"].append(f"Pattern hatası: {pattern_error}")
        
        # 6. Sapma payı hesaplama
        deviation = self._calculate_deviation(predicted_proba, predicted, actual)
        error_analysis["deviation"] = deviation
        error_analysis["suggested_corrections"].append(f"Sapma payı: {deviation:.2%}")
        
        return error_analysis
    
    def _detect_team_bias(
        self,
        home_team_id: Optional[int],
        away_team_id: Optional[int],
        market_type: MarketType,
        session
    ) -> Optional[Dict[str, Any]]:
        """Takım bazlı bias tespit eder"""
        if not home_team_id or not away_team_id:
            return None
        
        # Bu takımlar için geçmiş hataları kontrol et
        home_matches = MatchRepository.get_team_matches(session, home_team_id, limit=20)
        away_matches = MatchRepository.get_team_matches(session, away_team_id, limit=20)
        
        home_errors = 0
        away_errors = 0
        
        for match in home_matches:
            if match.home_score is None:
                continue
            results = ResultRepository.get_by_match(session, match.id)
            market = MarketRepository.get_or_create(session, name=market_type.value)
            result = next((r for r in results if r.market_id == market.id), None)
            if result:
                # Basit bias kontrolü - bu takım için sürekli yanlış tahmin var mı?
                home_errors += 1
        
        if home_errors > 5 or away_errors > 5:
            return {
                "home_team_bias": home_errors,
                "away_team_bias": away_errors,
                "suggestion": "Takım bazlı feature'lar güçlendirilmeli"
            }
        
        return None
    
    def _analyze_feature_contribution(
        self,
        features: np.ndarray,
        predicted: str,
        actual: str,
        market_type: MarketType
    ) -> Dict[str, float]:
        """Feature'ların hataya katkısını analiz eder"""
        # Feature importance tracking
        feature_names = [
            "home_avg_goals_scored", "home_avg_goals_conceded",
            "away_avg_goals_scored", "away_avg_goals_conceded",
            "home_win_rate", "away_win_rate",
            "home_btts_rate", "away_btts_rate",
            "home_over_25_rate", "away_over_25_rate",
            "league_id"
        ]
        
        contributions = {}
        for i, feat_name in enumerate(feature_names):
            if i < len(features):
                # Feature değerinin hataya etkisini hesapla
                feat_value = features[i]
                # Basit heuristik: Aşırı değerler hataya neden olabilir
                if abs(feat_value) > 2.0:  # Aşırı değer
                    contributions[feat_name] = abs(feat_value)
        
        return contributions
    
    def _detect_missing_features(
        self,
        match,
        features: np.ndarray,
        predicted: str,
        actual: str,
        session
    ) -> List[str]:
        """Eksik feature'ları tespit eder"""
        missing = []
        
        # H2H (head-to-head) eksik mi?
        h2h_matches = self._get_h2h_matches(match.home_team_id, match.away_team_id, match.match_date, session)
        if len(h2h_matches) > 0 and not any("h2h" in str(f) for f in features):
            missing.append("h2h_history")
        
        # Form trend eksik mi?
        home_recent = MatchRepository.get_team_matches(session, match.home_team_id, limit=5)
        if len(home_recent) >= 5:
            # Form trendi var mı kontrol et
            trend = self._calculate_form_trend(home_recent)
            if trend != 0 and not any("trend" in str(f) for f in features):
                missing.append("form_trend")
        
        # Takım kimyası eksik mi?
        # (Aynı takımlar sık karşılaşıyorsa özel pattern olabilir)
        if len(h2h_matches) > 3:
            missing.append("team_chemistry")
        
        return missing
    
    def _analyze_pattern_error(
        self,
        match,
        predicted: str,
        actual: str,
        market_type: MarketType,
        session
    ) -> Optional[str]:
        """Pattern hatası analizi"""
        # Örnek: Ev sahibi avantajı göz ardı edilmiş mi?
        if market_type == MarketType.MATCH_RESULT:
            if predicted == "2" and actual == "1":  # Deplasman tahmin, ev kazandı
                home_matches = MatchRepository.get_team_matches(session, match.home_team_id, limit=10)
                home_wins = sum(1 for m in home_matches if m.home_team_id == match.home_team_id and m.home_score and m.away_score and m.home_score > m.away_score)
                if home_wins > 7:  # Ev sahibi güçlü
                    return "Ev sahibi avantajı göz ardı edilmiş"
        
        return None
    
    def _calculate_deviation(
        self,
        predicted_proba: Dict[str, float],
        predicted: str,
        actual: str
    ) -> float:
        """Sapma payı hesaplar"""
        pred_prob = predicted_proba.get(predicted, 0.0)
        actual_prob = predicted_proba.get(actual, 0.0)
        
        # Sapma = tahmin edilen olasılık ile gerçek sonuç arasındaki fark
        deviation = abs(pred_prob - (1.0 if predicted == actual else 0.0))
        
        return deviation
    
    def _get_h2h_matches(self, home_id: int, away_id: int, before_date, session) -> List:
        """Head-to-head maçları getirir"""
        matches = MatchRepository.get_team_matches(session, home_id, limit=100)
        h2h = [
            m for m in matches
            if (m.home_team_id == home_id and m.away_team_id == away_id) or
               (m.home_team_id == away_id and m.away_team_id == home_id)
            and m.match_date < before_date
        ]
        return h2h
    
    def _calculate_form_trend(self, recent_matches: List) -> float:
        """Form trendini hesaplar (pozitif = yükseliş, negatif = düşüş)"""
        if len(recent_matches) < 3:
            return 0.0
        
        # Son 3 maçın gol ortalaması vs ilk 3 maçın
        first_half = recent_matches[:3]
        second_half = recent_matches[-3:]
        
        first_goals = sum(m.home_score or 0 for m in first_half) / len(first_half)
        second_goals = sum(m.home_score or 0 for m in second_half) / len(second_half)
        
        return second_goals - first_goals
    
    def collect_errors(
        self,
        errors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Toplu hata analizi ve öğrenme"""
        error_summary = {
            "total_errors": len(errors),
            "error_types": defaultdict(int),
            "bias_count": 0,
            "variance_count": 0,
            "missing_features_all": defaultdict(int),
            "feature_importance_updates": {},
            "suggested_improvements": []
        }
        
        for error in errors:
            analysis = error.get("analysis", {})
            
            # Hata türlerini say
            error_type = analysis.get("error_category", "unknown")
            error_summary["error_types"][error_type] += 1
            
            # Bias/Variance say
            if analysis.get("bias_detected"):
                error_summary["bias_count"] += 1
            if analysis.get("variance_issue"):
                error_summary["variance_count"] += 1
            
            # Eksik feature'ları topla
            for missing_feat in analysis.get("missing_features", []):
                error_summary["missing_features_all"][missing_feat] += 1
            
            # Feature importance güncelle
            for feat, contrib in analysis.get("feature_contributions", {}).items():
                if feat not in error_summary["feature_importance_updates"]:
                    error_summary["feature_importance_updates"][feat] = 0.0
                error_summary["feature_importance_updates"][feat] += contrib
        
        # Öneriler oluştur
        if error_summary["bias_count"] > error_summary["total_errors"] * 0.3:
            error_summary["suggested_improvements"].append(
                "Bias problemi tespit edildi: Regularization artırılmalı veya feature engineering yapılmalı"
            )
        
        if error_summary["variance_count"] > error_summary["total_errors"] * 0.3:
            error_summary["suggested_improvements"].append(
                "Variance problemi tespit edildi: Daha fazla training data veya model complexity azaltılmalı"
            )
        
        # En çok eksik olan feature'lar
        if error_summary["missing_features_all"]:
            top_missing = sorted(
                error_summary["missing_features_all"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            error_summary["suggested_improvements"].append(
                f"Eksik feature'lar eklenmeli: {', '.join([f[0] for f in top_missing])}"
            )
        
        return error_summary







