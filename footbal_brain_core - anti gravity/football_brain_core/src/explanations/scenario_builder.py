from typing import Dict, Any, List, Optional
import logging

from football_brain_core.src.explanations.llm_client import LLMClient
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    MatchRepository, TeamRepository, ExplanationRepository, MarketRepository
)
from football_brain_core.src.db.schema import Match
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.features.feature_builder import FeatureBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioBuilder:
    def __init__(self, llm_client: Optional[LLMClient] = None, use_both_models: bool = True):
        """
        Args:
            llm_client: Özel LLM client (None ise varsayılan kullanılır)
            use_both_models: True ise hem GPT hem Grok kullanılır ve karşılaştırılır
        """
        self.use_both_models = use_both_models
        if llm_client is None:
            self.gpt_client = LLMClient(provider="openrouter")
            self.grok_client = LLMClient(provider="openrouter-grok")
            self.llm_client = self.gpt_client  # Varsayılan
        else:
            self.llm_client = llm_client
            self.gpt_client = llm_client
            self.grok_client = llm_client
        self.feature_builder = FeatureBuilder()
    
    def build_summary_stats(self, match: Match, session) -> Dict[str, Any]:
        home_features = self.feature_builder.build_team_features(
            match.home_team_id, match.match_date, session
        )
        away_features = self.feature_builder.build_team_features(
            match.away_team_id, match.match_date, session
        )
        
        return {
            "home_form": f"{home_features.get('win_rate', 0):.1%} win rate",
            "away_form": f"{away_features.get('win_rate', 0):.1%} win rate",
            "home_avg_goals": f"{home_features.get('avg_goals_scored', 0):.2f}",
            "away_avg_goals": f"{away_features.get('avg_goals_scored', 0):.2f}",
            "home_avg_conceded": f"{home_features.get('avg_goals_conceded', 0):.2f}",
            "away_avg_conceded": f"{away_features.get('avg_goals_conceded', 0):.2f}",
            "btts_rate": f"{home_features.get('btts_rate', 0):.1%}",
        }
    
    def generate_explanation(
        self,
        match: Match,
        predicted_outcomes: Dict[MarketType, str],
        market_types: List[MarketType]
    ) -> Dict[MarketType, Dict[str, Any]]:
        """
        Her iki modeli de çalıştırır ve sonuçları karşılaştırır
        
        Returns:
            Dict[MarketType, Dict] - Her market için:
                - "gpt_explanation": GPT modelinin açıklaması
                - "grok_explanation": Grok modelinin açıklaması
                - "gpt_time": GPT'nin süresi (saniye)
                - "grok_time": Grok'un süresi (saniye)
                - "best_model": En hızlı model ("gpt" veya "grok")
                - "explanation": Kullanılacak açıklama (en hızlı olan)
        """
        import time
        session = get_session()
        try:
            home_team = TeamRepository.get_by_id(session, match.home_team_id)
            away_team = TeamRepository.get_by_id(session, match.away_team_id)
            
            match_context = {
                "home_team": home_team.name if home_team else "Home Team",
                "away_team": away_team.name if away_team else "Away Team",
                "match_date": match.match_date.isoformat(),
            }
            
            summary_stats = self.build_summary_stats(match, session)
            
            predicted_dict = {
                market.value: predicted_outcomes.get(market, "N/A")
                for market in market_types
            }
            
            explanations = {}
            
            if self.use_both_models:
                # Her iki modeli de çalıştır ve karşılaştır
                for market_type in market_types:
                    # GPT ile açıklama
                    gpt_start = time.time()
                    try:
                        gpt_explanation = self.gpt_client.generate_explanation(
                            match_context, predicted_dict, summary_stats
                        )
                        gpt_time = time.time() - gpt_start
                    except Exception as e:
                        logger.error(f"GPT açıklama hatası: {e}")
                        gpt_explanation = "GPT açıklama oluşturulamadı"
                        gpt_time = 999
                    
                    # Grok ile açıklama
                    grok_start = time.time()
                    try:
                        grok_explanation = self.grok_client.generate_explanation(
                            match_context, predicted_dict, summary_stats
                        )
                        grok_time = time.time() - grok_start
                    except Exception as e:
                        logger.error(f"Grok açıklama hatası: {e}")
                        grok_explanation = "Grok açıklama oluşturulamadı"
                        grok_time = 999
                    
                    # En hızlı olanı seç
                    if gpt_time < grok_time:
                        best_model = "gpt"
                        explanation = gpt_explanation
                        best_time = gpt_time
                    else:
                        best_model = "grok"
                        explanation = grok_explanation
                        best_time = grok_time
                    
                    explanations[market_type] = {
                        "gpt_explanation": gpt_explanation,
                        "grok_explanation": grok_explanation,
                        "gpt_time": round(gpt_time, 3),
                        "grok_time": round(grok_time, 3),
                        "best_model": best_model,
                        "explanation": explanation,
                        "best_time": round(best_time, 3)
                    }
            else:
                # Sadece varsayılan modeli kullan
                explanation_text = self.llm_client.generate_explanation(
                    match_context, predicted_dict, summary_stats
                )
                for market_type in market_types:
                    explanations[market_type] = {
                        "explanation": explanation_text,
                        "best_model": "gpt",
                        "best_time": 0
                    }
            
            return explanations
        finally:
            session.close()
    
    def save_explanations(
        self,
        match: Match,
        explanations: Dict[MarketType, Dict[str, Any]],
        summary_stats: Dict[str, Any]
    ) -> None:
        session = get_session()
        try:
            for market_type, explanation_data in explanations.items():
                market = MarketRepository.get_or_create(
                    session, name=market_type.value
                )
                
                # Açıklama metnini oluştur (model bilgisi ile)
                explanation_text = explanation_data.get("explanation", "")
                best_model = explanation_data.get("best_model", "gpt")
                best_time = explanation_data.get("best_time", 0)
                
                # Model bilgisini de ekle
                full_explanation = f"[{best_model.upper()} - {best_time}s] {explanation_text}"
                
                # Özet istatistiklere model bilgilerini ekle
                enhanced_stats = summary_stats.copy()
                enhanced_stats.update({
                    "gpt_explanation": explanation_data.get("gpt_explanation", ""),
                    "grok_explanation": explanation_data.get("grok_explanation", ""),
                    "gpt_time": explanation_data.get("gpt_time", 0),
                    "grok_time": explanation_data.get("grok_time", 0),
                    "best_model": best_model,
                    "best_time": best_time
                })
                
                ExplanationRepository.create(
                    session,
                    match_id=match.id,
                    market_id=market.id,
                    llm_output=full_explanation,
                    summary_stats=enhanced_stats
                )
            
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving explanations: {e}")
            raise
        finally:
            session.close()

