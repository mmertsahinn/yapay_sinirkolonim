"""
Etkileşimli Öğrenme ve Evrim Mekanizması
- Model yanlış tahmin yaptığında nedenini düşünür
- LLM ile yorum yapar
- Mantıklı sebep bulamazsa kullanıcıya sorar
- Kullanıcıyla karşılıklı fikir alışverişi yapar
- Sürekli birlikte evrilir
"""
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime

from football_brain_core.src.explanations.llm_client import LLMClient
from football_brain_core.src.models.error_analyzer import ErrorAnalyzer
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    MatchRepository, TeamRepository, ResultRepository, MarketRepository
)
from football_brain_core.src.db.schema import Explanation
from football_brain_core.src.features.market_targets import MarketType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveLearning:
    """
    Model ile kullanıcı arasında etkileşimli öğrenme mekanizması.
    Hataları analiz eder, LLM ile yorum yapar, gerekirse kullanıcıya sorar.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.error_analyzer = ErrorAnalyzer(None)
        self.learning_memory = []  # Kullanıcıdan öğrenilenler
    
    def analyze_mistake_and_think(
        self,
        match_id: int,
        market_type: MarketType,
        predicted: str,
        actual: str,
        predicted_proba: Dict[str, float],
        match_context: Dict[str, Any],
        summary_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Yanlış tahminin nedenini düşünür ve yorum yapar.
        Mantıklı sebep bulamazsa kullanıcıya sorar.
        """
        session = get_session()
        try:
            match = MatchRepository.get_by_id(session, match_id)
            if not match:
                return {"error": "Match not found"}
            
            home_team = TeamRepository.get_by_id(session, match.home_team_id)
            away_team = TeamRepository.get_by_id(session, match.away_team_id)
            
            # 1. Hata analizi yap
            error_analysis = self.error_analyzer.analyze_error(
                match_id, market_type, predicted, actual, predicted_proba, session
            )
            
            # 2. LLM'e "neden yanlış oldu" diye sor
            llm_reasoning = self._ask_llm_why_wrong(
                match_context,
                predicted,
                actual,
                summary_stats,
                error_analysis
            )
            
            # 3. LLM'in cevabı mantıklı mı kontrol et
            is_reasonable = self._evaluate_reasoning_quality(llm_reasoning, error_analysis)
            
            result = {
                "error_analysis": error_analysis,
                "llm_reasoning": llm_reasoning,
                "is_reasonable": is_reasonable,
                "needs_user_input": not is_reasonable,
                "learning_points": []
            }
            
            # 4. Mantıklı değilse kullanıcıya sor
            if not is_reasonable:
                result["user_question"] = self._generate_user_question(
                    match_context, predicted, actual, error_analysis
                )
                result["learning_points"] = ["Kullanıcıdan öğrenme gerekli"]
            
            # 5. Öğrenme noktalarını çıkar
            if is_reasonable:
                result["learning_points"] = self._extract_learning_points(
                    llm_reasoning, error_analysis
                )
            
            return result
        
        finally:
            session.close()
    
    def _ask_llm_why_wrong(
        self,
        match_context: Dict[str, Any],
        predicted: str,
        actual: str,
        summary_stats: Dict[str, Any],
        error_analysis: Dict[str, Any]
    ) -> str:
        """LLM'e neden yanlış olduğunu sorar"""
        
        prompt = f"""Bir futbol tahmin modeli yanlış tahmin yaptı. Neden yanlış olduğunu analiz et.

Maç: {match_context.get('home_team')} vs {match_context.get('away_team')}
Tahmin Edilen: {predicted}
Gerçek Sonuç: {actual}

İstatistikler:
{json.dumps(summary_stats, indent=2)}

Hata Analizi:
- Hata Kategorisi: {error_analysis.get('error_category', 'N/A')}
- Güven: {error_analysis.get('confidence', 0):.2%}
- Sapma Payı: {error_analysis.get('deviation', 0):.2%}
- Bias Tespit Edildi: {error_analysis.get('bias_detected', False)}
- Variance Problemi: {error_analysis.get('variance_issue', False)}

Eksik Feature'lar: {', '.join(error_analysis.get('missing_features', []))}

Lütfen şunları analiz et:
1. Model neden bu tahmini yaptı? (Hangi pattern'e dayandı?)
2. Gerçek sonuç neden farklı oldu? (Hangi faktör göz ardı edildi?)
3. Modelin formülünde/feature'larında ne eksik veya yanlış?
4. Bu hatadan nasıl öğrenilebilir?

Kısa ve net bir analiz yap (3-4 cümle). Eğer kesin bir sebep bulamıyorsan "BELIRSIZ" yaz."""

        try:
            response = self.llm_client.generate_explanation(
                match_context,
                {"predicted": predicted, "actual": actual},
                {**summary_stats, **error_analysis}
            )
            return response
        except Exception as e:
            logger.error(f"LLM reasoning hatası: {e}")
            return "LLM analizi yapılamadı"
    
    def _evaluate_reasoning_quality(
        self,
        llm_reasoning: str,
        error_analysis: Dict[str, Any]
    ) -> bool:
        """LLM'in cevabının mantıklı olup olmadığını değerlendirir"""
        
        # Belirsizlik kontrolü
        if "BELIRSIZ" in llm_reasoning.upper() or "bilmiyorum" in llm_reasoning.lower():
            return False
        
        # Çok kısa cevaplar mantıksız olabilir
        if len(llm_reasoning.split()) < 10:
            return False
        
        # Hata analizi ile uyumlu mu?
        if error_analysis.get("bias_detected") and "bias" not in llm_reasoning.lower():
            # Bias tespit edilmiş ama LLM bahsetmemiş - şüpheli
            if len(llm_reasoning) < 100:
                return False
        
        # Eksik feature'lar bahsedilmiş mi?
        missing_features = error_analysis.get("missing_features", [])
        if missing_features:
            mentioned = any(feat.lower() in llm_reasoning.lower() for feat in missing_features)
            if not mentioned and len(llm_reasoning) < 150:
                return False
        
        return True
    
    def _generate_user_question(
        self,
        match_context: Dict[str, Any],
        predicted: str,
        actual: str,
        error_analysis: Dict[str, Any]
    ) -> str:
        """Kullanıcıya sorulacak soruyu oluşturur"""
        
        home_team = match_context.get("home_team", "Home Team")
        away_team = match_context.get("away_team", "Away Team")
        
        question = f"""
🤔 Model yanlış tahmin yaptı ve nedenini tam olarak anlayamadı. Yardım eder misin?

Maç: {home_team} vs {away_team}
Tahmin: {predicted}
Gerçek: {actual}

Hata Analizi:
- Kategori: {error_analysis.get('error_category', 'N/A')}
- Güven: {error_analysis.get('confidence', 0):.2%}
"""
        
        if error_analysis.get("bias_detected"):
            question += "- ⚠️ Bias problemi tespit edildi\n"
        
        if error_analysis.get("missing_features"):
            question += f"- 📋 Eksik feature'lar: {', '.join(error_analysis['missing_features'])}\n"
        
        question += f"""
Soru: Model neden yanlış tahmin yaptı? Hangi faktörü göz ardı etti veya yanlış yorumladı?

Örnek cevaplar:
- "Ev sahibi takımın son 3 maçta formu çok kötüydü ama model bunu yeterince dikkate almadı"
- "Bu iki takım arasında özel bir rekabet var, model bunu bilmiyor"
- "Hava koşulları/seyirci faktörü önemliydi"
- "Takım kadrosunda önemli bir değişiklik vardı"
- "Modelin formülünde X eksik"

Cevabın: """
        
        return question
    
    def _extract_learning_points(
        self,
        llm_reasoning: str,
        error_analysis: Dict[str, Any]
    ) -> List[str]:
        """LLM'in analizinden öğrenme noktalarını çıkarır"""
        points = []
        
        # Bias tespit edilmişse
        if error_analysis.get("bias_detected"):
            points.append("Bias düzeltmesi gerekli")
        
        # Eksik feature'lar
        missing = error_analysis.get("missing_features", [])
        if missing:
            points.append(f"Eklenecek feature'lar: {', '.join(missing)}")
        
        # LLM'in önerileri
        if "formül" in llm_reasoning.lower() or "formula" in llm_reasoning.lower():
            points.append("Model formülü güncellenmeli")
        
        if "feature" in llm_reasoning.lower():
            points.append("Feature engineering gerekli")
        
        return points
    
    def process_user_feedback(
        self,
        match_id: int,
        market_type: MarketType,
        user_feedback: str,
        error_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Kullanıcıdan gelen geri bildirimi işler ve öğrenir.
        """
        learning_entry = {
            "match_id": match_id,
            "market_type": market_type.value,
            "user_feedback": user_feedback,
            "error_analysis": error_analysis,
            "timestamp": datetime.now().isoformat(),
            "learned": False
        }
        
        # Geri bildirimden öğrenme noktalarını çıkar
        learning_points = self._extract_learning_from_feedback(user_feedback, error_analysis)
        learning_entry["learning_points"] = learning_points
        
        # Öğrenme hafızasına ekle
        self.learning_memory.append(learning_entry)
        
        logger.info(f"📚 Kullanıcı geri bildirimi kaydedildi: {user_feedback[:50]}...")
        logger.info(f"💡 Öğrenme noktaları: {learning_points}")
        
        return {
            "saved": True,
            "learning_points": learning_points,
            "memory_size": len(self.learning_memory)
        }
    
    def _extract_learning_from_feedback(
        self,
        user_feedback: str,
        error_analysis: Dict[str, Any]
    ) -> List[str]:
        """Kullanıcı geri bildiriminden öğrenme noktalarını çıkarır"""
        points = []
        feedback_lower = user_feedback.lower()
        
        # Feature eksikliği bahsedilmiş mi?
        if "eksik" in feedback_lower or "missing" in feedback_lower:
            if "feature" in feedback_lower or "özellik" in feedback_lower:
                points.append("Yeni feature eklenmeli")
        
        # Formül hatası bahsedilmiş mi?
        if "formül" in feedback_lower or "formula" in feedback_lower:
            points.append("Model formülü güncellenmeli")
        
        # Takım özel durumu
        if "özel" in feedback_lower or "special" in feedback_lower or "rekabet" in feedback_lower:
            points.append("Takım özel durumları feature'a eklenmeli")
        
        # Form/trend
        if "form" in feedback_lower or "trend" in feedback_lower:
            points.append("Form trendi feature'ı güçlendirilmeli")
        
        # Ev sahibi avantajı
        if "ev sahibi" in feedback_lower or "home" in feedback_lower:
            points.append("Ev sahibi avantajı feature'ı güncellenmeli")
        
        return points
    
    def apply_learned_knowledge(
        self,
        match_context: Dict[str, Any],
        current_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Öğrenilen bilgileri mevcut tahminlere uygular.
        """
        enhanced_features = current_features.copy()
        
        # Öğrenme hafızasından ilgili bilgileri bul
        home_team = match_context.get("home_team", "")
        away_team = match_context.get("away_team", "")
        
        relevant_learnings = [
            entry for entry in self.learning_memory
            if (home_team in str(entry.get("error_analysis", {})) or
                away_team in str(entry.get("error_analysis", {})))
        ]
        
        if relevant_learnings:
            logger.info(f"🧠 {len(relevant_learnings)} ilgili öğrenme bulundu")
            
            # Öğrenilen pattern'leri feature'lara ekle
            for learning in relevant_learnings[-5:]:  # Son 5 öğrenme
                feedback = learning.get("user_feedback", "")
                if "form" in feedback.lower():
                    enhanced_features["learned_form_adjustment"] = 1.1
                if "özel" in feedback.lower() or "special" in feedback.lower():
                    enhanced_features["learned_special_case"] = True
        
        return enhanced_features
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Öğrenme özetini döndürür"""
        return {
            "total_learnings": len(self.learning_memory),
            "recent_learnings": self.learning_memory[-10:] if len(self.learning_memory) > 10 else self.learning_memory,
            "learning_topics": list(set([
                point
                for entry in self.learning_memory
                for point in entry.get("learning_points", [])
            ]))
        }







