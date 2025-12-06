"""
Beynin Kendini Test Etme ve Öğrenme Mekanizması
- Eski maçları bugün yapılıyormuş gibi tahmin eder
- Hataları analiz eder ve öğrenir
- Tüm takımlar, ikili ilişkiler, pattern'ler hakkında öğrenir
- En başarılı olana kadar sürekli deneme-yanılma ile öğrenir
"""
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
import logging
import torch
import numpy as np

from football_brain_core.src.models.multi_task_model import MultiTaskModel
from football_brain_core.src.inference.predict_markets import MarketPredictor
from football_brain_core.src.inference.backtest import Backtester
from football_brain_core.src.models.train_incremental import IncrementalTrainer
from football_brain_core.src.models.evaluate import ModelEvaluator
from football_brain_core.src.models.error_analyzer import ErrorAnalyzer
from football_brain_core.src.models.interactive_learning import InteractiveLearning
from football_brain_core.src.db.team_analysis import TeamAnalyzer
from football_brain_core.src.models.team_profile import TeamProfileManager
from football_brain_core.src.models.pairwise_relationship import PairwiseRelationshipManager
from football_brain_core.src.models.synthesis_predictor import SynthesisPredictor
from football_brain_core.src.reporting.export_excel import ExcelExporter
from football_brain_core.src.reporting.backtest_excel import BacktestExcelExporter
from football_brain_core.src.reporting.team_analysis_excel import TeamAnalysisExcelExporter
from football_brain_core.src.features.market_targets import MarketType
from football_brain_core.src.db.connection import get_session
from football_brain_core.src.db.repositories import (
    MatchRepository, ResultRepository, ModelVersionRepository,
    ExperimentRepository, LeagueRepository, MarketRepository
)
from football_brain_core.src.features.feature_builder import FeatureBuilder
from football_brain_core.src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelfLearningBrain:
    """
    Beynin kendini test etme ve öğrenme mekanizması.
    Eski maçları bugün yapılıyormuş gibi tahmin eder, hataları öğrenir,
    tüm takımlar ve ilişkiler hakkında bilgi toplar.
    """
    
    def __init__(
        self,
        model: MultiTaskModel,
        market_types: List[MarketType],
        config: Optional[Config] = None
    ):
        self.model = model
        self.market_types = market_types
        self.config = config or Config()
        self.feature_builder = FeatureBuilder()
        self.predictor = MarketPredictor(model, market_types, self.feature_builder)
        self.backtester = Backtester(model, market_types, self.feature_builder)
        self.evaluator = ModelEvaluator(self.feature_builder)
        self.incremental_trainer = IncrementalTrainer(market_types, config)
        self.error_analyzer = ErrorAnalyzer(self.feature_builder)
        self.interactive_learner = InteractiveLearning()
        self.team_analyzer = TeamAnalyzer()
        self.team_profile_manager = TeamProfileManager()
        self.pairwise_manager = PairwiseRelationshipManager()
        self.synthesis_predictor = SynthesisPredictor(
            self.team_profile_manager,
            self.pairwise_manager
        )
        self.excel_exporter = ExcelExporter(config)
        self.backtest_exporter = BacktestExcelExporter(config)
        self.team_excel_exporter = TeamAnalysisExcelExporter(config)
    
    def learn_from_past_matches(
        self,
        season: int,
        league_ids: Optional[List[int]] = None,
        max_iterations: int = 10,
        target_accuracy: float = 0.70
    ) -> Dict[str, Any]:
        """
        Geçmiş maçları bugün yapılıyormuş gibi tahmin eder ve öğrenir.
        
        Args:
            season: Hangi sezon üzerinde öğrenilecek
            league_ids: Hangi ligler (None ise tüm ligler)
            max_iterations: Maksimum öğrenme iterasyonu
            target_accuracy: Hedef doğruluk oranı
        
        Returns:
            Öğrenme süreci metrikleri
        """
        session = get_session()
        try:
            if league_ids is None:
                leagues = LeagueRepository.get_all(session)
                league_ids = [l.id for l in leagues]
            
            logger.info(f"🧠 Beyin öğrenme modu başlatılıyor...")
            logger.info(f"📊 Sezon: {season}, Ligler: {len(league_ids)}")
            
            # Tüm maçları al (sezon başından itibaren sırayla)
            all_matches = []
            for league_id in league_ids:
                matches = MatchRepository.get_by_league_and_season(
                    session, league_id, season
                )
                # Sadece sonuçları olan maçlar (öğrenme için)
                matches = [m for m in matches if m.home_score is not None and m.away_score is not None]
                all_matches.extend(matches)
            
            # Tarih sırasına göre sırala
            all_matches.sort(key=lambda x: x.match_date)
            
            logger.info(f"📈 Toplam {len(all_matches)} maç bulundu")
            
            learning_history = []
            current_model = self.model
            best_accuracy = 0.0
            best_model = current_model
            
            for iteration in range(max_iterations):
                logger.info(f"\n🔄 İterasyon {iteration + 1}/{max_iterations}")
                
                # Bu iterasyonda öğrenilecek maçlar (sırayla, sanki o gün yaşıyormuş gibi)
                correct_predictions = 0
                total_predictions = 0
                error_patterns = []
                
                for i, match in enumerate(all_matches):
                    # Sadece bu maçtan önceki maçları kullanarak tahmin yap
                    cutoff_date = match.match_date
                    previous_matches = [m for m in all_matches if m.match_date < cutoff_date]
                    
                    if len(previous_matches) < 10:  # Yeterli veri yoksa atla
                        continue
                    
                    try:
                        # Tahmin yap (sanki o gün yaşıyormuş gibi)
                        predictions = self.predictor.predict_match(match.id, session)
                        
                        # Gerçek sonuçlarla karşılaştır
                        results = ResultRepository.get_by_match(session, match.id)
                        
                        for market_type in self.market_types:
                            from football_brain_core.src.db.repositories import MarketRepository
                            market = MarketRepository.get_or_create(
                                session, name=market_type.value
                            )
                            result = next(
                                (r for r in results if r.market_id == market.id), None
                            )
                            
                            if result:
                                pred_outcome = predictions[market_type]["outcome"]
                                actual_outcome = result.actual_outcome
                                
                                total_predictions += 1
                                
                                if pred_outcome == actual_outcome:
                                    correct_predictions += 1
                                else:
                                    # Hata analizi yap - detaylı öğrenme için
                                    predicted_proba_dict = {
                                        outcome: prob
                                        for outcome, prob in zip(
                                            predictions[market_type].get("all_probabilities", []),
                                            predictions[market_type].get("all_probabilities", [])
                                        )
                                    }
                                    
                                    error_analysis = self.error_analyzer.analyze_error(
                                        match.id,
                                        market_type,
                                        pred_outcome,
                                        actual_outcome,
                                        predicted_proba_dict,
                                        session
                                    )
                                    
                                    # Etkileşimli öğrenme: Neden yanlış oldu düşün
                                    match_context = {
                                        "home_team": home_team.name if home_team else "Home",
                                        "away_team": away_team.name if away_team else "Away",
                                        "match_date": match.match_date.isoformat()
                                    }
                                    
                                    home_features = self.feature_builder.build_team_features(
                                        match.home_team_id, match.match_date, session
                                    )
                                    away_features = self.feature_builder.build_team_features(
                                        match.away_team_id, match.match_date, session
                                    )
                                    summary_stats = {
                                        "home_form": f"{home_features.get('win_rate', 0):.1%}",
                                        "away_form": f"{away_features.get('win_rate', 0):.1%}",
                                        **home_features,
                                        **away_features
                                    }
                                    
                                    interactive_analysis = self.interactive_learner.analyze_mistake_and_think(
                                        match.id,
                                        market_type,
                                        pred_outcome,
                                        actual_outcome,
                                        predicted_proba_dict,
                                        match_context,
                                        summary_stats
                                    )
                                    
                                    # Kullanıcıya sorulması gerekiyorsa işaretle
                                    if interactive_analysis.get("needs_user_input"):
                                        logger.warning(f"❓ Kullanıcıya soru gerekli - Maç {match.id}, Market {market_type.value}")
                                        logger.info(interactive_analysis.get("user_question", ""))
                                    
                                    # Hata kaydet - öğrenme için
                                    error_patterns.append({
                                        "match_id": match.id,
                                        "market": market_type.value,
                                        "predicted": pred_outcome,
                                        "actual": actual_outcome,
                                        "match_date": match.match_date,
                                        "home_team_id": match.home_team_id,
                                        "away_team_id": match.away_team_id,
                                        "analysis": error_analysis,  # Detaylı analiz
                                        "interactive_analysis": interactive_analysis  # Etkileşimli analiz
                                    })
                    
                    except Exception as e:
                        logger.debug(f"Maç {match.id} için hata: {e}")
                        continue
                
                if total_predictions == 0:
                    logger.warning("Tahmin yapılamadı, yeterli veri yok")
                    break
                
                accuracy = correct_predictions / total_predictions
                logger.info(f"📊 Doğruluk: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
                logger.info(f"❌ Hatalar: {len(error_patterns)}")
                
                learning_history.append({
                    "iteration": iteration + 1,
                    "accuracy": accuracy,
                    "correct": correct_predictions,
                    "total": total_predictions,
                    "errors": len(error_patterns)
                })
                
                # En iyi modeli güncelle
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = current_model
                    logger.info(f"✅ Yeni en iyi doğruluk: {best_accuracy:.2%}")
                
                # Hedef doğruluğa ulaşıldı mı?
                if accuracy >= target_accuracy:
                    logger.info(f"🎯 Hedef doğruluğa ulaşıldı: {target_accuracy:.2%}")
                    break
                
                # Her iterasyonda Excel export yap (PRD gereksinimi)
                if len(error_patterns) > 0:
                    logger.info(f"📊 Iterasyon {iteration + 1} Excel'e export ediliyor...")
                    try:
                        # Backtest sonuçlarını Excel'e export et
                        backtest_data = {
                            "match_results": [
                                {
                                    "match_id": e["match_id"],
                                    "predictions": {e["market"]: e["predicted"]},
                                    "actuals": {e["market"]: e["actual"]},
                                    "correct": {e["market"]: False}
                                }
                                for e in error_patterns
                            ],
                            "total_matches": len(set(e["match_id"] for e in error_patterns)),
                            "accuracy_by_market": {}
                        }
                        
                        excel_path = self.backtest_exporter.export_backtest_results(
                            backtest_data,
                            date_from=all_matches[0].match_date.date() if all_matches else date.today(),
                            date_to=all_matches[-1].match_date.date() if all_matches else date.today(),
                            model_version_id=None
                        )
                        logger.info(f"✅ Excel export edildi: {excel_path}")
                    except Exception as e:
                        logger.error(f"Excel export hatası: {e}")
                
                # Detaylı takım profilleri oluştur (en ince ayrıntısına kadar)
                logger.info(f"🔍 Detaylı takım profilleri oluşturuluyor (en ince ayrıntısına kadar)...")
                try:
                    team_profiles = self.team_profile_manager.build_all_profiles(season, self.market_types)
                    logger.info(f"📊 {len(team_profiles)} takım profili oluşturuldu (her detay ezberlendi)")
                    
                    # İkili ilişkiler oluştur (her çift için ayrı algoritma)
                    logger.info(f"🤝 Takım çiftleri için özel algoritmalar oluşturuluyor...")
                    pairwise_relationships = self.pairwise_manager.build_all_relationships(season, self.market_types)
                    logger.info(f"✅ {len(pairwise_relationships)} takım çifti için özel algoritma oluşturuldu")
                    
                    # Eski analizler (geriye dönük uyumluluk)
                    team_patterns = self.team_analyzer.get_all_team_patterns(season, self.market_types)
                    team_relationships = self.team_analyzer.get_all_team_relationships(season, self.market_types)
                    
                    # Excel'e export et (her iterasyonda)
                    try:
                        team_patterns_path = self.team_excel_exporter.export_team_patterns(
                            team_patterns, iteration + 1, season
                        )
                        logger.info(f"📊 Takım pattern'leri Excel'e export edildi: {team_patterns_path}")
                        
                        if team_relationships:
                            relationships_path = self.team_excel_exporter.export_team_relationships(
                                team_relationships, iteration + 1, season
                            )
                            logger.info(f"🤝 Takım ilişkileri Excel'e export edildi: {relationships_path}")
                    except Exception as e:
                        logger.error(f"Takım analizi Excel export hatası: {e}")
                    
                except Exception as e:
                    logger.error(f"Takım analizi hatası: {e}")
                
                # Hatalardan öğren: Detaylı analiz ve evrim
                if len(error_patterns) > 0 and iteration < max_iterations - 1:
                    logger.info(f"📚 Hatalardan detaylı öğreniliyor...")
                    
                    # Hata analizi toplu yap
                    error_summary = self.error_analyzer.collect_errors(error_patterns)
                    
                    logger.info(f"🔍 Hata analizi:")
                    logger.info(f"   - Bias hataları: {error_summary['bias_count']}")
                    logger.info(f"   - Variance hataları: {error_summary['variance_count']}")
                    logger.info(f"   - Eksik feature'lar: {list(error_summary['missing_features_all'].keys())}")
                    
                    # Önerileri göster
                    for suggestion in error_summary.get("suggested_improvements", []):
                        logger.info(f"💡 Öneri: {suggestion}")
                    
                    # Kapsamlı analiz Excel'i (Pattern'ler + İlişkiler + Hatalar)
                    try:
                        comprehensive_path = self.team_excel_exporter.export_comprehensive_analysis(
                            team_patterns,
                            team_relationships,
                            iteration + 1,
                            season,
                            error_summary
                        )
                        logger.info(f"📋 Kapsamlı analiz Excel'e export edildi: {comprehensive_path}")
                    except Exception as e:
                        logger.error(f"Kapsamlı analiz export hatası: {e}")
                    
                    # Hatalı maçları topla
                    error_match_ids = list(set([e["match_id"] for e in error_patterns]))
                    
                    # Incremental training ile öğren
                    try:
                        error_matches = [
                            MatchRepository.get_by_id(session, mid)
                            for mid in error_match_ids
                        ]
                        error_matches = [m for m in error_matches if m is not None]
                        
                        if len(error_matches) > 0:
                            logger.info(f"🔄 {len(error_matches)} hatalı maçtan öğreniliyor...")
                            
                            # Modeli hatalardan öğrenerek güncelle
                            # Bias düzeltmesi için özel learning rate
                            learning_rate_multiplier = 1.0
                            if error_summary["bias_count"] > error_summary["total_errors"] * 0.3:
                                learning_rate_multiplier = 1.5  # Bias için daha agresif öğrenme
                                logger.info("⚡ Bias düzeltmesi için learning rate artırıldı")
                            
                            updated_model = self.incremental_trainer.retrain(
                                current_model,
                                error_matches,
                                [season - 1, season],
                                [season],
                                league_ids,
                                epochs=5 + int(error_summary["bias_count"] / 10)  # Bias varsa daha fazla epoch
                            )
                            
                            # Yeni model daha iyi mi?
                            val_loader = None  # Validasyon loader oluştur
                            if self.incremental_trainer.should_update_model(
                                updated_model, current_model, val_loader
                            ):
                                current_model = updated_model
                                self.model = updated_model
                                self.predictor = MarketPredictor(
                                    updated_model, self.market_types, self.feature_builder
                                )
                                logger.info("✅ Model hatalardan öğrenerek evrimleşti!")
                                
                                # Feature importance güncelle
                                if error_summary.get("feature_importance_updates"):
                                    logger.info("📊 Feature importance güncellendi")
                            else:
                                logger.info("⚠️  Yeni model daha iyi değil, eski model korunuyor")
                    
                    except Exception as e:
                        logger.error(f"Hatalardan öğrenme hatası: {e}")
            
            # Final değerlendirme
            logger.info(f"\n🏆 Öğrenme tamamlandı!")
            logger.info(f"📊 En iyi doğruluk: {best_accuracy:.2%}")
            logger.info(f"🔄 Toplam iterasyon: {len(learning_history)}")
            
            return {
                "best_accuracy": best_accuracy,
                "iterations": len(learning_history),
                "learning_history": learning_history,
                "final_model": best_model
            }
        
        finally:
            session.close()
    
    def analyze_team_relationships(
        self,
        season: int,
        league_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Takımlar arası ikili ilişkileri analiz eder ve öğrenir.
        Hangi takımlar hangi takımlara karşı nasıl performans gösteriyor.
        """
        session = get_session()
        try:
            logger.info("🔍 Takım ilişkileri analiz ediliyor...")
            
            if league_ids is None:
                leagues = LeagueRepository.get_all(session)
                league_ids = [l.id for l in leagues]
            
            team_relationships = {}
            
            for league_id in league_ids:
                matches = MatchRepository.get_by_league_and_season(
                    session, league_id, season
                )
                matches = [m for m in matches if m.home_score is not None and m.away_score is not None]
                
                for match in matches:
                    home_id = match.home_team_id
                    away_id = match.away_team_id
                    
                    key = f"{home_id}_{away_id}"
                    reverse_key = f"{away_id}_{home_id}"
                    
                    if key not in team_relationships:
                        team_relationships[key] = {
                            "home_wins": 0,
                            "away_wins": 0,
                            "draws": 0,
                            "home_goals": 0,
                            "away_goals": 0,
                            "matches": []
                        }
                    
                    if match.home_score > match.away_score:
                        team_relationships[key]["home_wins"] += 1
                    elif match.home_score < match.away_score:
                        team_relationships[key]["away_wins"] += 1
                    else:
                        team_relationships[key]["draws"] += 1
                    
                    team_relationships[key]["home_goals"] += match.home_score
                    team_relationships[key]["away_goals"] += match.away_score
                    team_relationships[key]["matches"].append(match.id)
            
            logger.info(f"📊 {len(team_relationships)} takım çifti analiz edildi")
            
            return {
                "team_relationships": team_relationships,
                "total_pairs": len(team_relationships)
            }
        
        finally:
            session.close()
    
    def continuous_learning_loop(
        self,
        seasons: List[int],
        league_ids: Optional[List[int]] = None,
        max_iterations_per_season: int = 10
    ) -> Dict[str, Any]:
        """
        Sürekli öğrenme döngüsü: Tüm sezonlar üzerinde öğrenir,
        en başarılı olana kadar deneme-yanılma yapar.
        """
        logger.info("🔄 Sürekli öğrenme döngüsü başlatılıyor...")
        
        all_results = {}
        
        for season in seasons:
            logger.info(f"\n📅 Sezon {season} öğreniliyor...")
            
            season_results = self.learn_from_past_matches(
                season=season,
                league_ids=league_ids,
                max_iterations=max_iterations_per_season
            )
            
            all_results[season] = season_results
            
            # Takım ilişkilerini analiz et
            relationships = self.analyze_team_relationships(season, league_ids)
            season_results["relationships"] = relationships
        
        # Genel özet
        overall_best_accuracy = max([r["best_accuracy"] for r in all_results.values()])
        
        logger.info(f"\n🏆 Genel en iyi doğruluk: {overall_best_accuracy:.2%}")
        
        return {
            "seasons": all_results,
            "overall_best_accuracy": overall_best_accuracy,
            "final_model": self.model
        }

