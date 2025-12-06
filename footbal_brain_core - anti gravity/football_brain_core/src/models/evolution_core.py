"""
PRD: Evolution Core - Hata Analizi ve Evrim Döngüsü
- Error Inbox: Hataları toplar
- Hata Cluster'ları: Benzer hataları gruplar
- Üç Seviyeli Çözüm: Level 1 (içsel), Level 2 (veri zenginleştirme), Level 3 (unresolved)
- Kullanıcıya Soru Sorma: Unresolved cluster'lar için
"""
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN

from src.db.connection import get_session
from src.db.repositories import (
    ErrorCaseRepository, ErrorClusterRepository,
    HumanFeedbackRepository, EvolutionPlanRepository,
    MatchRepository, ResultRepository, PredictionRepository,
    MarketRepository, LeagueRepository, ModelVersionRepository,
    ExplanationRepository
)
from src.db.schema import ErrorCase, ErrorCluster
from src.features.market_targets import MarketType
from src.features.feature_builder import FeatureBuilder
from src.explanations.llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvolutionCore:
    """
    PRD: Evolution Core
    Hataları analiz eder, cluster'lara ayırır, çözmeye çalışır, çözemediğinde kullanıcıya sorar.
    """
    
    def __init__(self, feature_builder: Optional[FeatureBuilder] = None):
        self.feature_builder = feature_builder or FeatureBuilder()
        self.llm_client = LLMClient()
    
    def collect_errors_to_inbox(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> int:
        """
        PRD: Error Inbox
        Maç sonuçları ile tahminleri karşılaştır, hataları error_cases tablosuna ekle
        """
        session = get_session()
        try:
            # Aktif model versiyonu
            active_model = ModelVersionRepository.get_active(session)
            if not active_model:
                logger.warning("Aktif model bulunamadı")
                return 0
            
            # Tarih aralığındaki maçları bul
            if date_from and date_to:
                matches = MatchRepository.get_by_date_range(session, date_from.date(), date_to.date())
            else:
                # Son 7 gün
                from datetime import timedelta
                date_to = datetime.now()
                date_from = date_to - timedelta(days=7)
                matches = MatchRepository.get_by_date_range(session, date_from.date(), date_to.date())
            
            # Sadece sonuçları olan maçlar
            matches = [m for m in matches if m.home_score is not None and m.away_score is not None]
            
            error_count = 0
            
            for match in matches:
                predictions = PredictionRepository.get_by_match(session, match.id)
                results = ResultRepository.get_by_match(session, match.id)
                
                for pred in predictions:
                    # Gerçek sonuç var mı?
                    result = next((r for r in results if r.market_id == pred.market_id), None)
                    if not result:
                        continue
                    
                    # Hata var mı?
                    if pred.predicted_outcome != result.actual_outcome:
                    # Zaten var mı kontrol et
                    from src.db.schema import ErrorCase
                    existing = session.query(ErrorCase).filter(
                        ErrorCase.match_id == match.id,
                        ErrorCase.market_id == pred.market_id,
                        ErrorCase.model_version_id == active_model.id
                    ).first()
                        
                        if not existing:
                            # LLM comment al (varsa)
                            from src.db.schema import Explanation
                            explanations = session.query(Explanation).filter(
                                Explanation.match_id == match.id,
                                Explanation.market_id == pred.market_id
                            ).first()
                            
                            llm_comment = explanations.llm_output if explanations else None
                            
                            # Error case oluştur
                            ErrorCaseRepository.create(
                                session,
                                match_id=match.id,
                                market_id=pred.market_id,
                                predicted_outcome=pred.predicted_outcome,
                                actual_outcome=result.actual_outcome,
                                model_version_id=active_model.id,
                                llm_comment=llm_comment
                            )
                            error_count += 1
            
            session.commit()
            logger.info(f"✅ {error_count} hata Error Inbox'a eklendi")
            return error_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error Inbox toplama hatası: {e}")
            return 0
        finally:
            session.close()
    
    def cluster_errors(self) -> Dict[str, Any]:
        """
        PRD: Hata Cluster'ları
        Benzer hataları feature vector'lerine göre gruplar
        """
        session = get_session()
        try:
            # Unresolved error cases
            error_cases = ErrorCaseRepository.get_unresolved(session)
            
            if len(error_cases) == 0:
                logger.info("Cluster'lanacak hata yok")
                return {"clusters_created": 0, "clusters": []}
            
            logger.info(f"📊 {len(error_cases)} hata cluster'lanıyor...")
            
            # Her error case için feature vector oluştur
            vectors = []
            error_data = []
            
            for error_case in error_cases:
                match = MatchRepository.get_by_id(session, error_case.match_id)
                if not match:
                    continue
                
                # Feature vector: [lig_id, market_id, predicted_outcome_type, actual_outcome_type, form_metrics...]
                vector = self._build_error_vector(match, error_case, session)
                if vector:
                    vectors.append(vector)
                    error_data.append(error_case)
            
            if len(vectors) < 2:
                logger.info("Yeterli hata yok, cluster'lama atlanıyor")
                return {"clusters_created": 0, "clusters": []}
            
            # DBSCAN ile cluster'lama
            vectors_array = np.array(vectors)
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(vectors_array)
            
            # Cluster'ları oluştur
            clusters_created = 0
            cluster_info = {}
            
            for label in set(clustering.labels_):
                if label == -1:  # Noise, cluster'a atama
                    continue
                
                cluster_errors = [error_data[i] for i, l in enumerate(clustering.labels_) if l == label]
                
                if len(cluster_errors) < 2:
                    continue
                
                # Cluster özeti oluştur
                cluster_summary = self._create_cluster_summary(cluster_errors, session)
                
                # Cluster oluştur
                cluster = ErrorClusterRepository.create(
                    session,
                    cluster_name=f"Cluster_{label}_{datetime.now().strftime('%Y%m%d')}",
                    error_summary=cluster_summary["summary"],
                    league_id=cluster_summary.get("league_id"),
                    market_id=cluster_summary.get("market_id"),
                    feature_vector=cluster_summary.get("feature_vector")
                )
                
                # Error cases'i cluster'a ata
                for error_case in cluster_errors:
                    error_case.error_cluster_id = cluster.id
                    error_case.resolution_level = "unresolved"
                
                cluster.example_count = len(cluster_errors)
                clusters_created += 1
                cluster_info[cluster.id] = {
                    "name": cluster.cluster_name,
                    "count": len(cluster_errors),
                    "summary": cluster_summary["summary"]
                }
            
            session.commit()
            logger.info(f"✅ {clusters_created} cluster oluşturuldu")
            
            return {
                "clusters_created": clusters_created,
                "clusters": cluster_info
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Cluster'lama hatası: {e}")
            import traceback
            traceback.print_exc()
            return {"clusters_created": 0, "clusters": []}
        finally:
            session.close()
    
    def _build_error_vector(
        self,
        match: "Match",
        error_case: "ErrorCase",
        session
    ) -> Optional[List[float]]:
        """Error case için feature vector oluştur"""
        try:
            # Basit vector: [league_id, market_id, predicted_type, actual_type, home_form, away_form]
            league_id = match.league_id
            market_id = error_case.market_id
            
            # Form metrikleri
            home_features = self.feature_builder.build_team_features(
                match.home_team_id, match.match_date, session
            )
            away_features = self.feature_builder.build_team_features(
                match.away_team_id, match.match_date, session
            )
            
            # Outcome type encoding (basit)
            pred_type = hash(error_case.predicted_outcome) % 10
            actual_type = hash(error_case.actual_outcome) % 10
            
            vector = [
                float(league_id),
                float(market_id),
                float(pred_type),
                float(actual_type),
                home_features.get("win_rate", 0.0),
                away_features.get("win_rate", 0.0),
                home_features.get("avg_goals_scored", 0.0),
                away_features.get("avg_goals_scored", 0.0),
            ]
            
            return vector
        except:
            return None
    
    def _create_cluster_summary(
        self,
        error_cases: List["ErrorCase"],
        session
    ) -> Dict[str, Any]:
        """Cluster için özet oluştur"""
        if not error_cases:
            return {"summary": "Empty cluster"}
        
        # En çok görülen lig ve market
        league_counts = defaultdict(int)
        market_counts = defaultdict(int)
        
        for error_case in error_cases:
            match = MatchRepository.get_by_id(session, error_case.match_id)
            if match:
                league_counts[match.league_id] += 1
            market_counts[error_case.market_id] += 1
        
        most_common_league = max(league_counts.items(), key=lambda x: x[1])[0] if league_counts else None
        most_common_market = max(market_counts.items(), key=lambda x: x[1])[0] if market_counts else None
        
        league = LeagueRepository.get_by_id(session, most_common_league) if most_common_league else None
        market = MarketRepository.get_by_id(session, most_common_market) if most_common_market else None
        
        summary = f"{league.name if league else 'Unknown'}, {market.name if market else 'Unknown'}, {len(error_cases)} hata"
        
        return {
            "summary": summary,
            "league_id": most_common_league,
            "market_id": most_common_market,
            "feature_vector": {"league": league.name if league else None, "market": market.name if market else None}
        }
    
    def solve_level1(self, cluster_id: int) -> Dict[str, Any]:
        """
        PRD: Seviye 1 - İçsel açıklama (mevcut veriden çözme)
        Cluster içindeki maçlar için mevcut feature'larla root-cause bulmaya çalışır
        """
        session = get_session()
        try:
            cluster = session.query(ErrorCluster).filter(
                ErrorCluster.id == cluster_id
            ).first()
            
            if not cluster:
                return {"solved": False, "reason": "Cluster bulunamadı"}
            
            error_cases = ErrorCaseRepository.get_by_cluster(session, cluster_id)
            
            if len(error_cases) < 2:
                return {"solved": False, "reason": "Yeterli örnek yok"}
            
            # Cluster'daki maçları analiz et
            match_ids = [ec.match_id for ec in error_cases]
            matches = [MatchRepository.get_by_id(session, mid) for mid in match_ids]
            matches = [m for m in matches if m]
            
            # Pattern analizi
            patterns = self._analyze_error_patterns(matches, error_cases, session)
            
            # LLM ile root-cause bul
            root_cause = self._find_root_cause_with_llm(cluster, error_cases, patterns, session)
            
            if root_cause and root_cause.get("confidence", 0) > 0.6:
                # Seviye 1'de çözüldü
                ErrorClusterRepository.update_resolution(
                    session, cluster_id, "level1", root_cause.get("explanation")
                )
                
                # Evolution plan oluştur
                EvolutionPlanRepository.create(
                    session,
                    cluster_id,
                    "calibration",
                    root_cause.get("explanation", ""),
                    root_cause.get("suggested_changes", {})
                )
                
                session.commit()
                logger.info(f"✅ Cluster {cluster_id} Seviye 1'de çözüldü")
                
                return {
                    "solved": True,
                    "level": "level1",
                    "root_cause": root_cause.get("explanation"),
                    "suggested_changes": root_cause.get("suggested_changes")
                }
            else:
                return {"solved": False, "reason": "Yeterli açıklama bulunamadı"}
                
        except Exception as e:
            session.rollback()
            logger.error(f"Seviye 1 çözüm hatası: {e}")
            return {"solved": False, "reason": str(e)}
        finally:
            session.close()
    
    def _analyze_error_patterns(
        self,
        matches,
        error_cases: List[ErrorCase],
        session
    ) -> Dict[str, Any]:
        """Hata pattern'lerini analiz et"""
        patterns = {
            "common_leagues": defaultdict(int),
            "common_markets": defaultdict(int),
            "form_trends": [],
            "score_patterns": []
        }
        
        for match in matches:
            patterns["common_leagues"][match.league_id] += 1
            
            home_features = self.feature_builder.build_team_features(
                match.home_team_id, match.match_date, session
            )
            away_features = self.feature_builder.build_team_features(
                match.away_team_id, match.match_date, session
            )
            
            patterns["form_trends"].append({
                "home_form": home_features.get("win_rate", 0),
                "away_form": away_features.get("win_rate", 0)
            })
        
        for error_case in error_cases:
            patterns["common_markets"][error_case.market_id] += 1
        
        return patterns
    
    def _find_root_cause_with_llm(
        self,
        cluster: ErrorCluster,
        error_cases: List[ErrorCase],
        patterns: Dict[str, Any],
        session
    ) -> Optional[Dict[str, Any]]:
        """LLM ile root-cause bul"""
        try:
            # Örnek error case'leri al
            sample_cases = error_cases[:5]
            
            prompt = f"""Bir futbol tahmin modelinde benzer hatalar kümesi var. Root-cause analizi yap.

Cluster: {cluster.error_summary}
Örnek sayısı: {len(error_cases)}

Pattern'ler:
- Ligler: {patterns.get('common_leagues', {})}
- Marketler: {patterns.get('common_markets', {})}
- Form trendleri: {patterns.get('form_trends', [])[:3]}

Lütfen şunları analiz et:
1. Bu hataların ortak nedeni ne olabilir?
2. Mevcut feature'larla açıklanabilir mi?
3. Hangi küçük kalibrasyonlar yapılabilir? (pencere boyutu, ağırlık, vb.)

Kısa ve net bir açıklama yap (2-3 cümle). Eğer kesin bir sebep bulamıyorsan "BELIRSIZ" yaz."""

            response = self.llm_client.generate_explanation(
                {"cluster": cluster.error_summary},
                {"patterns": str(patterns)},
                {}
            )
            
            if "BELIRSIZ" in response.upper():
                return None
            
            # Önerilen değişiklikleri çıkar
            suggested_changes = {}
            if "pencere" in response.lower() or "window" in response.lower():
                suggested_changes["window_size"] = "adjust"
            if "ağırlık" in response.lower() or "weight" in response.lower():
                suggested_changes["feature_weights"] = "adjust"
            
            return {
                "explanation": response,
                "confidence": 0.7 if len(response) > 50 else 0.5,
                "suggested_changes": suggested_changes
            }
            
        except Exception as e:
            logger.error(f"LLM root-cause hatası: {e}")
            return None
    
    def solve_level2(self, cluster_id: int) -> Dict[str, Any]:
        """
        PRD: Seviye 2 - Veri zenginleştirme (isteğe bağlı)
        API-FOOTBALL'dan ek veri çekip tekrar analiz
        """
        # Şimdilik placeholder - API entegrasyonu gerekli
        return {"solved": False, "reason": "Seviye 2 henüz implement edilmedi (API entegrasyonu gerekli)"}
    
    def ask_user_question(self, cluster_id: int) -> Dict[str, Any]:
        """
        PRD: Seviye 3 - Kullanıcıya soru sorma
        Unresolved cluster için soru üret
        """
        session = get_session()
        try:
            cluster = session.query(ErrorCluster).filter(
                ErrorCluster.id == cluster_id
            ).first()
            
            if not cluster:
                return {"question_created": False}
            
            # Soru oluştur
            question = self._generate_user_question(cluster, session)
            
            # HumanFeedback oluştur
            feedback = HumanFeedbackRepository.create(
                session,
                cluster_id,
                question
            )
            
            session.commit()
            logger.info(f"❓ Kullanıcıya soru soruldu: Cluster {cluster_id}")
            
            return {
                "question_created": True,
                "feedback_id": feedback.id,
                "question": question
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Soru oluşturma hatası: {e}")
            return {"question_created": False}
        finally:
            session.close()
    
    def _generate_user_question(
        self,
        cluster: ErrorCluster,
        session
    ) -> str:
        """Kullanıcıya sorulacak soruyu oluştur"""
        league = LeagueRepository.get_by_id(session, cluster.league_id) if cluster.league_id else None
        market = MarketRepository.get_by_id(session, cluster.market_id) if cluster.market_id else None
        
        question = f"""
🤔 Model bu tip hatalarda zorlanıyor ve nedenini tam olarak anlayamadı.

Cluster: {cluster.error_summary}
Lig: {league.name if league else 'Çeşitli'}
Market: {market.name if market else 'Çeşitli'}
Örnek sayısı: {cluster.example_count}

Soru: Bu lig/market kombinasyonunda modelin göz ardı ettiği önemli bir faktör var mı?

Örnek cevaplar:
- "Bu ligde tempo/hakem/oyun stili önemli, veri setinde bu tür bir feature yok"
- "Bu market için veri çok az, şimdilik analiz dışı bırak"
- "Bu takım tipi için özel bir pattern/model açılmalı"
- "Şimdilik dokunma, sadece izlemeye devam et"

Cevabın: """
        
        return question
    
    def process_evolution_cycle(self) -> Dict[str, Any]:
        """
        PRD: Tam evrim döngüsü
        1. Error Inbox'a hataları topla
        2. Cluster'la
        3. Her cluster için Seviye 1-2-3 dene
        """
        logger.info("🔄 Evrim döngüsü başlıyor...")
        
        # 1. Error Inbox
        error_count = self.collect_errors_to_inbox()
        logger.info(f"📥 {error_count} hata Error Inbox'a eklendi")
        
        # 2. Cluster'la
        cluster_result = self.cluster_errors()
        clusters_created = cluster_result.get("clusters_created", 0)
        logger.info(f"📊 {clusters_created} cluster oluşturuldu")
        
        # 3. Her cluster için çözüm dene
        solved_level1 = 0
        unresolved = 0
        questions_asked = 0
        
        for cluster_id in cluster_result.get("clusters", {}).keys():
            # Seviye 1 dene
            result = self.solve_level1(cluster_id)
            if result.get("solved"):
                solved_level1 += 1
            else:
                # Seviye 2 dene (şimdilik atla)
                # Seviye 3: Kullanıcıya sor
                question_result = self.ask_user_question(cluster_id)
                if question_result.get("question_created"):
                    questions_asked += 1
                else:
                    unresolved += 1
        
        return {
            "errors_collected": error_count,
            "clusters_created": clusters_created,
            "solved_level1": solved_level1,
            "questions_asked": questions_asked,
            "unresolved": unresolved
        }

