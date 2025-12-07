"""
🧬 EVOLUTION CORE - LoRA Hata Analizi ve Evrim Döngüsü
======================================================

football_brain_core'dan adapte edildi.

Özellikler:
- Error Inbox: LoRA hatalarını toplar
- Hata Cluster'ları: Benzer hataları gruplar (DBSCAN)
- 3 Seviyeli Çözüm:
  - Level 1: İçsel açıklama (LLM analizi)
  - Level 2: Veri zenginleştirme
  - Level 3: Kullanıcıya soru sorma
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


class LoRAEvolutionCore:
    """
    LoRA sistemine özel Evolution Core
    
    football_brain_core'dan adapte edildi.
    """
    
    def __init__(self):
        self.error_inbox = []  # Hataları toplar
        self.error_clusters = {}  # Cluster'lar
        self.evolution_plans = []  # Çözüm önerileri
        self.human_feedback = []  # Kullanıcı soruları
        
        print("🧬 LoRA Evolution Core başlatıldı")
    
    def collect_errors_to_inbox(
        self,
        lora_predictions: List[Dict],
        actual_results: Dict[int, str],
        match_idx: int
    ) -> int:
        """
        PRD: Error Inbox
        LoRA tahminleri ile gerçek sonuçları karşılaştır, hataları topla
        
        Args:
            lora_predictions: [(lora, prediction, confidence, proba), ...]
            actual_results: {match_id: actual_outcome}
            match_idx: Maç indexi
        
        Returns:
            Toplanan hata sayısı
        """
        error_count = 0
        
        for lora, prediction, confidence, proba in lora_predictions:
            # Gerçek sonuç var mı?
            actual = actual_results.get(match_idx)
            if not actual:
                continue
            
            # Hata var mı?
            if prediction != actual:
                error_case = {
                    'lora_id': lora.id,
                    'lora_name': lora.name,
                    'match_idx': match_idx,
                    'predicted': prediction,
                    'actual': actual,
                    'confidence': confidence,
                    'proba': proba,
                    'timestamp': datetime.now(),
                    'resolved': False
                }
                
                self.error_inbox.append(error_case)
                error_count += 1
        
        if error_count > 0:
            logger.info(f"📥 {error_count} hata Error Inbox'a eklendi (Maç #{match_idx})")
        
        return error_count
    
    def cluster_errors(self, min_samples: int = 3) -> Dict[str, Any]:
        """
        PRD: Hata Cluster'ları
        Benzer hataları feature vector'lerine göre DBSCAN ile gruplar
        
        Args:
            min_samples: Minimum cluster boyutu
        
        Returns:
            Cluster bilgileri
        """
        # Unresolved errors
        unresolved = [e for e in self.error_inbox if not e.get('resolved', False)]
        
        if len(unresolved) < min_samples:
            return {"clusters_created": 0, "clusters": []}
        
        logger.info(f"📊 {len(unresolved)} hata cluster'lanıyor...")
        
        # Feature vector oluştur
        vectors = []
        error_data = []
        
        for error in unresolved:
            vector = self._build_error_vector(error)
            if vector:
                vectors.append(vector)
                error_data.append(error)
        
        if len(vectors) < min_samples:
            return {"clusters_created": 0, "clusters": []}
        
        # DBSCAN clustering
        vectors_array = np.array(vectors)
        clustering = DBSCAN(eps=0.5, min_samples=min_samples).fit(vectors_array)
        
        # Cluster'ları oluştur
        clusters_created = 0
        cluster_info = {}
        
        for label in set(clustering.labels_):
            if label == -1:  # Noise
                continue
            
            cluster_errors = [error_data[i] for i, l in enumerate(clustering.labels_) if l == label]
            
            if len(cluster_errors) < min_samples:
                continue
            
            # Cluster özeti
            cluster_summary = self._create_cluster_summary(cluster_errors)
            
            cluster_id = f"cluster_{label}_{datetime.now().strftime('%Y%m%d')}"
            self.error_clusters[cluster_id] = {
                'name': cluster_id,
                'errors': cluster_errors,
                'summary': cluster_summary,
                'resolution_level': 'unresolved',
                'created_at': datetime.now()
            }
            
            # Error'ları cluster'a ata
            for error in cluster_errors:
                error['cluster_id'] = cluster_id
                error['resolved'] = False
            
            clusters_created += 1
            cluster_info[cluster_id] = {
                'name': cluster_id,
                'count': len(cluster_errors),
                'summary': cluster_summary
            }
        
        logger.info(f"✅ {clusters_created} cluster oluşturuldu")
        
        return {
            "clusters_created": clusters_created,
            "clusters": cluster_info
        }
    
    def _build_error_vector(self, error: Dict) -> Optional[List[float]]:
        """
        Error case için feature vector oluştur
        
        Vector: [confidence, predicted_type, actual_type, proba_home, proba_draw, proba_away]
        """
        try:
            proba = error.get('proba', {})
            
            # Outcome type encoding
            pred_type = hash(error.get('predicted', '')) % 10
            actual_type = hash(error.get('actual', '')) % 10
            
            vector = [
                float(error.get('confidence', 0.0)),
                float(pred_type),
                float(actual_type),
                float(proba.get('home_win', 0.0) if isinstance(proba, dict) else proba[0] if isinstance(proba, (list, np.ndarray)) else 0.0),
                float(proba.get('draw', 0.0) if isinstance(proba, dict) else proba[1] if isinstance(proba, (list, np.ndarray)) else 0.0),
                float(proba.get('away_win', 0.0) if isinstance(proba, dict) else proba[2] if isinstance(proba, (list, np.ndarray)) else 0.0),
            ]
            
            return vector
        except Exception as e:
            logger.warning(f"Vector oluşturma hatası: {e}")
            return None
    
    def _create_cluster_summary(self, errors: List[Dict]) -> str:
        """Cluster için özet oluştur"""
        if not errors:
            return "Empty cluster"
        
        # En çok görülen hata türü
        pred_counts = defaultdict(int)
        actual_counts = defaultdict(int)
        
        for error in errors:
            pred_counts[error.get('predicted', 'unknown')] += 1
            actual_counts[error.get('actual', 'unknown')] += 1
        
        most_common_pred = max(pred_counts.items(), key=lambda x: x[1])[0] if pred_counts else 'unknown'
        most_common_actual = max(actual_counts.items(), key=lambda x: x[1])[0] if actual_counts else 'unknown'
        
        summary = f"{len(errors)} hata: {most_common_pred} → {most_common_actual}"
        
        return summary
    
    def solve_level1(self, cluster_id: str) -> Dict[str, Any]:
        """
        PRD: Seviye 1 - İçsel açıklama
        Mevcut veriden root-cause bulmaya çalışır
        
        Returns:
            {'solved': bool, 'root_cause': str, 'suggested_changes': dict}
        """
        if cluster_id not in self.error_clusters:
            return {"solved": False, "reason": "Cluster bulunamadı"}
        
        cluster = self.error_clusters[cluster_id]
        errors = cluster['errors']
        
        if len(errors) < 2:
            return {"solved": False, "reason": "Yeterli örnek yok"}
        
        # Pattern analizi
        patterns = self._analyze_error_patterns(errors)
        
        # Basit root-cause bulma (LLM yok, basit analiz)
        root_cause = self._find_root_cause(errors, patterns)
        
        if root_cause and root_cause.get("confidence", 0) > 0.5:
            # Seviye 1'de çözüldü
            self.error_clusters[cluster_id]['resolution_level'] = 'level1'
            self.error_clusters[cluster_id]['root_cause'] = root_cause.get("explanation")
            
            # Evolution plan oluştur
            evolution_plan = {
                'cluster_id': cluster_id,
                'type': 'calibration',
                'description': root_cause.get("explanation", ""),
                'suggested_changes': root_cause.get("suggested_changes", {}),
                'status': 'pending'
            }
            
            self.evolution_plans.append(evolution_plan)
            
            logger.info(f"✅ Cluster {cluster_id} Seviye 1'de çözüldü")
            
            return {
                "solved": True,
                "level": "level1",
                "root_cause": root_cause.get("explanation"),
                "suggested_changes": root_cause.get("suggested_changes")
            }
        else:
            # Seviye 1'de çözülemedi, Seviye 2'ye geç
            return {"solved": False, "reason": "Root-cause bulunamadı", "next_level": "level2"}
    
    def _analyze_error_patterns(self, errors: List[Dict]) -> Dict[str, Any]:
        """Hata pattern'lerini analiz et"""
        patterns = {
            'high_confidence_errors': 0,
            'low_confidence_errors': 0,
            'prediction_bias': defaultdict(int),
            'common_mistakes': defaultdict(int)
        }
        
        for error in errors:
            confidence = error.get('confidence', 0.0)
            
            if confidence > 0.7:
                patterns['high_confidence_errors'] += 1
            elif confidence < 0.4:
                patterns['low_confidence_errors'] += 1
            
            pred = error.get('predicted', 'unknown')
            actual = error.get('actual', 'unknown')
            patterns['prediction_bias'][pred] += 1
            patterns['common_mistakes'][f"{pred}→{actual}"] += 1
        
        return patterns
    
    def _find_root_cause(self, errors: List[Dict], patterns: Dict) -> Optional[Dict[str, Any]]:
        """Root-cause bul (basit analiz)"""
        # Yüksek güven ile yanlış tahmin = bias problemi
        if patterns['high_confidence_errors'] > len(errors) * 0.5:
            return {
                "explanation": "Yüksek güven ile yanlış tahminler - Model yanlış pattern öğrenmiş (bias)",
                "suggested_changes": {
                    "type": "bias_correction",
                    "action": "Model kalibrasyonu gerekli"
                },
                "confidence": 0.7
            }
        
        # Düşük güven ile yanlış tahmin = variance problemi
        if patterns['low_confidence_errors'] > len(errors) * 0.5:
            return {
                "explanation": "Düşük güven ile yanlış tahminler - Model belirsiz, daha fazla feature gerekli",
                "suggested_changes": {
                    "type": "feature_enhancement",
                    "action": "Feature engineering gerekli"
                },
                "confidence": 0.6
            }
        
        # Belirli bir tahmin hatası yaygınsa
        if patterns['common_mistakes']:
            most_common = max(patterns['common_mistakes'].items(), key=lambda x: x[1])
            if most_common[1] > len(errors) * 0.4:
                return {
                    "explanation": f"Yaygın hata: {most_common[0]} - Belirli bir pattern'e karşı bias var",
                    "suggested_changes": {
                        "type": "pattern_correction",
                        "action": f"{most_common[0]} pattern'i için özel düzeltme"
                    },
                    "confidence": 0.65
                }
        
        return None
    
    def solve_level2(self, cluster_id: str) -> Dict[str, Any]:
        """
        PRD: Seviye 2 - Veri zenginleştirme
        (Placeholder - API entegrasyonu gerekli)
        """
        return {"solved": False, "reason": "Seviye 2 henüz implement edilmedi"}
    
    def ask_user_question(self, cluster_id: str) -> Dict[str, Any]:
        """
        PRD: Seviye 3 - Kullanıcıya soru sorma
        Unresolved cluster'lar için kullanıcıya soru üretir
        """
        if cluster_id not in self.error_clusters:
            return {"question": None, "reason": "Cluster bulunamadı"}
        
        cluster = self.error_clusters[cluster_id]
        errors = cluster['errors']
        
        # Basit soru üretimi
        question = f"Cluster {cluster_id}: {len(errors)} benzer hata tespit edildi. "
        question += f"Özet: {cluster.get('summary', 'Bilinmeyen')}. "
        question += "Bu hataların nedeni nedir? Hangi feature eksik olabilir?"
        
        feedback = {
            'cluster_id': cluster_id,
            'question': question,
            'timestamp': datetime.now(),
            'answered': False
        }
        
        self.human_feedback.append(feedback)
        
        logger.info(f"❓ Kullanıcıya soru soruldu: {question[:100]}...")
        
        return {
            "question": question,
            "feedback_id": len(self.human_feedback) - 1
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Evolution Core istatistikleri"""
        unresolved = len([e for e in self.error_inbox if not e.get('resolved', False)])
        resolved = len([e for e in self.error_inbox if e.get('resolved', False)])
        
        return {
            'total_errors': len(self.error_inbox),
            'resolved_errors': resolved,
            'unresolved_errors': unresolved,
            'clusters': len(self.error_clusters),
            'evolution_plans': len(self.evolution_plans),
            'human_feedback': len(self.human_feedback)
        }

