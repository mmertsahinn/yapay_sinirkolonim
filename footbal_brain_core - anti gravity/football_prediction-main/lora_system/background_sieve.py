"""
🕸️ BACKGROUND SIEVE SYSTEM - KATEGORİZASYON ELEĞİ
==================================================

Kullanıcının istediği "Arka plan elek sistemi".

LoRA'ları sadece etiketlerine göre değil, gerçek davranışlarına (prediction vectors)
ve hatalarına göre analiz edip "Kabilelere" (Tribes) ayırır.

Bu sistem sürekli çalışır ve LoRA'ları doğru kutulara yerleştirir.

Bilimsel Temel:
- Unsupervised Learning (Clustering)
- Behavioral Analysis
- Pattern Recognition
- DBSCAN: Density-Based Spatial Clustering of Applications with Noise

Özellikler:
✅ Prediction history tracking (circular buffer)
✅ Error history tracking
✅ Feature extraction (avg_error, home_bias, draw_bias, risk_appetite, confidence)
✅ DBSCAN clustering (doğal kümeler)
✅ Tribe etiketleme (behavioral categories)
✅ Lazy update (performance için)
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')  # DBSCAN convergence warnings


class BackgroundSieve:
    """
    Arka Plan Elek Sistemi
    
    LoRA'ları davranışlarına göre kategorize eder.
    """
    
    def __init__(self, buffer_size: int = 50, update_frequency: int = 10, min_samples_for_clustering: int = 10):
        """
        Args:
            buffer_size: Her LoRA için tutulacak prediction sayısı
            update_frequency: Kaç maçta bir clustering yapılacak
            min_samples_for_clustering: Clustering için minimum örnek sayısı
        """
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.min_samples_for_clustering = min_samples_for_clustering
        
        # Prediction history (circular buffer)
        self.prediction_history = defaultdict(lambda: deque(maxlen=buffer_size))
        
        # Error history (circular buffer)
        self.error_history = defaultdict(lambda: deque(maxlen=buffer_size))
        
        # Clustering results
        self.clusters = {}  # {lora_id: cluster_id}
        self.cluster_profiles = {}  # {cluster_id: {'name': 'tribe_elite', 'features': [...]}}
        
        # Update tracking
        self.last_update_match = 0
        self.population_snapshot = {}  # Son clustering'deki popülasyon
        
        print(f"✅ BackgroundSieve initialized (buffer={buffer_size}, update_freq={update_frequency})")
    
    def record_behavior(self, 
                       lora_id: str, 
                       prediction_vector: np.ndarray, 
                       is_correct: bool, 
                       error_margin: float):
        """
        Her maç sonrası LoRA'nın davranışını kaydet.
        
        Args:
            lora_id: LoRA ID
            prediction_vector: [p_home, p_draw, p_away] (3 boyut)
            is_correct: Doğru tahmin mi?
            error_margin: Hata marjı (0-1, yanlışsa confidence, doğruysa 0)
        """
        # Prediction history
        self.prediction_history[lora_id].append(prediction_vector.copy())
        
        # Error history (yanlışsa error_margin, doğruysa 0)
        error_value = error_margin if not is_correct else 0.0
        self.error_history[lora_id].append(error_value)
    
    def _extract_features(self, lora_id: str) -> Optional[np.ndarray]:
        """
        Bir LoRA için feature extraction
        
        Features:
        1. avg_error: Ortalama hata (0-1)
        2. home_bias: Home tercih oranı (0-1)
        3. draw_bias: Draw tercih oranı (0-1)
        4. risk_appetite: Varyans (yüksek = dalgalı, düşük = tutarlı)
        5. confidence_avg: Ortalama güven (0-1)
        
        Returns:
            Feature vector [5] veya None (yeterli veri yoksa)
        """
        if len(self.prediction_history[lora_id]) < self.min_samples_for_clustering:
            return None
        
        preds = np.array(list(self.prediction_history[lora_id]))
        errors = np.array(list(self.error_history[lora_id]))
        
        # Feature 1: Ortalama hata
        avg_error = np.mean(errors)
        
        # Feature 2: Home bias (home tercih oranı)
        home_bias = np.mean(preds[:, 0])  # İlk sütun = home
        
        # Feature 3: Draw bias
        draw_bias = np.mean(preds[:, 1])  # İkinci sütun = draw
        
        # Feature 4: Risk appetite (varyans)
        # Yüksek varyans = çok emin veya çok kararsız değil, dalgalı tahminler
        risk_appetite = np.var(preds.flatten())
        
        # Feature 5: Ortalama güven (max probability)
        confidence_avg = np.mean(np.max(preds, axis=1))
        
        features = np.array([
            avg_error,
            home_bias,
            draw_bias,
            risk_appetite,
            confidence_avg
        ], dtype=np.float32)
        
        return features
    
    def _generate_tribe_tag(self, cluster_features: np.ndarray) -> str:
        """
        Küme özelliklerine göre tribe etiketi oluştur
        
        Args:
            cluster_features: [avg_error, home_bias, draw_bias, risk_appetite, confidence_avg]
            
        Returns:
            Tribe tag string
        """
        avg_err, home, draw, risk, conf = cluster_features
        
        # Elite: Düşük hata, yüksek güven
        if avg_err < 0.3 and conf > 0.7:
            return "tribe_elite"
        
        # Overconfident: Yüksek güven ama yüksek hata
        if conf > 0.8 and avg_err > 0.5:
            return "tribe_overconfident"
        
        # Chaotic: Yüksek risk (varyans)
        if risk > 0.1:
            return "tribe_chaotic"
        
        # Home Lover: Home bias yüksek
        if home > 0.5:
            return "tribe_home_lover"
        
        # Draw Hunter: Draw bias yüksek
        if draw > 0.4:
            return "tribe_draw_hunter"
        
        # Conservative: Düşük risk, orta güven
        if risk < 0.05 and 0.5 < conf < 0.7:
            return "tribe_conservative"
        
        # Average: Diğerleri
        return "tribe_average"
    
    def _should_update(self, current_match: int, population: List[Any]) -> bool:
        """
        Clustering güncellemesi gerekli mi?
        
        Koşullar:
        1. update_frequency maç geçti
        2. Popülasyon %20'den fazla değişti
        """
        # Koşul 1: Frequency check
        if current_match - self.last_update_match >= self.update_frequency:
            return True
        
        # Koşul 2: Popülasyon değişimi
        current_ids = {lora.id for lora in population}
        previous_ids = set(self.population_snapshot.keys())
        
        if len(previous_ids) == 0:
            return True  # İlk kez
        
        # Değişim oranı
        new_loras = current_ids - previous_ids
        removed_loras = previous_ids - current_ids
        change_ratio = (len(new_loras) + len(removed_loras)) / max(len(previous_ids), 1)
        
        if change_ratio > 0.2:  # %20'den fazla değişim
            return True
        
        return False
    
    def run_sieve(self, population: List[Any], current_match: int = 0, force_update: bool = False):
        """
        Eleği çalıştır: LoRA'ları kümelere ayır.
        
        Process:
        1. Update gerekli mi kontrol et (lazy update)
        2. Feature extraction (yeterli verisi olanlar için)
        3. DBSCAN clustering
        4. Tribe etiketleme
        5. LoRA'lara tag ekle
        
        Args:
            population: LoRA popülasyonu
            current_match: Mevcut maç sayısı
            force_update: Zorla güncelle (lazy update'ı bypass et)
        """
        if len(population) < 5:
            return  # Yeterli popülasyon yok
        
        # Lazy update kontrolü
        if not force_update and not self._should_update(current_match, population):
            return  # Henüz güncelleme gerekmiyor
        
        # Feature extraction
        features = []
        valid_loras = []
        
        for lora in population:
            lora_features = self._extract_features(lora.id)
            if lora_features is not None:
                features.append(lora_features)
                valid_loras.append(lora)
        
        if len(features) < 5:
            return  # Yeterli feature yok
        
        X = np.array(features)
        
        # Feature normalization (DBSCAN için önemli!)
        # Her feature'ı 0-1 arasına normalize et
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Division by zero önleme
        X_normalized = (X - X_min) / X_range
        
        # DBSCAN clustering (Density-based, noise handling)
        # eps: Komşuluk yarıçapı (normalize edilmiş feature space'de)
        # min_samples: Minimum komşu sayısı (küme için)
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='euclidean')
        labels = clustering.fit_predict(X_normalized)
        
        # Sonuçları işle
        new_clusters = {}
        cluster_feature_means = defaultdict(list)
        
        for i, label in enumerate(labels):
            lora = valid_loras[i]
            
            if label != -1:  # Noise değilse (küme içinde)
                new_clusters[lora.id] = int(label)
                cluster_feature_means[label].append(features[i])
        
        # Cluster profilleri oluştur
        for cluster_id, cluster_feats in cluster_feature_means.items():
            cluster_mean = np.mean(cluster_feats, axis=0)
            tribe_tag = self._generate_tribe_tag(cluster_mean)
            
            self.cluster_profiles[cluster_id] = {
                'name': tribe_tag,
                'features': cluster_mean.tolist(),
                'size': len(cluster_feats)
            }
        
        # LoRA'lara tag ekle
        for lora in valid_loras:
            if lora.id in new_clusters:
                cluster_id = new_clusters[lora.id]
                tribe_tag = self.cluster_profiles[cluster_id]['name']
                
                # LoRA'ya tag ekle
                if not hasattr(lora, 'sieve_tags'):
                    lora.sieve_tags = []
                
                # Yeni tag ekle (duplicate kontrolü)
                if tribe_tag not in lora.sieve_tags:
                    lora.sieve_tags.append(tribe_tag)
        
        # Güncelleme tracking
        self.clusters = new_clusters
        self.population_snapshot = {lora.id: lora for lora in population}
        self.last_update_match = current_match
        
        # İstatistikler
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = list(labels).count(-1)
        
        if num_clusters > 0:
            print(f"🕸️ SIEVE: {num_clusters} kabile tespit edildi ({len(valid_loras)} LoRA, {num_noise} noise)")
    
    def get_tribe_distribution(self, population: List[Any]) -> Dict[str, int]:
        """
        Tribe dağılımını döndür
        
        Returns:
            {tribe_name: count}
        """
        distribution = defaultdict(int)
        
        for lora in population:
            if hasattr(lora, 'sieve_tags') and lora.sieve_tags:
                for tag in lora.sieve_tags:
                    distribution[tag] += 1
        
        return dict(distribution)
    
    def get_lora_tribe(self, lora_id: str) -> Optional[str]:
        """
        Bir LoRA'nın tribe'ini döndür
        
        Returns:
            Tribe name veya None
        """
        if lora_id in self.clusters:
            cluster_id = self.clusters[lora_id]
            if cluster_id in self.cluster_profiles:
                return self.cluster_profiles[cluster_id]['name']
        return None
    
    def get_tribes(self, population: List[Any]) -> Dict[int, List[Any]]:
        """
        Popülasyondan kabileleri döndür (cluster_id → [lora1, lora2, ...])
        
        Args:
            population: LoRA popülasyonu
        
        Returns:
            {cluster_id: [lora1, lora2, ...]} - Kabileler
        """
        tribes = {}
        
        for lora in population:
            if lora.id in self.clusters:
                cluster_id = self.clusters[lora.id]
                if cluster_id not in tribes:
                    tribes[cluster_id] = []
                tribes[cluster_id].append(lora)
        
        # Noise'ları (cluster_id == -1) hariç tut
        tribes = {k: v for k, v in tribes.items() if k != -1}
        
        return tribes
    
    def clear_history(self, lora_id: str):
        """Bir LoRA'nın geçmişini temizle (ölüm sonrası)"""
        if lora_id in self.prediction_history:
            del self.prediction_history[lora_id]
        if lora_id in self.error_history:
            del self.error_history[lora_id]
        if lora_id in self.clusters:
            del self.clusters[lora_id]


# Global instance
_global_sieve = None


def get_background_sieve(buffer_size: int = 50, 
                        update_frequency: int = 10) -> BackgroundSieve:
    """Global BackgroundSieve instance"""
    global _global_sieve
    if _global_sieve is None:
        _global_sieve = BackgroundSieve(
            buffer_size=buffer_size,
            update_frequency=update_frequency
        )
    return _global_sieve

