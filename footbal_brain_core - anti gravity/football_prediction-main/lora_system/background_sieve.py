"""
🕸️ BACKGROUND SIEVE SYSTEM - KATEGORİZASYON ELEĞİ
==================================================

Kullanıcının istediği "Arka plan elek sistemi".
LoRA'ları sadece etiketlerine göre değil, gerçek davranışlarına (prediction vectors)
ve hatalarına göre analiz edip "Kabilelere" (Tribes) ayırır.

Bu sistem sürekli çalışır ve LoRA'ları doğru kutulara yerleştirir.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from typing import List, Dict, Any
import collections

class BackgroundSieve:
    """
    Arka Plan Elek Sistemi
    """

    def __init__(self, buffer_size: int = 50):
        self.buffer_size = buffer_size
        self.prediction_history = collections.defaultdict(list) # {lora_id: [pred1, pred2...]}
        self.error_history = collections.defaultdict(list) # {lora_id: [error1, error2...]}
        self.clusters = {} # {lora_id: cluster_id}
        self.cluster_profiles = {} # {cluster_id: 'High Risk Taker', 'Safe Player' etc.}

    def record_behavior(self, lora_id: str, prediction_vector: np.ndarray, is_correct: bool, error_margin: float):
        """
        Her maç sonrası LoRA'nın davranışını kaydet.
        """
        # Sadece son N davranışı tut
        if len(self.prediction_history[lora_id]) >= self.buffer_size:
            self.prediction_history[lora_id].pop(0)
            self.error_history[lora_id].pop(0)

        self.prediction_history[lora_id].append(prediction_vector) # [p_home, p_draw, p_away]
        self.error_history[lora_id].append(error_margin if not is_correct else 0.0)

    def run_sieve(self, population: List[Any]):
        """
        Eleği çalıştır: LoRA'ları kümelere ayır.
        """
        if len(population) < 5:
            return

        # Feature extraction for clustering
        # Her LoRA için: [Ortalama Hata, Home Tercih Oranı, Draw Tercih Oranı, Risk İştahı (Variance)]
        features = []
        valid_loras = []

        for lora in population:
            pid = lora.id
            if len(self.prediction_history[pid]) < 10:
                continue

            preds = np.array(self.prediction_history[pid])
            errors = np.array(self.error_history[pid])

            avg_error = np.mean(errors)
            home_bias = np.mean(preds[:, 0])
            draw_bias = np.mean(preds[:, 1])
            risk_appetite = np.var(preds) # Yüksek varyans = çok emin veya çok kararsız değil, dalgalı
            confidence_avg = np.mean(np.max(preds, axis=1)) # Kendine ne kadar güveniyor?

            features.append([avg_error, home_bias, draw_bias, risk_appetite, confidence_avg])
            valid_loras.append(lora)

        if not features:
            return

        X = np.array(features)

        # DBSCAN ile doğal kümeleri bul (Density based)
        # eps ve min_samples parametreleri veri dağılımına göre ayarlanmalı
        clustering = DBSCAN(eps=0.2, min_samples=2).fit(X)
        labels = clustering.labels_

        # Sonuçları işle
        new_clusters = {}
        for i, label in enumerate(labels):
            lora = valid_loras[i]
            if label != -1: # Noise değilse
                new_clusters[lora.id] = int(label)
                # LoRA'ya etiketini yapıştır
                if not hasattr(lora, 'sieve_tags'):
                    lora.sieve_tags = []

                # Küme özelliklerine göre tag ver
                cluster_feats = X[labels == label].mean(axis=0)
                # [Err, Home, Draw, Risk, Conf]
                tag = self._generate_tag(cluster_feats)
                lora.sieve_tags.append(tag)

        self.clusters = new_clusters
        print(f"🕸️ ELEK SİSTEMİ: {len(set(labels)) - (1 if -1 in labels else 0)} kabile tespit edildi.")

        return self._group_by_cluster(valid_loras, labels)

    def _group_by_cluster(self, loras, labels):
        """
        LoRA'ları cluster ID'lerine göre grupla.
        Returns: {cluster_id: [lora1, lora2...]}
        """
        groups = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1: # Noise hariç
                groups[int(label)].append(loras[i])
        return groups

    def _generate_tag(self, features):
        """Kümeye isim ver"""
        avg_err, home, draw, risk, conf = features

        if conf > 0.8:
            return "tribe_overconfident"
        if risk > 0.1:
            return "tribe_chaotic"
        if home > 0.5:
            return "tribe_home_lover"
        if draw > 0.4:
            return "tribe_draw_hunter"
        if avg_err < 0.3:
            return "tribe_elite"

        return "tribe_average"
