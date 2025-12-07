"""
🦋 BUTTERFLY EFFECT MODULE - Sosyal Kaos & Noise Injection
===========================================================

Bu modül, "Kelebek Etkisi"ni simüle eder.
Bir LoRA'daki küçük bir değişiklik (ağırlık değişimi, travma, başarı),
sosyal ağ üzerinden yayılarak diğer LoRA'larda kaotik dalgalanmalara neden olur.

Özellikler:
1. Trigger Event: Bir LoRA'da "önemli" bir olay (büyük loss, rank değişimi).
2. Propagation: Sosyal ağdaki komşularına dalga yayılımı.
3. Noise Injection: Dalganın vurduğu LoRA'ların ağırlıklarına veya temperament'ına gürültü ekleme.
"""

import numpy as np
import torch
from typing import List, Dict, Any

class ButterflyEffect:
    """
    Kelebek Etkisi ve Kaos Modülü
    """

    def __init__(self, social_network):
        self.social_network = social_network
        self.chaos_history = []

    def trigger_effect(self, source_lora: Any, event_magnitude: float, population: List[Any]) -> List[str]:
        """
        Kelebek etkisini tetikle.

        Args:
            source_lora: Olayın kaynağı olan LoRA
            event_magnitude: Olayın büyüklüğü (0.0 - 1.0)
            population: Tüm LoRA popülasyonu

        Returns:
            List of affected LoRA names (log için)
        """
        # Eşik kontrolü (Çok küçük olaylar kelebek etkisi yaratmaz)
        if event_magnitude < 0.3:
            return []

        affected_names = []

        # 1. Sosyal Komşuları Bul
        # SocialNetwork sınıfında get_social_cluster var
        # Eğer social_network parametresi bir instance ise direkt kullanırız
        neighbors = self.social_network.get_social_cluster(source_lora.id, threshold=0.3)

        if not neighbors:
            return []

        # 2. Dalga Yayılımı (Propagation)
        for neighbor_id in neighbors:
            neighbor = next((l for l in population if l.id == neighbor_id), None)
            if not neighbor:
                continue

            # Bağ gücünü al
            bond_strength = self.social_network.get_bond_strength(source_lora.id, neighbor_id)

            # Etki hesapla: Magnitude * Bond * ChaosFactor
            impact = event_magnitude * bond_strength * (np.random.random() + 0.5)

            # 3. Noise Injection (Etkiyi uygula)
            self._inject_noise(neighbor, impact)
            affected_names.append(neighbor.name)

            # 4. İkinci Derece Yayılım (Zincirleme Reaksiyon - Azalarak)
            # %20 ihtimalle komşunun komşusuna da sıçrar
            if impact > 0.5 and np.random.random() < 0.2:
                secondary_neighbors = self.social_network.get_social_cluster(neighbor_id, threshold=0.4)
                for sec_id in secondary_neighbors:
                    if sec_id == source_lora.id: continue # Geri sekme yok

                    sec_neighbor = next((l for l in population if l.id == sec_id), None)
                    if sec_neighbor:
                        sec_impact = impact * 0.5 # Yarıya düşer
                        self._inject_noise(sec_neighbor, sec_impact)

        # Log
        if affected_names:
            print(f"🦋 KELEBEK ETKİSİ: {source_lora.name} -> {len(affected_names)} komşuyu etkiledi!")

        return affected_names

    def _inject_noise(self, lora: Any, impact: float):
        """
        LoRA'ya gürültü (noise) enjekte et.

        Etkiler:
        1. Temperament değişimi (geçici mood swing)
        2. Ağırlık perturbasyonu (kalıcı micro-change)
        """
        # 1. Temperament Noise
        # Risk iştahını veya dürtüselliği artır/azalt
        if hasattr(lora, 'temperament'):
            noise = (np.random.random() - 0.5) * impact * 0.5
            lora.temperament['risk_appetite'] = np.clip(lora.temperament.get('risk_appetite', 0.5) + noise, 0.0, 1.0)
            lora.temperament['impulsiveness'] = np.clip(lora.temperament.get('impulsiveness', 0.5) + noise, 0.0, 1.0)

        # 2. Weight Perturbation (Gaussian Noise to LoRA matrices)
        # Sadece çok yüksek impact varsa parametrelere dokun
        if impact > 0.6:
            try:
                params = lora.get_all_lora_params()
                device = params['fc1']['lora_A'].device

                # Rastgele bir katmanı seç
                layer = np.random.choice(['fc1', 'fc2', 'fc3'])
                matrix = np.random.choice(['lora_A', 'lora_B'])

                target_tensor = params[layer][matrix]
                noise_tensor = torch.randn_like(target_tensor, device=device) * (impact * 0.01) # Çok küçük gürültü

                # Uygula
                params[layer][matrix] += noise_tensor
                lora.set_all_lora_params(params)

            except Exception as e:
                # Parametre erişiminde hata olursa (örn: tensör tipi) yut
                pass
