"""
💕 ULTRA KAOTİK ÇİFTLEŞME SİSTEMİ
==================================

Partner seçimi TAM AKIŞKAN!

FAKTÖRLER:
1. SOSYAL BAĞ (50%) - En güçlü bağ
2. MİZAÇ ÇEKİMİ (20%) - Benzer veya zıt mizaçlar
3. SÜRPRİZ (20%) - Cani + Yumuşak gibi beklenmedik!
4. TAM RASTGELE (10%) - Kaos!

"En cani insanın kıyamadığı yumuşak biri vardır!" 💘
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import random


class UltraChaoticMating:
    """
    Ultra kaotik partner seçimi
    """
    
    @staticmethod
    def select_partner(lora, population: List, social_bonds: Dict = None) -> Tuple:
        """
        AKIŞKAN PARTNER SEÇİMİ!
        
        Kodlanmış %30-%30 YOK!
        Tam dinamik, mizaç + bağ + sürpriz!
        
        Args:
            lora: Seçim yapan LoRA
            population: Mevcut popülasyon
            social_bonds: Sosyal bağlar (opsiyonel)
        
        Returns:
            (partner, reason)
        """
        if len(population) < 2:
            return None, "Popülasyon yetersiz"
        
        others = [l for l in population if l.id != lora.id]
        if len(others) == 0:
            return None, "Başka LoRA yok"
        
        # ============================================
        # FAKTÖR 1: SOSYAL BAĞ (50%)
        # ============================================
        
        bond_candidates = []
        
        if hasattr(lora, 'social_bonds') and len(lora.social_bonds) > 0:
            # En güçlü bağa sahip LoRA'ları bul
            for other in others:
                bond_strength = lora.social_bonds.get(other.id, 0.0)
                
                if bond_strength > 0.3:  # Anlamlı bağ
                    bond_candidates.append({
                        'lora': other,
                        'bond': bond_strength,
                        'score': bond_strength * 0.50  # %50 ağırlık
                    })
        
        # ============================================
        # FAKTÖR 2: MİZAÇ ÇEKİMİ (20%)
        # ============================================
        
        temperament_candidates = []
        
        for other in others:
            # Mizaç benzerliği veya zıtlığı
            compatibility = UltraChaoticMating._calculate_temperament_compatibility(
                lora, other
            )
            
            if compatibility > 0.5:  # Uyumlu veya ilginç zıt!
                temperament_candidates.append({
                    'lora': other,
                    'compatibility': compatibility,
                    'score': compatibility * 0.20  # %20 ağırlık
                })
        
        # ============================================
        # FAKTÖR 3: SÜRPRİZ (20%) - CANİ + YUMUŞAK!
        # ============================================
        
        surprise_candidates = []
        
        # Beklenmedik kombinasyonlar!
        for other in others:
            surprise_score = UltraChaoticMating._calculate_surprise_factor(
                lora, other
            )
            
            if surprise_score > 0.6:  # Sürpriz yüksek!
                surprise_candidates.append({
                    'lora': other,
                    'surprise': surprise_score,
                    'score': surprise_score * 0.20  # %20 ağırlık
                })
        
        # ============================================
        # FAKTÖR 4: TAM RASTGELE (10%)
        # ============================================
        
        random_candidate = random.choice(others)
        random_score = 0.10
        
        # ============================================
        # TÜM ADAYLARI BİRLEŞTİR
        # ============================================
        
        all_candidates = {}
        
        # Sosyal bağ adayları
        for cand in bond_candidates:
            lora_id = cand['lora'].id
            if lora_id not in all_candidates:
                all_candidates[lora_id] = {'lora': cand['lora'], 'total_score': 0.0, 'reasons': []}
            all_candidates[lora_id]['total_score'] += cand['score']
            all_candidates[lora_id]['reasons'].append(f"Güçlü bağ ({cand['bond']:.2f})")
        
        # Mizaç adayları
        for cand in temperament_candidates:
            lora_id = cand['lora'].id
            if lora_id not in all_candidates:
                all_candidates[lora_id] = {'lora': cand['lora'], 'total_score': 0.0, 'reasons': []}
            all_candidates[lora_id]['total_score'] += cand['score']
            all_candidates[lora_id]['reasons'].append(f"Mizaç uyumu ({cand['compatibility']:.2f})")
        
        # Sürpriz adayları
        for cand in surprise_candidates:
            lora_id = cand['lora'].id
            if lora_id not in all_candidates:
                all_candidates[lora_id] = {'lora': cand['lora'], 'total_score': 0.0, 'reasons': []}
            all_candidates[lora_id]['total_score'] += cand['score']
            all_candidates[lora_id]['reasons'].append(f"Sürpriz ({cand['surprise']:.2f})")
        
        # Rastgele adayı ekle
        if random_candidate.id not in all_candidates:
            all_candidates[random_candidate.id] = {
                'lora': random_candidate,
                'total_score': random_score,
                'reasons': ['Tam rastgele (kaos!)']
            }
        else:
            all_candidates[random_candidate.id]['total_score'] += random_score
            all_candidates[random_candidate.id]['reasons'].append('Rastgele bonus')
        
        # En yüksek skoru seç
        if len(all_candidates) == 0:
            return random.choice(others), "Rastgele (varsayılan)"
        
        best_candidate = max(all_candidates.values(), key=lambda x: x['total_score'])
        
        reason = ', '.join(best_candidate['reasons'])
        
        return best_candidate['lora'], reason
    
    @staticmethod
    def _calculate_temperament_compatibility(lora1, lora2) -> float:
        """
        Mizaç uyumluluğu (Benzer VEYA ilginç zıt!)
        
        Returns:
            0-1 arası (0.5+ = uyumlu)
        """
        temp1 = lora1.temperament
        temp2 = lora2.temperament
        
        # BENZERLIK SKORU
        similarities = []
        for key in temp1.keys():
            val1 = temp1.get(key, 0.5)
            val2 = temp2.get(key, 0.5)
            similarity = 1.0 - abs(val1 - val2)  # 0-1 arası
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        
        # ZITLIK SKORU (İlginç kombinasyonlar!)
        # Bazı özellikler zıt olduğunda ilginç!
        interesting_opposites = 0.0
        
        # Hırslı + Sakin = İlginç!
        if temp1.get('ambition', 0.5) > 0.7 and temp2.get('stress_tolerance', 0.5) > 0.7:
            interesting_opposites += 0.2
        
        # Duygusal + Bağımsız = İlginç!
        if temp1.get('emotional_depth', 0.5) > 0.7 and temp2.get('independence', 0.5) > 0.7:
            interesting_opposites += 0.2
        
        # Sinirli + Sabırlı = İlginç!
        if temp1.get('anger_tendency', 0.5) > 0.7 and temp2.get('patience', 0.5) > 0.7:
            interesting_opposites += 0.2
        
        # TOPLAM: Benzerlik VEYA İlginç Zıtlık
        compatibility = max(avg_similarity, interesting_opposites)
        
        return compatibility
    
    @staticmethod
    def _calculate_surprise_factor(lora1, lora2) -> float:
        """
        Sürpriz faktörü (Cani + Yumuşak gibi!)
        
        Returns:
            0-1 arası (0.6+ = çok sürpriz!)
        """
        temp1 = lora1.temperament
        temp2 = lora2.temperament
        
        surprises = []
        
        # CANİ (Sinirli + Dürtüsel + Risk sever) + YUMUŞAK (Empati + Duygusal)
        lora1_aggressive = (
            temp1.get('anger_tendency', 0.5) +
            temp1.get('impulsiveness', 0.5) +
            temp1.get('risk_appetite', 0.5)
        ) / 3.0
        
        lora2_gentle = (
            temp2.get('empathy', 0.5) +
            temp2.get('emotional_depth', 0.5) +
            temp2.get('patience', 0.5)
        ) / 3.0
        
        # Hem cani hem yumuşak mı?
        if lora1_aggressive > 0.7 and lora2_gentle > 0.7:
            surprises.append(0.8)  # Çok sürpriz!
        
        # Veya tersi?
        lora1_gentle = (temp1.get('empathy', 0.5) + temp1.get('emotional_depth', 0.5)) / 2.0
        lora2_aggressive = (temp2.get('anger_tendency', 0.5) + temp2.get('impulsiveness', 0.5)) / 2.0
        
        if lora1_gentle > 0.7 and lora2_aggressive > 0.7:
            surprises.append(0.8)
        
        # ZENGİN + FAKIR (Yüksek fitness + Düşük fitness)
        fit1 = lora1.get_recent_fitness()
        fit2 = lora2.get_recent_fitness()
        
        if abs(fit1 - fit2) > 0.40:  # Çok farklı!
            surprises.append(0.7)
        
        # YAŞLI + GENÇ
        age1 = len(lora1.fitness_history)
        age2 = len(lora2.fitness_history)
        
        if abs(age1 - age2) > 100:  # 100+ maç fark
            surprises.append(0.6)
        
        # UZMAN + ACEMI
        spec1 = getattr(lora1, 'specialization', None)
        spec2 = getattr(lora2, 'specialization', None)
        
        if (spec1 and not spec2) or (not spec1 and spec2):
            surprises.append(0.5)
        
        # En yüksek sürprizi döndür
        return max(surprises) if surprises else 0.3


# Global instance
ultra_chaotic_mating = UltraChaoticMating()

