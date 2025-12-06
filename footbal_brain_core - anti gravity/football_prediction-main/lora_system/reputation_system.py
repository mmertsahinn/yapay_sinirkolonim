"""
🏆 İTİBAR SİSTEMİ (Algısal Kimlik & Etiket Değeri)
===================================================

Kim konuşuyor? Einstein mi, sıradan biri mi?

İtibar Kaynakları:
1. FITNESS (Başarı oranı)
2. UZMANLIK (Specialist mi?)
3. DENEYIM (Yaş)
4. SOSYAL ETKI (Başkaları ona bakıyor mu?)
5. MUCİZE (Hall of Fame'de mi?)

Her LoRA'nın sözü FARKLI ağırlıkta!
"""

from typing import Dict, List, Tuple
import numpy as np


class ReputationSystem:
    """
    İtibar ve algısal kimlik sistemi
    """
    
    @staticmethod
    def calculate_reputation(lora, population: List, all_loras_ever: Dict = None, match_count: int = 0) -> Dict:
        """
        LoRA'nın itibarını hesapla (Algısal değer!)
        
        FORMÜL:
        İtibar = 
          Performance (40%) +
          Expertise (25%) +
          Experience (15%) +
          Social Influence (10%) +
          Legend Status (10%)
        
        Returns:
            {
                'total_reputation': 0-1 arası,
                'tier': 'Sıradan', 'İyi', 'Uzman', 'Usta', 'Efsane',
                'badges': ['Çifte Uzman', 'Yaşlı Bilge', ...],
                'authority_weight': 0-3 arası (yazı ağırlığı)
            }
        """
        # ============================================
        # 1. PERFORMANS (40%)
        # ============================================
        fitness = lora.get_recent_fitness()
        performance_score = fitness * 0.40
        
        # ============================================
        # 2. UZMANLIK (25%)
        # ============================================
        specialization = getattr(lora, 'specialization', None)
        expertise_score = 0.0
        badges = []
        
        if specialization:
            expertise_score = 0.15  # Uzman!
            badges.append(f"🎯 {specialization}")
            
            # Çifte uzman mı? (2+ pattern'de %70+ başarı)
            if hasattr(lora, 'pattern_attractions') and len(lora.pattern_attractions) >= 2:
                strong_patterns = [p for p, score in lora.pattern_attractions.items() if score > 0.70]
                if len(strong_patterns) >= 2:
                    expertise_score = 0.25  # ÇİFTE UZMAN!
                    badges.append("🏆 Çifte Uzman")
        
        # ============================================
        # 3. DENEYİM (15%)
        # ============================================
        age = match_count - lora.birth_match if match_count else len(lora.fitness_history)
        
        if age >= 300:
            experience_score = 0.15
            badges.append("🧓 Yaşlı Bilge")
        elif age >= 150:
            experience_score = 0.10
            badges.append("👴 Deneyimli")
        elif age >= 50:
            experience_score = 0.05
            badges.append("🧑 Olgun")
        else:
            experience_score = 0.02
        
        # ============================================
        # 4. SOSYAL ETKİ (10%)
        # ============================================
        social_influence = 0.0
        
        # Başkaları bu LoRA'ya bağlı mı?
        if hasattr(lora, 'social_bonds'):
            # Bu LoRA'nın ID'sine kaç LoRA bağlı?
            influenced_count = 0
            for other_lora in population:
                if hasattr(other_lora, 'social_bonds'):
                    if lora.id in other_lora.social_bonds:
                        bond_strength = other_lora.social_bonds[lora.id]
                        if bond_strength > 0.5:
                            influenced_count += 1
            
            # Etki oranı
            if len(population) > 0:
                influence_ratio = influenced_count / len(population)
                social_influence = min(influence_ratio * 0.5, 0.10)  # Max 0.10
                
                if influenced_count >= 5:
                    badges.append("👑 Lider")
        
        # ============================================
        # 5. EFSANE STATÜSÜ (10%)
        # ============================================
        legend_score = 0.0
        
        # Mucize mi?
        if hasattr(lora, 'is_miracle') and lora.is_miracle:
            legend_score = 0.10
            badges.append("🌟 Mucize")
        
        # Diriltilmiş mi?
        elif getattr(lora, 'resurrection_count', 0) > 0:
            legend_score = 0.05
            badges.append("⚡ Diriltilmiş")
        
        # Çok çocuk mu? (Genetik lider!)
        if getattr(lora, 'children_count', 0) >= 10:
            badges.append("👪 Genetik Lider")
        
        # ============================================
        # TOPLAM İTİBAR
        # ============================================
        total_reputation = (
            performance_score +
            expertise_score +
            experience_score +
            social_influence +
            legend_score
        )
        
        # TIER BELİRLE
        if total_reputation >= 0.80:
            tier = "Efsane"
            authority_weight = 3.0  # x3 ağırlık!
        elif total_reputation >= 0.65:
            tier = "Usta"
            authority_weight = 2.0  # x2 ağırlık
        elif total_reputation >= 0.50:
            tier = "Uzman"
            authority_weight = 1.5  # x1.5 ağırlık
        elif total_reputation >= 0.35:
            tier = "İyi"
            authority_weight = 1.0  # Normal
        else:
            tier = "Sıradan"
            authority_weight = 0.7  # Düşük ağırlık
        
        return {
            'total_reputation': total_reputation,
            'tier': tier,
            'badges': badges,
            'authority_weight': authority_weight,
            'breakdown': {
                'performance': performance_score,
                'expertise': expertise_score,
                'experience': experience_score,
                'social_influence': social_influence,
                'legend_status': legend_score
            }
        }
    
    @staticmethod
    def should_listen_to(listener_lora, speaker_lora, speaker_reputation: Dict) -> Tuple[bool, float, str]:
        """
        Dinlemeli mi? (Akışkan karar!)
        
        Args:
            listener_lora: Dinleyen LoRA
            speaker_lora: Konuşan LoRA
            speaker_reputation: Konuşanın itibarı
        
        Returns:
            (should_listen, attention_weight, reason)
        """
        temp = listener_lora.temperament
        
        # TEMEL FAKTÖRLER
        independence = temp.get('independence', 0.5)
        social_intelligence = temp.get('social_intelligence', 0.5)
        herd_tendency = temp.get('herd_tendency', 0.5)
        contrarian = temp.get('contrarian_score', 0.5)
        
        # KONUŞAN KİM?
        speaker_tier = speaker_reputation['tier']
        speaker_authority = speaker_reputation['authority_weight']
        speaker_badges = speaker_reputation['badges']
        
        # ============================================
        # FORMÜL BAZLI KARAR (Akışkan!)
        # ============================================
        
        # BASE DİNLEME OLASILIĞI (mizaç bazlı)
        base_listen = (
            social_intelligence * 0.40 +  # Sosyal zeki çok dinler
            herd_tendency * 0.30 +        # Sürü eğilimi dinler
            (1 - independence) * 0.20 +   # Bağımsız az dinler
            (1 - contrarian) * 0.10       # Karşıt dinlemez
        )
        
        # KONUŞANIN İTİBARI ETKİSİ (authority weight)
        reputation_boost = (speaker_authority - 1.0) * 0.3  # -0.3 to +0.6
        
        # ÖZEL DURUMLAR (Badges!)
        special_boost = 0.0
        
        if "🏆 Çifte Uzman" in speaker_badges:
            # Çifte uzman → BAĞIMSIZ BİLE DURAKSAR!
            special_boost += 0.25
        
        if "🌟 Mucize" in speaker_badges:
            # Mucize → Herkes dinler!
            special_boost += 0.30
        
        if "🧓 Yaşlı Bilge" in speaker_badges:
            # Yaşlı bilge → Deneyim saygısı
            special_boost += 0.15
        
        # TOPLAM
        final_listen_probability = base_listen + reputation_boost + special_boost
        final_listen_probability = max(0.0, min(1.0, final_listen_probability))
        
        # ATTENTION WEIGHT (yazıya ne kadar ağırlık verilir?)
        attention_weight = final_listen_probability * speaker_authority
        
        # KARAR
        should_listen = final_listen_probability > 0.40
        
        # SEBEP
        if "🏆 Çifte Uzman" in speaker_badges and independence > 0.8:
            reason = "Bağımsızım ama bu kişi çifte uzman, duraksadım!"
        elif speaker_tier == "Efsane":
            reason = f"Efsane birisi konuşuyor ({speaker_tier}), dinlemeliyim!"
        elif social_intelligence > 0.7:
            reason = f"Sosyal zekanım yüksek, {speaker_tier} birinden öğrenebilirim"
        elif independence > 0.8:
            reason = "Bağımsızım, kendi kafama göre yaparım"
        elif herd_tendency > 0.7:
            reason = "Başarılı biri ne diyorsa onu yaparım"
        else:
            reason = "Dengeli yaklaşım"
        
        return should_listen, attention_weight, reason


# Global instance
reputation_system = ReputationSystem()

