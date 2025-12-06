"""
🎭 MİZAÇ BAZLI ÖĞRENME SİSTEMİ
================================

Her LoRA başkalarının öğrenmelerini KENDİ MİZACINA GÖRE yorumlar!

Örnek:
- Bağımsız: "Einstein öyle demiş ama ben kendi kafama göre yaparım!" (%10 kullanır)
- Sosyal Zeki: "Einstein'dan çok şey öğrenebilirim!" (%80 kullanır)
- Karşıt: "Einstein öyle dedi, ben tersini yapayım!" (Ters kullanır!)
"""

from typing import Dict, List
import random


class TemperamentBasedLearning:
    """
    Mizaç bazlı öğrenme yorumlayıcısı
    """
    
    @staticmethod
    def interpret_others_learning(lora, others_learning: Dict, collective_memory) -> Dict:
        """
        Başkalarının öğrenmelerini MİZAÇA GÖRE yorumla!
        
        Args:
            lora: Bu LoRA (kim yorumluyor?)
            others_learning: Başkalarının öğrenme geçmişi
            collective_memory: Ortak hafıza instance
        
        Returns:
            {
                'adopted_learnings': [...],  # Benimsediği öğrenmeler
                'rejected_learnings': [...], # Reddettiği öğrenmeler
                'personal_insights': "...",  # Kendi yorumu
                'influence_weights': {...}   # Kimden ne kadar etkilendi
            }
        """
        temp = lora.temperament
        
        adopted = []
        rejected = []
        influence_weights = {}
        
        # 🏆 İTİBAR SİSTEMİ İMPORT
        from .reputation_system import reputation_system
        
        # Her LoRA'nın öğrenmelerini incele
        for other_id, other_data in others_learning.items():
            other_name = other_data['name']
            other_success = other_data['success_rate']
            other_learnings = other_data['learnings']
            other_temp = other_data['temperament']
            other_reputation = other_data.get('reputation', {})  # 🏆 İtibar!
            
            # 🏆 YAZAR KİM? (İtibar ne?)
            speaker_tier = other_reputation.get('tier', 'Sıradan')
            speaker_authority = other_reputation.get('authority_weight', 1.0)
            speaker_badges = other_reputation.get('badges', [])
            
            # Bu LoRA'dan ne kadar etkilenmeliyim? (İtibar bazlı!)
            influence = TemperamentBasedLearning._calculate_influence_with_reputation(
                lora, other_data, other_reputation
            )
            
            influence_weights[other_id] = influence
            
            # Öğrenmeleri değerlendir
            for learning in other_learnings[-3:]:  # Son 3 öğrenme
                # MİZAÇA + İTİBARA GÖRE KARAR VER!
                decision = TemperamentBasedLearning._decide_on_learning_with_reputation(
                    lora, learning, influence, other_success, other_reputation
                )
                
                if decision == 'ADOPT':
                    adopted.append({
                        'from': other_name,
                        'learning': learning,
                        'influence': influence,
                        'reason': TemperamentBasedLearning._get_adoption_reason(temp)
                    })
                elif decision == 'REJECT':
                    rejected.append({
                        'from': other_name,
                        'learning': learning,
                        'reason': TemperamentBasedLearning._get_rejection_reason(temp)
                    })
        
        # Kişisel yorum oluştur
        personal_insight = TemperamentBasedLearning._generate_personal_insight(
            lora, adopted, rejected, others_learning
        )
        
        return {
            'adopted_learnings': adopted,
            'rejected_learnings': rejected,
            'personal_insights': personal_insight,
            'influence_weights': influence_weights
        }
    
    @staticmethod
    def _calculate_influence(lora, other_data: Dict) -> float:
        """
        Bu LoRA'dan ne kadar etkilenmeliyim? (ESKİ - geriye uyumluluk)
        """
        return TemperamentBasedLearning._calculate_influence_with_reputation(
            lora, other_data, other_data.get('reputation', {})
        )
    
    @staticmethod
    def _calculate_influence_with_reputation(lora, other_data: Dict, other_reputation: Dict) -> float:
        """
        Bu LoRA'dan ne kadar etkilenmeliyim? (İTİBAR BAZLI! - AKIŞKAN!)
        
        Faktörler:
        - Başarı oranı (25%)
        - İtibar seviyesi (30%) ⭐ YENİ!
        - Sosyal zeka (20%)
        - Bağımsızlık (negatif, 15%)
        - Özel durumlar (Çifte uzman, vs.) (10%)
        """
        temp = lora.temperament
        other_success = other_data['success_rate']
        
        # 1) BAŞARI FAKTÖRÜ (25%)
        success_factor = other_success * 0.25
        
        # 2) İTİBAR FAKTÖRÜ (30%) ⭐ YENİ!
        reputation_value = other_reputation.get('total_reputation', 0.5)
        authority_weight = other_reputation.get('authority_weight', 1.0)
        
        # İtibar yüksek → daha çok dinle!
        reputation_factor = (reputation_value * 0.20) + ((authority_weight - 1.0) * 0.10)
        
        # 3) SOSYAL ZEKA (20%)
        social_factor = temp.get('social_intelligence', 0.5) * 0.20
        
        # 4) BAĞIMSIZLIK (negatif, 15%)
        independence_penalty = temp.get('independence', 0.5) * 0.15
        
        # 5) ÖZEL DURUMLAR (10%)
        special_bonus = 0.0
        speaker_badges = other_reputation.get('badges', [])
        
        # Çifte uzman → BAĞIMSIZ BİLE DURAKSAR!
        if "🏆 Çifte Uzman" in speaker_badges:
            special_bonus += 0.08
        
        # Mucize → Herkes dinler!
        if "🌟 Mucize" in speaker_badges:
            special_bonus += 0.10
        
        # Yaşlı bilge → Deneyim saygısı
        if "🧓 Yaşlı Bilge" in speaker_badges:
            special_bonus += 0.05
        
        # TOPLAM
        influence = success_factor + reputation_factor + social_factor - independence_penalty + special_bonus
        
        # 0-1 arası sınırla
        return max(0.0, min(1.0, influence))
    
    @staticmethod
    def _decide_on_learning(lora, learning: str, influence: float, other_success: float) -> str:
        """
        Bu öğrenmeyi benimsemeli mi? (ESKİ - geriye uyumluluk)
        """
        return TemperamentBasedLearning._decide_on_learning_with_reputation(
            lora, learning, influence, other_success, {}
        )
    
    @staticmethod
    def _decide_on_learning_with_reputation(lora, learning: str, influence: float, 
                                           other_success: float, other_reputation: Dict) -> str:
        """
        Bu öğrenmeyi benimsemeli mi? (İTİBAR BAZLI! - AKIŞKAN!)
        
        Kodlanmış karar YOK! Sadece formül!
        
        Returns:
            'ADOPT', 'REJECT', veya 'IGNORE'
        """
        temp = lora.temperament
        
        # KONUŞAN KİM?
        speaker_tier = other_reputation.get('tier', 'Sıradan')
        speaker_badges = other_reputation.get('badges', [])
        
        # ============================================
        # AKIŞKAN FORMÜL
        # ============================================
        
        # TEMEL DİNLEME OLASILIĞI
        base_listen_prob = (
            temp.get('social_intelligence', 0.5) * 0.35 +
            temp.get('herd_tendency', 0.5) * 0.25 +
            (1 - temp.get('independence', 0.5)) * 0.20 +
            (1 - temp.get('contrarian_score', 0.5)) * 0.20
        )
        
        # İTİBAR BOOST
        reputation_boost = 0.0
        
        if speaker_tier == "Efsane":
            reputation_boost = 0.35  # Efsane → +%35
        elif speaker_tier == "Usta":
            reputation_boost = 0.25  # Usta → +%25
        elif speaker_tier == "Uzman":
            reputation_boost = 0.15  # Uzman → +%15
        elif speaker_tier == "İyi":
            reputation_boost = 0.05
        
        # ÖZEL BADGE BOOST
        if "🏆 Çifte Uzman" in speaker_badges:
            # Çifte uzman → BAĞIMSIZ BİLE DURAKSAR!
            reputation_boost += 0.20
        
        if "🌟 Mucize" in speaker_badges:
            # Mucize → Herkes dinler!
            reputation_boost += 0.25
        
        # BAŞARI BOOST
        success_boost = other_success * 0.15
        
        # TOPLAM OLASIL IK
        final_probability = base_listen_prob + reputation_boost + success_boost
        final_probability = max(0.0, min(1.0, final_probability))
        
        # KARAR (Akışkan! Random ile)
        if random.random() < final_probability:
            return 'ADOPT'
        elif temp.get('contrarian_score', 0.5) > 0.6 and random.random() < 0.3:
            return 'REJECT'  # Karşıt bazen bilinçli reddeder
        else:
            return 'IGNORE'
    
    @staticmethod
    def _get_adoption_reason(temp: Dict) -> str:
        """Neden benimsedi?"""
        if temp.get('social_intelligence', 0) > 0.7:
            return "Sosyal zekanı yüksek, başarılılardan öğreniyorum"
        elif temp.get('herd_tendency', 0) > 0.7:
            return "Çoğunluk ne diyorsa doğrudur"
        else:
            return "Mantıklı geldi, deneyeceğim"
    
    @staticmethod
    def _get_rejection_reason(temp: Dict) -> str:
        """Neden reddetti?"""
        if temp.get('independence', 0) > 0.8:
            return "Bağımsızım, kendi yolumu giderim"
        elif temp.get('contrarian_score', 0) > 0.7:
            return "Çoğunluğa karşıyım, kendi düşüncem farklı"
        else:
            return "Bana uymadı"
    
    @staticmethod
    def _generate_personal_insight(lora, adopted: List, rejected: List, others_learning: Dict) -> str:
        """
        Kişisel yorum oluştur
        """
        temp = lora.temperament
        
        total_observed = len(adopted) + len(rejected)
        
        if total_observed == 0:
            return "Henüz başkalarından öğrenecek bir şey göremedim."
        
        # Mizaç bazlı yorum
        if temp.get('independence', 0) > 0.8:
            return f"{len(others_learning)} LoRA'nın deneyimini gördüm ama kendi yolumdan gideceğim."
        
        elif temp.get('social_intelligence', 0) > 0.7:
            return f"{len(adopted)} öğrenmeyi benimsedim, başarılılardan çok şey öğreniyorum!"
        
        elif temp.get('herd_tendency', 0) > 0.7:
            return f"Çoğunluğu takip ediyorum, {len(adopted)} öğrenmeyi kabul ettim."
        
        elif temp.get('contrarian_score', 0) > 0.7:
            return f"{len(rejected)} öğrenmeyi reddettim. Ben farklı düşünüyorum!"
        
        else:
            return f"{len(adopted)} öğrenmeyi benimsedim, {len(rejected)} reddettim. Dengeli yaklaşım."


# Global instance
temperament_learning = TemperamentBasedLearning()

