"""
🔬 KOLEKTİF ZEKA SİSTEMİ
========================

Newton bir formül buldu → Tüm insanlık kullandı
LoRA bir strateji keşfetti → Tüm LoRAlar kullanıyor

MATEMATİK:
- Keşif Tespiti: Accuracy + Improvement + Uniqueness
- Yayılım: Sosyal ağ üzerinden (yakın → uzak)
- İçselleştirme: Mizaca göre adoption rate
- Kümülatif: Her nesil öncekinin üzerine ekler
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os


class Discovery:
    """Bir LoRA'nın keşfi"""
    def __init__(self, discoverer_id: str, discoverer_name: str, 
                 pattern: str, formula, accuracy: float, match_idx: int):
        self.discoverer_id = discoverer_id
        self.discoverer_name = discoverer_name
        self.pattern = pattern
        self.formula = formula  # LoRA'nın öğrendiği ağırlıklar
        self.accuracy = accuracy
        self.match_idx = match_idx
        self.adoption_count = 0  # Kaç LoRA kabul etti
        self.generation = 1  # Kaçıncı nesil keşif


class CollectiveIntelligence:
    """
    Kolektif Zeka Motoru
    
    LoRAlar birbirinden öğrenir, keşifler yayılır,
    her nesil öncekinin üzerine ekler!
    """
    
    def __init__(self, log_dir: str = "evolution_logs"):
        self.discoveries = []  # Tüm keşifler
        self.pattern_experts = {}  # Pattern → En iyi LoRA
        self.collective_knowledge = {}  # Pattern → En iyi formula
        
        self.log_file = os.path.join(log_dir, "collective_intelligence.log")
        os.makedirs(log_dir, exist_ok=True)
        
        print("🔬 Collective Intelligence başlatıldı")
    
    def detect_discovery(self, lora, population: List, match_idx: int) -> Optional[Discovery]:
        """
        LoRA bir keşif yaptı mı?
        
        KRİTERLER:
        1. Accuracy > 0.8 (son 10 maçta)
        2. Improvement > 0.2 (önceki performansa göre)
        3. Uniqueness: Başka LoRAlar bu kadar başarılı değil
        """
        # En az 20 maç gerekli
        if len(lora.fitness_history) < 20:
            return None
        
        # 1. ACCURACY CHECK
        recent_10 = lora.fitness_history[-10:]
        recent_accuracy = sum(1 for f in recent_10 if f > 0.5) / 10
        
        if recent_accuracy < 0.8:
            return None  # Yeterince başarılı değil
        
        # 2. IMPROVEMENT CHECK
        previous_10 = lora.fitness_history[-20:-10]
        previous_accuracy = sum(1 for f in previous_10 if f > 0.5) / 10
        
        improvement = recent_accuracy - previous_accuracy
        
        if improvement < 0.2:
            return None  # Yeterli gelişme yok
        
        # 3. UNIQUENESS CHECK
        # Diğer LoRAların son 10 maç average'ı
        other_averages = []
        for other in population:
            if other.id == lora.id:
                continue
            if len(other.fitness_history) >= 10:
                other_recent = other.fitness_history[-10:]
                other_avg = sum(other_recent) / 10
                other_averages.append(other_avg)
        
        if not other_averages:
            return None
        
        # Bu LoRA en az top 3'te olmalı
        lora_avg = sum(recent_10) / 10
        top_3_threshold = sorted(other_averages, reverse=True)[min(2, len(other_averages)-1)]
        
        if lora_avg < top_3_threshold:
            return None  # Yeterince unique değil
        
        # ✅ KEŞİF YAPILDI!
        # Pattern belirle (specialization'dan veya dynamic olarak)
        pattern = getattr(lora, 'specialization', 'general_strategy')
        
        # Formula: LoRA'nın öğrendiği ağırlıklar
        formula = lora.get_all_lora_params()
        
        discovery = Discovery(
            discoverer_id=lora.id,
            discoverer_name=lora.name,
            pattern=pattern,
            formula=formula,
            accuracy=recent_accuracy,
            match_idx=match_idx
        )
        
        # Kaydet
        self.discoveries.append(discovery)
        
        # Expert registry güncelle
        if pattern not in self.pattern_experts:
            self.pattern_experts[pattern] = lora.id
            self.collective_knowledge[pattern] = formula
        else:
            # Mevcut expert'ten daha mı iyi?
            current_expert_id = self.pattern_experts[pattern]
            current_expert = next((l for l in population if l.id == current_expert_id), None)
            
            if current_expert:
                current_accuracy = sum(1 for f in current_expert.fitness_history[-10:] if f > 0.5) / 10
                if recent_accuracy > current_accuracy:
                    # Yeni expert!
                    self.pattern_experts[pattern] = lora.id
                    self.collective_knowledge[pattern] = formula
        
        # Log
        self._log_discovery(discovery)
        
        print(f"\n🔬 KEŞİF! {lora.name} → {pattern} (Accuracy: {recent_accuracy:.2f}, Improvement: {improvement:+.2f})")
        
        return discovery
    
    def broadcast_discovery(self, discovery: Discovery, population: List, social_network) -> int:
        """
        Keşfi topluluğa yay!
        
        YAYILIM: Sosyal ağ üzerinden
        - Keşfedici'ye yakın olanlar önce öğrenir
        - Uzak olanlar daha az etkiliyor
        - Mizaca göre kabul oranı değişir
        
        Returns:
            Kaç LoRA kabul etti
        """
        adopted_count = 0
        
        for lora in population:
            if lora.id == discovery.discoverer_id:
                continue  # Keşfedici zaten biliyor
            
            # Sosyal mesafe hesapla
            bond_strength = social_network.get_bond_strength(
                discovery.discoverer_id, lora.id
            )
            
            # Yakınlık faktörü
            # Güçlü bağ (0.7-1.0) → 1.0 proximity
            # Zayıf bağ (0.0-0.3) → 0.1 proximity
            proximity_factor = 0.1 + bond_strength * 0.9
            
            # Mizaç faktörü
            temperament_factor = self._calculate_adoption_rate(lora.temperament)
            
            # Kabul oranı
            adoption_rate = proximity_factor * temperament_factor
            
            # Keşfi kabul et
            if self._adopt_discovery(lora, discovery, adoption_rate):
                adopted_count += 1
        
        discovery.adoption_count = adopted_count
        
        print(f"   📡 {adopted_count} LoRA keşfi kabul etti")
        
        return adopted_count
    
    def _calculate_adoption_rate(self, temperament: Dict) -> float:
        """
        Mizaca göre kabul oranı
        
        Formül:
        adoption = openness × (1 - contrarian) × (1 - independence × 0.5)
        """
        openness = temperament.get('openness', 0.5)
        contrarian = temperament.get('contrarian_score', 0.5)
        independence = temperament.get('independence', 0.5)
        
        # Eğer openness yok ama benzer özellikler varsa
        if 'openness' not in temperament:
            # Risk tolerance ve curiosity'den hesapla
            risk = temperament.get('risk_tolerance', 0.5)
            hype = temperament.get('hype_sensitivity', 0.5)
            openness = (risk + hype) / 2
        
        adoption = openness * (1 - contrarian * 0.8) * (1 - independence * 0.5)
        
        return adoption
    
    def _adopt_discovery(self, lora, discovery: Discovery, adoption_rate: float) -> bool:
        """
        LoRA keşfi kabul eder mi ve uygular
        
        UYGULAMA: Parametre blending
        New_params = (1 - α) × Current + α × Discovery
        """
        import torch
        import random
        
        # Rastgele kabul (adoption_rate'e göre)
        if random.random() > adoption_rate:
            return False  # Kabul etmedi
        
        # Parametreleri blend et
        current_params = lora.get_all_lora_params()
        discovery_params = discovery.formula
        
        # Blend strength: adoption_rate × 0.3 (max %30 etki)
        blend_alpha = adoption_rate * 0.3
        
        for layer in ['fc1', 'fc2', 'fc3']:
            for matrix in ['lora_A', 'lora_B']:
                current = current_params[layer][matrix]
                discovered = discovery_params[layer][matrix]
                
                # Blend
                new_param = (1 - blend_alpha) * current + blend_alpha * discovered
                current_params[layer][matrix] = new_param
        
        # Geri yaz
        lora.set_all_lora_params(current_params)
        
        # Metadata ekle
        if not hasattr(lora, 'adopted_discoveries'):
            lora.adopted_discoveries = []
        
        lora.adopted_discoveries.append({
            'discovery': discovery.pattern,
            'from': discovery.discoverer_name,
            'match_idx': discovery.match_idx,
            'adoption_rate': adoption_rate,
            'blend_alpha': blend_alpha
        })
        
        return True
    
    def enable_cumulative_evolution(self, parent, child):
        """
        Kümülatif Evrim: Çocuk ebeveynin tüm keşiflerini miras alır
        
        Gen1: Einstein → Discovery A
        Gen2: Newton (Einstein'ın çocuğu) → Discovery A + B
        Gen3: Darwin (Newton'un çocuğu) → Discovery A + B + C
        """
        if not hasattr(parent, 'adopted_discoveries'):
            return
        
        # Ebeveynin tüm keşiflerini çocuğa aktar
        if not hasattr(child, 'adopted_discoveries'):
            child.adopted_discoveries = []
        
        # Inheritance: %50 strength ile aktar
        for disc in parent.adopted_discoveries:
            child.adopted_discoveries.append({
                'discovery': disc['discovery'],
                'from': disc['from'],
                'match_idx': disc['match_idx'],
                'adoption_rate': disc['adoption_rate'] * 0.5,  # Zayıflamış
                'blend_alpha': disc['blend_alpha'] * 0.5,
                'inherited_from_parent': parent.name
            })
        
        print(f"   🧬 {child.name} → {len(parent.adopted_discoveries)} keşfi miras aldı ({parent.name}'den)")
    
    def _log_discovery(self, discovery: Discovery):
        """Keşfi log dosyasına yaz"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"🔬 YENİ KEŞİF!\n")
                f.write(f"{'='*80}\n")
                f.write(f"📅 Maç: #{discovery.match_idx}\n")
                f.write(f"👨‍🔬 Keşfedici: {discovery.discoverer_name}\n")
                f.write(f"📊 Pattern: {discovery.pattern}\n")
                f.write(f"🎯 Accuracy: {discovery.accuracy:.2%}\n")
                f.write(f"🌍 Bu keşif topluluğa yayılacak!\n")
                f.write(f"{'='*80}\n\n")
        except Exception as e:
            print(f"⚠️ Discovery log yazılamadı: {e}")
    
    def get_community_knowledge_summary(self) -> Dict:
        """Topluluk bilgisinin özeti"""
        return {
            'total_discoveries': len(self.discoveries),
            'patterns_discovered': list(self.collective_knowledge.keys()),
            'expert_registry': {
                pattern: self.pattern_experts.get(pattern, 'N/A')
                for pattern in self.collective_knowledge.keys()
            }
        }


# Global instance
collective_intelligence = CollectiveIntelligence()
