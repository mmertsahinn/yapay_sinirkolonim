"""
🌍 EVRİMLEŞEN DOĞA SİSTEMİ (ADAPTIVE NATURE)
=============================================

Doğa da öğrenir, evrimleşir, adapte olur!

GÜNCELLEME (PARÇACIK FİZİĞİ!):
- NatureThermostat ile entegre!
- Sıcaklık (T) artık entropi bazlı!
- Fiziksel yasalarla tepki!

MANTIK:
- LoRA'lar depreme bağışık oldu mu? → Doğa yeni şey yapar!
- LoRA'lar çok güçlü mü? → Doğa zorlaşır!
- LoRA'lar zayıf mı? → Doğa yumuşar!
- Entropi düşük → Doğa ısınır! (Kaos artar!)

DOĞA VERSİYONLARI:
- V1: Klasik doğa (deprem, veba, vs.)
- V2: Evrimleşmiş (yeni tepkiler!)
- V3: İleri evrim (daha karmaşık!)
"""

import random
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class NatureVersion:
    """Doğa versiyonu"""
    version: int
    name: str
    unlocked_at_match: int
    capabilities: List[str]


class AdaptiveNatureSystem:
    """
    Evrimleşen, öğrenen doğa sistemi
    """
    
    def __init__(self):
        self.nature_version = 1  # Başlangıç: V1
        self.evolution_history = []
        
        # LoRA'ların bağışıklık seviyeleri
        self.lora_immunity = {}  # lora_id -> {'deprem': 0.5, 'veba': 0.2, ...}
        
        # Doğanın öğrenme hafızası
        self.nature_memory = {
            'attempted_events': [],  # Denenen olaylar
            'success_rates': {},     # Her olayın başarı oranı
            'lora_adaptations': []   # LoRA'ların adaptasyonları
        }
        
        # Mevcut yetenekler
        self.capabilities = {
            'v1': ['deprem', 'veba', 'stres_dalgası', 'sarsıntı'],
            'v2': ['psikolojik_saldırı', 'enerji_çekimi', 'zaman_bükülmesi'],
            'v3': ['kuantum_belirsizlik', 'kaos_dalgası', 'varoluşsal_kriz']
        }
        
        # 🌡️ NATURE THERMOSTAT ENTEGRASYONU!
        from lora_system.nature_thermostat import nature_thermostat
        self.thermostat = nature_thermostat
        
        print("🌍 Evrimleşen Doğa Sistemi başlatıldı (V1)")
        print("   🌡️ Nature Thermostat entegre edildi!")
    
    def observe_lora_immunity(self, population: List, event_type: str, success_rate: float):
        """
        Doğa gözlemler: LoRA'lar bağışık mı?
        
        Args:
            population: Mevcut popülasyon
            event_type: Denenen olay
            success_rate: Ne kadar başarılı? (ölüm oranı)
        """
        # Olayı kaydet
        self.nature_memory['attempted_events'].append({
            'event': event_type,
            'match': len(self.nature_memory['attempted_events']),
            'success_rate': success_rate,
            'population_size': len(population)
        })
        
        # Başarı oranını güncelle
        if event_type not in self.nature_memory['success_rates']:
            self.nature_memory['success_rates'][event_type] = []
        
        self.nature_memory['success_rates'][event_type].append(success_rate)
        
        # Son 5 denemede başarı oranı düşük mü?
        recent_rates = self.nature_memory['success_rates'][event_type][-5:]
        avg_recent = np.mean(recent_rates) if recent_rates else 1.0
        
        # %30'un altındaysa (LoRA'lar bağışık!)
        if avg_recent < 0.30 and len(recent_rates) >= 3:
            print(f"\n🌍 DOĞA FARK ETTİ: {event_type} artık etkisiz! (Başarı: %{avg_recent*100:.0f})")
            print(f"   💡 LoRA'lar bağışık oldu, yeni strateji gerekli!")
            return True  # Bağışıklık tespit edildi!
        
        return False
    
    def evolve_nature(self, population: List, match_count: int) -> Optional[str]:
        """
        Doğayı evrimleştir!
        
        LoRA'lar çok güçlüyse → Doğa V2'ye geçer!
        
        Returns:
            Yeni yetenekler mesajı veya None
        """
        # V1 → V2 koşulları
        if self.nature_version == 1 and match_count >= 300:
            # LoRA'lar klasik olaylara bağışık mı?
            immune_count = 0
            
            for event_type in ['deprem', 'veba']:
                if event_type in self.nature_memory['success_rates']:
                    recent = self.nature_memory['success_rates'][event_type][-5:]
                    if len(recent) >= 3 and np.mean(recent) < 0.30:
                        immune_count += 1
            
            if immune_count >= 2:
                # EVRİMLEŞ!
                self.nature_version = 2
                self.evolution_history.append({
                    'match': match_count,
                    'from_version': 1,
                    'to_version': 2,
                    'reason': 'LoRA\'lar klasik olaylara bağışık oldu'
                })
                
                return f"🌍🌍 DOĞA EVRİMLEŞTİ! V1 → V2\nYeni yetenekler: {', '.join(self.capabilities['v2'])}"
        
        # V2 → V3 koşulları
        elif self.nature_version == 2 and match_count >= 800:
            # V2 yetenekleri de etkisiz mi?
            v2_immune = 0
            
            for event_type in self.capabilities['v2']:
                if event_type in self.nature_memory['success_rates']:
                    recent = self.nature_memory['success_rates'][event_type][-3:]
                    if len(recent) >= 2 and np.mean(recent) < 0.25:
                        v2_immune += 1
            
            if v2_immune >= 2:
                self.nature_version = 3
                self.evolution_history.append({
                    'match': match_count,
                    'from_version': 2,
                    'to_version': 3,
                    'reason': 'LoRA\'lar V2 yeteneklerine de adapte oldu'
                })
                
                return f"🌍🌍🌍 DOĞA İLERİ EVRİM! V2 → V3\nYeni yetenekler: {', '.join(self.capabilities['v3'])}"
        
        return None
    
    def learn_optimal_thresholds(self, population: List, nature_state) -> Dict:
        """
        DOĞA KENDİ EŞİKLERİNİ ÖĞRENIR!
        
        LoRA'lar güçlüyse → Eşikler düşer (daha sert!)
        LoRA'lar zayıfsa → Eşikler yükselir (yumuşar!)
        
        Returns:
            Dinamik eşikler
        """
        avg_fitness = np.mean([lora.get_recent_fitness() for lora in population]) if population else 0.5
        avg_immunity = 0.0
        
        if len(self.lora_immunity) > 0:
            all_immunities = []
            for lora_id, immunities in self.lora_immunity.items():
                all_immunities.extend(list(immunities.values()))
            avg_immunity = np.mean(all_immunities) if all_immunities else 0.0
        
        population_size = len(population)
        
        # BASE THRESHOLDS
        base_health_critical = 0.20
        base_anger_high = 0.70
        
        # ADAPTATION FACTOR
        # Güçlü LoRA'lar → Eşikler düşer (doğa sertleşir!)
        strength_factor = (avg_fitness * 0.6) + (avg_immunity * 0.4)
        
        # Population factor
        # Kalabalık → Eşikler düşer (doğa daha agresif!)
        population_factor = min(population_size / 200.0, 1.0)
        
        # TOTAL ADAPTATION
        adaptation = (strength_factor * 0.7) + (population_factor * 0.3)
        
        # DİNAMİK THRESHOLDS
        dynamic_health_critical = base_health_critical * (1.0 + adaptation * 0.5)  # Güçlü LoRA → 0.30'a çıkar
        dynamic_anger_high = base_anger_high * (1.0 - adaptation * 0.3)  # Güçlü LoRA → 0.49'a düşer
        
        # Sınırla
        dynamic_health_critical = max(0.10, min(0.40, dynamic_health_critical))
        dynamic_anger_high = max(0.50, min(0.85, dynamic_anger_high))
        
        return {
            'health_critical': dynamic_health_critical,
            'anger_high': dynamic_anger_high,
            'adaptation_level': adaptation,
            'reason': f"LoRA gücü: {strength_factor:.2f}, Nüfus: {population_size}"
        }
    
    def select_adaptive_response(self, population: List, nature_state, match_count: int) -> Optional[Dict]:
        """
        Akışkan doğa tepkisi seç!
        
        SABİT FORMÜL YOK!
        Doğa öğrenir, adapte olur, evrimleşir!
        """
        # Mevcut versiyon yetenekleri
        available_events = self.capabilities[f'v{self.nature_version}']
        
        # LoRA'ların bağışıklık seviyelerine bak
        event_effectiveness = {}
        
        for event_type in available_events:
            # Bu olay ne kadar etkili?
            if event_type in self.nature_memory['success_rates']:
                recent = self.nature_memory['success_rates'][event_type][-5:]
                effectiveness = np.mean(recent) if recent else 0.5
            else:
                effectiveness = 0.7  # Yeni olay, varsayılan etkili
            
            event_effectiveness[event_type] = effectiveness
        
        # En etkili olayı seç (LoRA'lar bağışık olmayan!)
        best_event = max(event_effectiveness, key=event_effectiveness.get)
        best_effectiveness = event_effectiveness[best_event]
        
        # Çok etkisiz olayları filtrele
        if best_effectiveness < 0.20:
            # Hiçbir olay etkili değil → EVRİMLEŞ!
            evolution_msg = self.evolve_nature(population, match_count)
            if evolution_msg:
                print(evolution_msg)
                # Yeni versiyondan seç
                available_events = self.capabilities[f'v{self.nature_version}']
                best_event = random.choice(available_events)
        
        return best_event
    
    def calculate_adaptive_severity(self, population: List, event_type: str, 
                                    base_severity: float) -> float:
        """
        Akışkan severity (ağırlık) hesapla!
        
        LoRA'lar güçlüyse → Daha sert!
        LoRA'lar zayıfsa → Daha yumuşak!
        """
        # Popülasyon gücü
        avg_fitness = np.mean([lora.get_recent_fitness() for lora in population])
        
        # Bağışıklık seviyesi
        immunity_levels = []
        for lora in population:
            lora_immunity = self.lora_immunity.get(lora.id, {})
            event_immunity = lora_immunity.get(event_type, 0.0)
            immunity_levels.append(event_immunity)
        
        avg_immunity = np.mean(immunity_levels) if immunity_levels else 0.0
        
        # ADAPTASYON FORMÜLÜ
        # Güçlü LoRA + Yüksek bağışıklık → Daha sert tepki!
        adaptation_factor = 1.0 + (avg_fitness * 0.5) + (avg_immunity * 0.3)
        
        adaptive_severity = base_severity * adaptation_factor
        
        # 0-1 arası sınırla
        return min(1.0, adaptive_severity)
    
    def lora_survived_event(self, lora, event_type: str, survived_by: str = "luck"):
        """
        LoRA bir olaydan kurtuldu!
        
        Args:
            lora: LoRA instance
            event_type: Olay tipi
            survived_by: 'luck', 'armor', 'adaptation'
        """
        # Bağışıklık kazandır!
        if lora.id not in self.lora_immunity:
            self.lora_immunity[lora.id] = {}
        
        # Bu olay için bağışıklık artır
        current_immunity = self.lora_immunity[lora.id].get(event_type, 0.0)
        
        # Bağışıklık artışı (nasıl kurtulduğuna göre)
        if survived_by == "adaptation":
            immunity_gain = 0.15  # Adaptasyon ile → en yüksek
        elif survived_by == "armor":
            immunity_gain = 0.10  # Zırh ile → orta
        else:  # luck
            immunity_gain = 0.05  # Şans ile → düşük
        
        new_immunity = min(1.0, current_immunity + immunity_gain)
        self.lora_immunity[lora.id][event_type] = new_immunity
        
        # Hafızaya kaydet
        self.nature_memory['lora_adaptations'].append({
            'lora_id': lora.id,
            'event': event_type,
            'survived_by': survived_by,
            'immunity_before': current_immunity,
            'immunity_after': new_immunity
        })


# Global instance
adaptive_nature = AdaptiveNatureSystem()

