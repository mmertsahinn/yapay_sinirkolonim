"""
🌍 DOĞA + ENTROPİ SİSTEMİ
==========================

LoRA'lar doğaya zarar verir (hata yaparak)
Doğa geri vurur (Kara Veba, Deprem, Kaos)

Entropi: Her şey zamanla dağılır, soğur, unutulur
"""

import numpy as np
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TraumaEvent:
    """LoRA travma olayı"""
    type: str              # 'survivor_guilt', 'kara_veba', 'pattern_failure', vs
    severity: float        # 0-1 arası şiddet
    timestamp: int         # Hangi maçta oldu
    description: str = ""  # Opsiyonel açıklama


@dataclass
class NatureState:
    """Doğanın durumu"""
    health: float = 1.0              # 0-1 arası, 0 = çok kötü, 1 = çok iyi
    anger: float = 0.0               # 0-1 arası, 1 = çok öfkeli
    chaos_index: float = 0.0         # 0-1 arası, kaos seviyesi
    entropy_rate: float = 0.01       # Entropi hızı (her maç)
    
    # Kara Veba riski (doğanın öfkesine bağlı)
    kara_veba_base_prob: float = 0.0001  # Temel olasılık
    
    # İstatistikler
    total_lora_mistakes: int = 0
    total_lora_success: int = 0
    last_kara_veba_match: int = 0


class NatureEntropySystem:
    """
    Doğa + Entropi Yöneticisi
    
    LoRA'lar hata yaptıkça doğaya zarar verirler.
    Doğa belli bir noktadan sonra geri vurur.
    Her şey zamanla soğur (entropi).
    """
    
    def __init__(self):
        self.nature = NatureState()
        self.match_count = 0
        self.event_history = []
        
        # Entropi parametreleri
        self.attraction_decay_rate = 0.998    # Her maç %0.2 azalma
        self.memory_decay_rate = 0.995        # Hafıza azalması
        self.goal_enthusiasm_decay = 0.999    # Hedef hevesi azalması
        
        print("🌍 Doğa + Entropi Sistemi başlatıldı")
        print(f"   Doğa Sağlığı: {self.nature.health:.2f}")
        print(f"   Entropi Hızı: {self.nature.entropy_rate:.4f}")
    
    def lora_made_mistake(self, severity: float = 0.1, population_size: int = 20):
        """
        LoRA hata yaptı → Doğaya zarar
        
        severity: 0-1 arası, hatanın ağırlığı
        population_size: Mevcut LoRA sayısı (çok fazlaysa zarar artar!)
        """
        self.nature.total_lora_mistakes += 1
        
        # 🌊 AKIŞKAN NÜFUS ZARARI (Sabit 50 YOK!)
        # Zarar, nüfusun "beklenen seviye"ye oranına göre
        # Beklenen seviye: dynamic_population_threshold (natural_triggers'dan)
        expected_population = getattr(self, 'dynamic_population_threshold', 100)
        
        if population_size > expected_population:
            # Eşiği aşan her LoRA için ekstra zarar
            overpopulation_multiplier = 1.0 + ((population_size - expected_population) / expected_population) * 0.5
        else:
            overpopulation_multiplier = 1.0
        
        # Doğanın sağlığı azalır
        damage = severity * 0.02 * overpopulation_multiplier
        self.nature.health = max(0.0, self.nature.health - damage)
        
        # Doğanın öfkesi artar (nüfus fazlası ise çok daha fazla!)
        anger_increase = damage * 2 * overpopulation_multiplier
        self.nature.anger = min(1.0, self.nature.anger + anger_increase)
        
        # Kaos seviyesi artar
        self.nature.chaos_index = min(1.0, self.nature.chaos_index + damage * 1.5)
    
    def lora_succeeded(self, quality: float = 0.1, population_size: int = 20):
        """
        LoRA başarılı oldu → Doğayı iyileştirir
        
        Args:
            quality: 0-1 arası, başarının kalitesi
            population_size: Koloni büyüklüğü
        
        KOLONİ MANTIĞI:
        - Küçük koloni (< 10): Az etki (×0.5) - henüz öğreniyor
        - Orta koloni (10-50): Normal etki (×1.0)
        - Büyük koloni (> 50): Büyük etki (×1.5) - güçlü koloni!
        """
        self.nature.total_lora_success += 1
        
        # Popülasyon çarpanı
        # 🌊 AKIŞKAN POPÜLASYON ETKİSİ!
        expected_population = getattr(self, 'dynamic_population_threshold', 100)
        
        if population_size < expected_population * 0.2:  # Çok küçük
            pop_multiplier = 0.5
        elif population_size > expected_population:  # Eşiği aştı
            pop_multiplier = 1.5
        else:
            pop_multiplier = 1.0  # Normal etki
        
        # Doğa iyileşir (ama yavaş) - popülasyona göre ayarlı
        healing = quality * 0.01 * pop_multiplier
        self.nature.health = min(1.0, self.nature.health + healing)
        
        # Öfke azalır (ama çok yavaş)
        self.nature.anger = max(0.0, self.nature.anger - healing * 0.5)
        
        # Kaos azalır
        self.nature.chaos_index = max(0.0, self.nature.chaos_index - healing * 0.3)
    
    def check_nature_response(self, population_size: int = 20, adaptive_nature=None) -> Optional[Dict]:
        """
        🌍 ÖĞRENEN DOĞA: Zarar bazlı deterministik karar!
        
        Mantık:
        - Zarar YOKSA → Doğa hiçbir şey yapmaz (olasılık yok!)
        - Zarar VARSA → AdaptiveNature'ın öğrenen ağırlıklarına göre karar verir
        
        Args:
            population_size: Mevcut LoRA sayısı
            adaptive_nature: AdaptiveNature instance (öğrenen doğa)
        
        Returns:
            None (zarar yoksa) veya olay dict'i (zarar varsa + AdaptiveNature kararı)
        """
        self.match_count += 1
        
        # 1) ZARAR SEVİYESİNİ HESAPLA (Deterministik!)
        damage_level = self._calculate_damage_level()
        
        # 2) ZARAR YOKSA → HİÇBİR ŞEY YAPMA!
        if damage_level <= 0.0:
            return None  # Doğa zarar görmemiş, tepki vermez!
        
        # 3) ZARAR VARSA → ADAPTIVE NATURE KARAR VERİR (Öğrenen!)
        if adaptive_nature is None:
            # AdaptiveNature yoksa eski sisteme dön (fallback)
            return self._fallback_probability_based_response(population_size)
        
        # AdaptiveNature'ın state'ini senkronize et
        adaptive_nature.state['anger'] = self.nature.anger
        adaptive_nature.state['chaos'] = self.nature.chaos_index
        adaptive_nature.state['health'] = self.nature.health
        
        # Öğrenen doğa karar verir (mercy, minor_disaster, major_disaster, resource_boom)
        action = adaptive_nature.decide_nature_action()
        
        # 4) KARARA GÖRE FELAKET TETİKLE
        if action == 'mercy':
            # Merhamet → Hiçbir şey yapma veya çok küçük uyarı
            if damage_level > 0.7:  # Çok yüksek zarar varsa bile merhamet göstermez
                return self._trigger_mini_tremor()  # Sadece küçük uyarı
            return None  # Zarar düşükse hiçbir şey yapma
        
        elif action == 'minor_disaster':
            # Küçük felaket → Deprem veya Mini Tremor
            if self.nature.chaos_index > 0.5:
                return self._trigger_quake()
            else:
                return self._trigger_mini_tremor()
        
        elif action == 'major_disaster':
            # Büyük felaket → Kara Veba (sadece çok yüksek zararda!)
            if damage_level > 0.6 and self.nature.anger > 0.7:
                return self._trigger_kara_veba()
            else:
                # Zarar yüksek ama henüz Kara Veba seviyesinde değil → Deprem
                return self._trigger_quake()
        
        elif action == 'resource_boom':
            # Kaynak patlaması → Doğa iyileşir, felaket yok!
            # (Bu durumda zarar azalır, felaket tetiklenmez)
            return None
        
        # Fallback (olması gerekmez ama güvenlik için)
        return None
    
    def _calculate_damage_level(self) -> float:
        """
        Zarar seviyesini hesapla (0.0 - 1.0)
        
        Zarar = Öfke + (1 - Sağlık) + Hata oranı
        """
        # Öfke bileşeni (0-1)
        anger_component = self.nature.anger
        
        # Sağlık bileşeni (sağlık düşükse zarar yüksek)
        health_component = 1.0 - self.nature.health
        
        # Hata oranı (toplam hata / (hata + başarı))
        total_events = self.nature.total_lora_mistakes + self.nature.total_lora_success
        if total_events > 0:
            mistake_ratio = self.nature.total_lora_mistakes / total_events
        else:
            mistake_ratio = 0.0
        
        # Ağırlıklı toplam
        damage_level = (
            anger_component * 0.4 +      # Öfke %40
            health_component * 0.3 +     # Sağlık %30
            mistake_ratio * 0.3           # Hata oranı %30
        )
        
        return min(1.0, max(0.0, damage_level))
    
    def _fallback_probability_based_response(self, population_size: int) -> Optional[Dict]:
        """
        Fallback: AdaptiveNature yoksa eski olasılık bazlı sistemi kullan
        (Geçici çözüm, idealde AdaptiveNature her zaman olmalı)
        """
        # Eski sistem (sadece fallback için)
        if self.nature.anger > 0.8 and self.nature.health < 0.3:
            return self._trigger_kara_veba()
        elif self.nature.chaos_index > 0.6:
            return self._trigger_quake()
        elif self.nature.anger > 0.3:
            return self._trigger_mini_tremor()
        
        return None
    
    def _calculate_kara_veba_probability(self, population_size: int = 20) -> float:
        """
        Kara Veba olasılığı hesapla
        
        Ne kadar çok hata → O kadar yüksek risk
        Ne kadar çok nüfus + başarısız → O kadar yüksek risk
        Ama yine de RASTGELE!
        """
        # Temel olasılık
        base_prob = self.nature.kara_veba_base_prob
        
        # Doğanın öfkesi riski artırır
        anger_multiplier = 1 + self.nature.anger * 50  # Max 51x
        
        # Doğanın sağlığı kötüyse risk artar
        health_multiplier = 1.0 / max(0.1, self.nature.health)  # Sağlık 0.1 → 10x
        
        # 🌊 AKIŞKAN NÜFUS RİSKİ!
        expected_population = getattr(self, 'dynamic_population_threshold', 100)
        
        if population_size > expected_population:
            # Eşiği aşan her LoRA için ekstra risk
            overpopulation_multiplier = 1.0 + ((population_size - expected_population) / expected_population) * 0.5
        else:
            overpopulation_multiplier = 1.0
        
        # Son Kara Veba'dan bu yana geçen süre (uzun süre geçtiyse risk azalır)
        matches_since_last = self.match_count - self.nature.last_kara_veba_match
        time_factor = np.exp(-matches_since_last / 500)  # 500 maç sonra %37
        
        # TOPLAM OLASILIK
        total_prob = (base_prob * anger_multiplier * health_multiplier * 
                     overpopulation_multiplier * (1 + time_factor))
        
        # Max %10 (çok sık olmasın ama nüfus patlamasında yüksek olabilir)
        return min(total_prob, 0.10)
    
    def _trigger_kara_veba(self) -> Dict:
        """☠️ KARA VEBA: Kitlesel ölüm, fitness önemsiz!"""
        self.nature.last_kara_veba_match = self.match_count
        
        # Doğa biraz sakinleşir (öfkesini kustu)
        self.nature.anger *= 0.3
        
        # Ama sağlık da daha kötü (herkes zarar gördü)
        self.nature.health *= 0.5
        
        event = {
            'type': 'kara_veba',
            'match': self.match_count,
            'severity': 0.95,
            'survival_rate': 0.20,  # %20 hayatta kalma
            'message': '☠️ KARA VEBA! Doğa geri vurdu! Fitness bir şey ifade etmiyor!'
        }
        
        self.event_history.append(event)
        return event
    
    def _trigger_quake(self) -> Dict:
        """🌍 DEPREM: Sosyal bağlar, çekimler sarsılır"""
        severity = self.nature.chaos_index * random.uniform(0.5, 1.0)
        
        event = {
            'type': 'quake',
            'match': self.match_count,
            'severity': severity,
            'affected_ratio': severity * 0.6,  # %60'a kadar LoRA etkilenir
            'message': f'🌍 DEPREM! Şiddet: {severity:.2f}, Sosyal bağlar sarsıldı!'
        }
        
        # Kaos biraz azalır (enerji boşaldı)
        self.nature.chaos_index *= 0.7
        
        self.event_history.append(event)
        return event
    
    def _trigger_mini_tremor(self) -> Dict:
        """⚡ MİNİ SALLANTI: Küçük gürültü, sürekli olan"""
        severity = random.uniform(0.05, 0.15)
        
        event = {
            'type': 'mini_tremor',
            'match': self.match_count,
            'severity': severity,
            'affected_ratio': random.uniform(0.05, 0.15),  # %5-15 LoRA
            'message': f'⚡ Mini sallantı (şiddet: {severity:.2f})'
        }
        
        self.event_history.append(event)
        return event
    
    def _trigger_overpopulation_purge_OLD(self, population_size: int) -> Dict:
        """
        ESKİ METOD - ARTIK KULLANILMIYOR!
        natural_triggers.py'deki akışkan sistem kullanılıyor!
        """
        """
        🌊 NÜFUS PATLAMASI CEZASI
        
        Çok fazla LoRA + başarısız → Doğa zorla öldürür!
        Fitness'e bakmaz, rastgele kitle ölümü!
        """
        # Ne kadar fazla nüfus?
        excess = population_size - 80
        kill_ratio = min(0.4, excess * 0.01)  # Max %40 ölüm
        
        event = {
            'type': 'overpopulation_purge',
            'match': self.match_count,
            'severity': 0.8,
            'kill_ratio': kill_ratio,
            'population_size': population_size,
            'message': f'🌊 NÜFUS PATLAMASI! Doğa {kill_ratio*100:.0f}% LoRA\'yı öldürüyor! (Nüfus: {population_size})'
        }
        
        # Doğa biraz rahatlar (nüfus azaltıldı)
        self.nature.anger *= 0.6
        self.nature.chaos_index *= 0.7
        
        # Ama sağlık da biraz azalır (katliam oldu)
        self.nature.health *= 0.8
        
        self.event_history.append(event)
        return event
    
    def apply_entropy(self, lora_population: List) -> Dict:
        """
        ENTROPİ: Her şey zamanla soğur, dağılır, unutulur
        
        Her maçta çağrılır.
        """
        entropy_effects = {
            'attractions_decayed': 0,
            'bonds_broken': 0,
            'goals_lost_enthusiasm': 0,
            'memories_faded': 0
        }
        
        for lora in lora_population:
            # 1) ENTROPİ: Pattern çekimleri azalır (SOĞUMA)
            # Her maç pattern_attractions %0.2 azalır (attraction_decay_rate = 0.998)
            # Zamanla LoRA'lar belirli pattern'lere olan ilgilerini kaybeder
            if hasattr(lora, 'pattern_attractions') and lora.pattern_attractions:
                for pattern in lora.pattern_attractions:
                    old_value = lora.pattern_attractions[pattern]
                    lora.pattern_attractions[pattern] *= self.attraction_decay_rate
                    
                    # Eşik altına düştüyse kayıt et
                    if old_value > 0.1 and lora.pattern_attractions[pattern] < 0.1:
                        entropy_effects['attractions_decayed'] += 1
            
            # 2) ENTROPİ: Sosyal bağlar zayıflar (SOĞUMA)
            # Her maç sosyal bağlar %0.2 azalır
            # Zamanla LoRA'lar arası ilişkiler zayıflar, bazıları kopar
            if hasattr(lora, 'social_bonds') and lora.social_bonds:
                bonds_to_remove = []
                for other_lora_id, bond_strength in lora.social_bonds.items():
                    # Her maç %0.2 azalma (attraction_decay_rate = 0.998)
                    new_strength = bond_strength * self.attraction_decay_rate
                    lora.social_bonds[other_lora_id] = new_strength
                    
                    # Çok zayıfladıysa (0.05 altı) bağ kırılır
                    if new_strength < 0.05:
                        bonds_to_remove.append(other_lora_id)
                        entropy_effects['bonds_broken'] += 1
                
                # Kırılan bağları temizle
                for bond_id in bonds_to_remove:
                    del lora.social_bonds[bond_id]
            
            # 3) ENTROPİ: Hedef hevesi azalır (SOĞUMA)
            # Her maç main_goal.heves %0.1 azalır (goal_enthusiasm_decay = 0.999)
            # Zamanla LoRA'lar hedeflerine olan bağlılıklarını kaybeder
            if hasattr(lora, 'main_goal') and lora.main_goal:
                old_heves = lora.main_goal.heves
                lora.main_goal.heves *= self.goal_enthusiasm_decay
                
                # Heves 0.3'ün altına düştüyse kayıt et
                if old_heves > 0.3 and lora.main_goal.heves < 0.3:
                    entropy_effects['goals_lost_enthusiasm'] += 1
            
            # 4) ENTROPİ: Hafıza (travma) soluklaşır (SOĞUMA)
            # Her maç travma severity'si %0.5 azalır (memory_decay_rate = 0.995)
            # Zamanla travmatik anılar unutulur, etkileri azalır
            for trauma in lora.trauma_history:
                # Trauma hem dict hem TraumaEvent objesi olabilir
                if isinstance(trauma, dict):
                    trauma['severity'] *= self.memory_decay_rate
                    if trauma['severity'] < 0.1:
                        entropy_effects['memories_faded'] += 1
                else:
                    # TraumaEvent objesi
                    trauma.severity *= self.memory_decay_rate
                    if trauma.severity < 0.1:
                        entropy_effects['memories_faded'] += 1
        
        return entropy_effects
    
    def get_nature_status(self, population_size: int = 20) -> Dict:
        """Doğanın durumunu döndür"""
        kara_veba_prob = self._calculate_kara_veba_probability(population_size)
        
        # 🌊 AKIŞKAN NÜFUS RİSKİ!
        expected_population = getattr(self, 'dynamic_population_threshold', 100)
        overpopulation_risk = 0.0
        if population_size > expected_population:
            overpopulation_risk = ((population_size - expected_population) / expected_population) * 0.02
        
        return {
            'match': self.match_count,
            'health': self.nature.health,
            'anger': self.nature.anger,
            'chaos': self.nature.chaos_index,
            'population_size': population_size,
            'overpopulation_risk': overpopulation_risk,
            'kara_veba_probability': kara_veba_prob,
            'total_mistakes': self.nature.total_lora_mistakes,
            'total_success': self.nature.total_lora_success,
            'success_ratio': (
                self.nature.total_lora_success / 
                max(1, self.nature.total_lora_mistakes + self.nature.total_lora_success)
            )
        }
    
    def print_nature_status(self, population_size: int = 20):
        """Doğanın durumunu yazdır"""
        status = self.get_nature_status(population_size)
        
        # Doğa sağlığı emoji
        if status['health'] > 0.8:
            health_emoji = "💚"
        elif status['health'] > 0.5:
            health_emoji = "💛"
        else:
            health_emoji = "❤️"
        
        # Öfke emoji
        if status['anger'] > 0.7:
            anger_emoji = "😡"
        elif status['anger'] > 0.4:
            anger_emoji = "😠"
        else:
            anger_emoji = "😐"
        
        # 🌊 AKIŞKAN NÜFUS EMOJİ!
        expected_population = getattr(self, 'dynamic_population_threshold', 100)
        
        if population_size > expected_population * 1.5:
            pop_emoji = "🚨"  # Tehlike!
        elif population_size > expected_population:
            pop_emoji = "⚠️"  # Dikkat
        else:
            pop_emoji = "👥"  # Normal
        
        print(f"\n{'='*70}")
        print(f"🌍 DOĞANIN DURUMU (Maç #{status['match']})")
        print(f"{'='*70}")
        print(f"  {health_emoji} Sağlık: {status['health']:.3f}")
        print(f"  {anger_emoji} Öfke: {status['anger']:.3f}")
        print(f"  🌪️ Kaos: {status['chaos']:.3f}")
        print(f"  {pop_emoji} Nüfus: {population_size} LoRA")
        if status['overpopulation_risk'] > 0:
            print(f"  🌊 Nüfus Patlaması Riski: {status['overpopulation_risk']*100:.1f}%")
        print(f"  ☠️ Kara Veba Riski: {status['kara_veba_probability']*100:.4f}%")
        print(f"  ✅ Başarı Oranı: {status['success_ratio']*100:.1f}%")
        print(f"{'='*70}\n")


class GoallessDriftSystem:
    """
    HEDEFSİZ SÜRÜKLENME
    
    Hedefsiz LoRA'lar:
    - Bilinçsizce çekimlere kapılır
    - Sosyal bağlara sürüklenir
    - Ya görevine ulaşır ya da yolda ölür
    """
    
    @staticmethod
    def update_goalless_lora(lora, all_loras, current_match: int = None):
        """
        Hedefsiz LoRA'yı güncelle
        
        YAŞ SİSTEMİ (10 maç = 1 yaş):
        - 0-10 maç (0-1 yaş): BEBEK 👶 - Hedefsizlik normal! Risk yok!
        - 10-100 maç (1-10 yaş): GENÇ 🧒 - Hedef seçebilir, hafif risk
        - 100+ maç (10+ yaş): YETİŞKİN 🧑 - Hedefsizlik tehlikeli!
        
        Çekimlere göre savrulur!
        """
        # ✅ YAŞ HESAPLA (MAÇ BAZLI! - Bilimsel standart!)
        age_in_matches = current_match - lora.birth_match if current_match else 0
        age_in_years = age_in_matches / 10.0  # 10 maç = 1 yaş
        
        # 👶 BEBEKLİK DÖNEMİ (0-100 maç)
        if age_in_matches < 100:
            # Bebeklik, hedefsiz olması normal!
            # Risk yok, öğreniyor! Dünyayı keşfediyor!
            # Hedefsizlik drift riski SIFIR!
            lora.goalless_death_risk = 0.0
            return
        
        # 🧒 GENÇLİK DÖNEMİ (100-180 maç)
        # Hedef seçme yaşı! Ama zorunlu değil
        if age_in_matches < 180:
            # Hedef seçmeye başlamalı ama stres yok
            search_intensity = 0.3  # Hafif arama
            risk_multiplier = 0.5   # Düşük risk
        
        # 🧑 YETİŞKİNLİK DÖNEMİ (180-250 maç)
        # Hedef olmalı! Yoksa savrulur!
        elif age_in_matches < 250:
            # Hedef araması agresif
            search_intensity = 0.6  # Yüksek arama
            risk_multiplier = 1.0   # Normal risk
        
        # 👴 OLGUNLUK DÖNEMİ (250+ maç)
        # Hedef zorunlu! Yoksa çok tehlikeli!
        else:
            # Hedef araması çok agresif
            search_intensity = 0.8  # Çok yüksek arama
            risk_multiplier = 2.0   # Yüksek risk!
        
        if not hasattr(lora, 'main_goal') or lora.main_goal is None:
            # HEDEFSİZ MOD (artık 10+ yaş!)
            
            # 1) En güçlü sosyal çekime sürüklenir
            if len(lora.social_bonds) > 0:
                strongest_bond_id = max(lora.social_bonds, key=lora.social_bonds.get)
                strongest_lora = next((l for l in all_loras if l.id == strongest_bond_id), None)
                
                if strongest_lora and hasattr(strongest_lora, 'main_goal') and strongest_lora.main_goal:
                    # O LoRA'nın hedefine bilinçsizce çekilir!
                    drift_strength = lora.social_bonds[strongest_bond_id]
                    
                    # Şans: Yaşa göre değişir
                    if random.random() < drift_strength * search_intensity:
                        lora.main_goal = strongest_lora.main_goal  # Aynı hedefi kopyalar!
                        
                        # Yaş etiketi
                        if age_in_years < 18.0:
                            age_tag = "🧒 Genç"
                        elif age_in_years < 25.0:
                            age_tag = "🧑 Yetişkin"
                        else:
                            age_tag = "👴 Olgun"
                        
                        print(f"  🌊 {lora.name} ({age_tag}, {age_in_years:.1f} yaş) hedefsizken {strongest_lora.name}'in hedefine sürüklendi!")
            
            # 2) En güçlü pattern çekimine sürüklenir
            if hasattr(lora, 'pattern_attractions') and lora.pattern_attractions:  # ✅ Boş değilse
                strongest_pattern = max(lora.pattern_attractions, key=lora.pattern_attractions.get)
                attraction_strength = lora.pattern_attractions[strongest_pattern]
                
                # Şans: Yaşa göre değişir
                if random.random() < attraction_strength * search_intensity:
                    from .nature_entropy_system import Goal
                    lora.main_goal = Goal(
                        type='pattern_mastery',
                        target_pattern=strongest_pattern,
                        priority='main',
                        patience=300
                    )
                    
                    # Yaş etiketi
                    if age_in_years < 18.0:
                        age_tag = "🧒 Genç"
                    elif age_in_years < 25.0:
                        age_tag = "🧑 Yetişkin"
                    else:
                        age_tag = "👴 Olgun"
                    
                    print(f"  🎯 {lora.name} ({age_tag}, {age_in_years:.1f} yaş) hedefsizken {strongest_pattern} pattern'ine sürüklendi!")
            
            # 3) Hiçbir şey yoksa: RASTGELE SÜRÜKLENME (tehlikeli!)
            if lora.main_goal is None:
                # Rastgele bir LoRA'ya çekilir
                if len(all_loras) > 1:
                    random_lora = random.choice([l for l in all_loras if l.id != lora.id])
                    drift_bond = random.uniform(0.3, 0.7)
                    lora.social_bonds[random_lora.id] = drift_bond
                    
                    # Yaş etiketi
                    if age_in_years < 18.0:
                        age_tag = "🧒 Genç"
                    elif age_in_years < 25.0:
                        age_tag = "🧑 Yetişkin"
                    else:
                        age_tag = "👴 Olgun"
                    
                    print(f"  🌀 {lora.name} ({age_tag}, {age_in_years:.1f} yaş) hedefsizken rastgele {random_lora.name}'e sürüklendi!")
            
            # 4) Hedefsiz olmak streslidir (YAŞ'A GÖRE!)
            if hasattr(lora, 'temperament'):
                if 'stress_tolerance' in lora.temperament:
                    # Stres azalması (yaşa göre)
                    if age_in_years < 18.0:
                        # 🧒 GENÇ (10-18 yaş): Hafif stres
                        lora.temperament['stress_tolerance'] *= 0.998  # Her maç %0.2 azalır
                    elif age_in_years < 25.0:
                        # 🧑 YETİŞKİN (18-25 yaş): Orta stres
                        lora.temperament['stress_tolerance'] *= 0.995  # Her maç %0.5 azalır
                    else:
                        # 👴 OLGUN (25+ yaş): Ağır stres!
                        lora.temperament['stress_tolerance'] *= 0.98   # Her maç %2 azalır
            
            # 5) Ölüm riski artar (YAŞ'A GÖRE!)
            lora.goalless_death_risk = getattr(lora, 'goalless_death_risk', 0.0)
            
            # Risk multiplier zaten belirlendi (yaş'a göre)
            base_risk_increase = 0.001  # Temel artış
            actual_risk_increase = base_risk_increase * risk_multiplier
            lora.goalless_death_risk += actual_risk_increase
            
            # Risk eşikleri (YAŞ'A GÖRE!)
            if age_in_years < 18.0:
                # 🧒 GENÇ (10-18 yaş): %5 risk normal
                risk_threshold = 0.05
                age_tag = "🧒 Genç"
            elif age_in_years < 25.0:
                # 🧑 YETİŞKİN (18-25 yaş): %10 risk alarm
                risk_threshold = 0.10
                age_tag = "🧑 Yetişkin"
            else:
                # 👴 OLGUN (25+ yaş): %15 risk çok tehlikeli!
                risk_threshold = 0.15
                age_tag = "👴 Olgun"
            
            if lora.goalless_death_risk > risk_threshold:
                print(f"  ⚠️ {lora.name} ({age_tag}, {age_in_years:.1f} yaş) hedefsiz! Risk: {lora.goalless_death_risk*100:.1f}%")


@dataclass
class Goal:
    """Hedef sınıfı"""
    type: str                    # 'pattern_mastery', 'fitness_target', 'social_bond'
    target_pattern: str = None   # Hedef pattern (varsa)
    target_value: float = 0.0    # Hedef değer
    priority: str = 'main'       # 'main', 'mid', 'micro'
    patience: int = 300          # Kaç maç bekleyecek
    heves: float = 1.0           # Heves (zamanla azalır)
    match_count_stuck: int = 0   # Kaç maçtır ilerleme yok

