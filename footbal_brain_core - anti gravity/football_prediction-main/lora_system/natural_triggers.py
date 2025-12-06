"""
🌍 DOĞAL TETİKLEYİCİLER - Etki → Tepki Sistemi
===============================================

Sıklık yok, sadece neden-sonuç!
Gerçek dünya gibi: Doğa belli eşiklere ulaşınca tepki verir.

Saatli sistem değil, organik sistem!
"""

import numpy as np
import random
import math
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class Threshold:
    """Eşik değerleri"""
    name: str
    value: float
    crossed: bool = False
    cross_count: int = 0
    last_cross_match: int = 0


class NaturalTriggerSystem:
    """
    Doğal tetikleyici sistemi
    
    Sıklık yok, sadece eşikler:
    - Doğa sağlığı < 0.3 → Bir şey olur
    - Öfke > 0.7 → Bir şey olur
    - Nüfus > 80 → Bir şey olur
    - Hata streak > 20 → Bir şey olur
    
    Ne zaman olacağı belli değil, ama mantığı var!
    """
    
    def __init__(self):
        # 🌍 EVRİMLEŞEN DOĞA (Dışarıdan set edilecek!)
        self.adaptive_nature = None
        
        # 🌡️ NATURE THERMOSTAT (Dışarıdan set edilecek!)
        self.nature_thermostat = None
        
        # 🌊 AKIŞKAN NÜFUS EŞİKLERİ (250 LoRA için optimize!)
        self.dynamic_population_threshold = 250.0  # İlk değer (250 LoRA başlangıç için!)
        self.last_population_entropy = 0.5  # İlk entropi
        self.last_lazarus_avg = 0.5  # İlk Lazarus ortalaması
        
        # Eşikler (Thresholds)
        self.thresholds = {
            # Doğa sağlığı eşikleri
            'health_critical': Threshold('Kritik Sağlık', 0.20),
            'health_low': Threshold('Düşük Sağlık', 0.40),
            'health_medium': Threshold('Orta Sağlık', 0.60),
            
            # Öfke eşikleri
            'anger_explosive': Threshold('Patlayıcı Öfke', 0.80),
            'anger_high': Threshold('Yüksek Öfke', 0.60),
            'anger_rising': Threshold('Yükselen Öfke', 0.40),
            
            # Kaos eşikleri
            'chaos_extreme': Threshold('Ekstrem Kaos', 0.80),
            'chaos_high': Threshold('Yüksek Kaos', 0.60),
            
            # Nüfus eşikleri artık AKIŞKAN! (Eski sabit değerler sadece fallback)
            'population_critical': Threshold('Kritik Nüfus', 300),  # Fallback
            'population_threshold': Threshold('Eşik Nüfus', 200),   # Fallback
            'population_warning': Threshold('Uyarı Nüfus', 150),    # Fallback
        }
        
        # Birikimli değerler (sadece artar, reset olmaz)
        self.cumulative_damage = 0.0      # Doğaya verilen toplam zarar
        self.cumulative_mistakes = 0      # Toplam hata sayısı
        self.mistake_streak = 0           # Ardışık hata
        self.success_streak = 0           # Ardışık başarı
        
        # Verimlilik takibi
        self.recent_successes = []  # Son 50 maçın başarıları (True/False)
        self.overall_success_rate = 0.5  # Genel başarı oranı
        
        # Son olaylar (bellek)
        self.recent_events = []
        self.match_count = 0
        
        print("🌍 Doğal Tetikleyici Sistemi başlatıldı")
        print("   → Sıklık yok, sadece neden-sonuç!")
    
    def _calculate_efficiency(self, population_size: int) -> float:
        """
        Verimlilik hesapla
        
        VERİMLİLİK = Başarı Oranı / (Nüfus / 250)  # 🌊 250 LoRA BASE!
        
        Returns:
            0-1 arası (0.20+ = verimli, <0.15 = verimsiz)
        """
        if len(self.recent_successes) == 0:
            return 0.5  # Henüz veri yok
        
        # Son 50 maçın başarı oranı
        success_rate = sum(self.recent_successes) / len(self.recent_successes)
        
        # Nüfus faktörü (50 = 1.0, 100 = 2.0, 200 = 4.0)
        population_factor = population_size / 250.0  # 🌊 250 LoRA BASE!
        
        # Verimlilik
        efficiency = success_rate / population_factor
        
        return efficiency
    
    def _calculate_expected_success(self, population_size: int) -> float:
        """
        Beklenen başarı oranı (nüfusa göre artar!)
        
        50 LoRA: %50 beklenir
        100 LoRA: %55 beklenir
        200 LoRA: %65 beklenir
        300 LoRA: %75 beklenir
        """
        expected = 0.50 + (population_size - 50) / 1000.0
        return min(0.80, expected)  # Max %80
    
    def update_state(self, nature_state, population_size: int, 
                     match_was_success: bool, mistake_severity: float = 0.0,
                     population_entropy: float = 0.5, lazarus_avg: float = 0.5):
        """
        Durumu güncelle ve eşikleri kontrol et
        
        Args:
            population_entropy: Popülasyon tahmin çeşitliliği (0-1)
            lazarus_avg: Ortalama Lazarus Lambda (0-1)
        
        Returns:
            Tetiklenen olay (varsa)
        """
        self.match_count += 1
        
        # Başarı geçmişi (son 50 maç)
        self.recent_successes.append(match_was_success)
        if len(self.recent_successes) > 50:
            self.recent_successes.pop(0)  # İlk elemanı çıkar (sliding window)
        
        # Genel başarı oranı güncelle
        if len(self.recent_successes) > 0:
            self.overall_success_rate = sum(self.recent_successes) / len(self.recent_successes)
        
        # 🔥 VERİMLİLİK BAZLI ÖFKE ARTIŞI!
        expected_success = self._calculate_expected_success(population_size)
        actual_success = self.overall_success_rate
        
        expectation_gap = expected_success - actual_success
        
        if expectation_gap > 0:
            # BEKLENTİ KARŞILANMADI! Öfke artar!
            anger_increase = expectation_gap * 0.05  # Yavaşça birikir
            nature_state.anger = min(1.0, nature_state.anger + anger_increase)
        else:
            # BEKLENTİ AŞILDI! Öfke azalır!
            anger_decrease = abs(expectation_gap) * 0.03
            nature_state.anger = max(0.0, nature_state.anger - anger_decrease)
        
        # Birikimli değerleri güncelle
        if not match_was_success:
            self.cumulative_damage += mistake_severity
            self.cumulative_mistakes += 1
            self.mistake_streak += 1
            self.success_streak = 0
        else:
            self.success_streak += 1
            self.mistake_streak = 0
        
        # Eşikleri kontrol et (🌊 AKIŞKAN PARAMETRELER İLE!)
        triggered_event = self._check_all_thresholds(
            nature_state, 
            population_size,
            population_entropy,
            lazarus_avg
        )
        
        return triggered_event
    
    def _calculate_dynamic_anger_threshold(self, level: str = 'low') -> float:
        """
        🌊 AKIŞKAN ANGER THRESHOLD
        
        Nature temperature'a göre: Sıcak → Hassas, Soğuk → Toleranslı
        """
        if self.nature_thermostat:
            temp = self.nature_thermostat.temperature
        else:
            temp = 0.5
        
        # Base threshold'lar (sıcaklık 0.5'te)
        base_thresholds = {
            'rising': 0.30,
            'high': 0.50,
            'explosive': 0.70,
            'veba': 0.80
        }
        
        base = base_thresholds.get(level, 0.50)
        
        # Sıcaklık yüksek → threshold düşer (hassas!)
        # Sıcaklık düşük → threshold yükselir (toleranslı!)
        dynamic_threshold = base * (1.5 - temp)
        # Temp 0 → threshold 1.5x (çok toleranslı)
        # Temp 0.5 → threshold 1.0x (normal)
        # Temp 1 → threshold 0.5x (çok hassas!)
        
        return dynamic_threshold
    
    def _calculate_dynamic_health_threshold(self, level: str = 'medium') -> float:
        """
        🌊 AKIŞKAN HEALTH THRESHOLD
        
        Nature temperature'a göre: Sıcak → Daha sağlıklı olması lazım
        """
        if self.nature_thermostat:
            temp = self.nature_thermostat.temperature
        else:
            temp = 0.5
        
        base_thresholds = {
            'medium': 0.70,
            'low': 0.50,
            'critical': 0.20
        }
        
        base = base_thresholds.get(level, 0.50)
        
        # Sıcaklık yüksek → health threshold yükselir (daha sağlıklı olması lazım)
        dynamic_threshold = base * (0.5 + temp)
        # Temp 0 → threshold 0.5x (düşük sağlık tolere edilir)
        # Temp 1 → threshold 1.5x (yüksek sağlık gerekir!)
        
        return dynamic_threshold
    
    def _calculate_dynamic_cooldown(self, base_cooldown: int, population_size: int, lazarus_avg: float) -> int:
        """
        🌊 AKIŞKAN COOLDOWN
        
        Population recovery capacity'ye göre: Güçlü → Hızlı, Zayıf → Yavaş
        """
        # 🌊 AKIŞKAN RECOVERY (250 LoRA scale'e göre!)
        # Nüfus büyük + Lazarus yüksek → Hızlı recovery
        population_factor = max(population_size / 250.0, 0.1)  # 250 LoRA = 1.0x
        lazarus_factor = lazarus_avg + 0.1  # Min 0.1
        
        recovery_capacity = population_factor * lazarus_factor
        
        # Recovery yüksek → cooldown kısa
        dynamic_cooldown = int(base_cooldown / recovery_capacity)
        
        # En az 3, en fazla 1000 maç (extreme durumlar için)
        dynamic_cooldown = max(3, min(1000, dynamic_cooldown))
        
        return dynamic_cooldown
    
    def _check_all_thresholds(self, nature_state, population_size: int, population_entropy=0.5, lazarus_avg=0.5) -> Optional[Dict]:
        """
        Tüm eşikleri kontrol et (🌊 TAM AKIŞKAN!)
        Bir eşik geçildiğinde doğa tepki verir!
        
        HİYERARŞİ:
        SEVİYE 1: Küçük (her 5-10 maç)
        SEVİYE 2: Orta (her 30-50 maç)
        SEVİYE 3: Büyük (100-200 maç)
        SEVİYE 4: SON - KARA VEBA (500+ maç, SADECE 1 KEZ!)
        
        🌊 TÜM THRESHOLD'LAR DİNAMİK!
        """
        
        # ============================================
        # SEVİYE 1: KÜÇÜK TEPKİLER (Sık olur!)
        # ============================================
        
        # 🌊 DİNAMİK THRESHOLD'LARI HESAPLA!
        anger_rising_threshold = self._calculate_dynamic_anger_threshold('rising')
        
        # 1A) HAFİF SARSINTI (🌊 DİNAMİK!)
        dynamic_cooldown_rising = self._calculate_dynamic_cooldown(5, population_size, lazarus_avg)
        
        if (nature_state.anger > anger_rising_threshold and 
            self.match_count >= 5 and
            self._is_fresh_threshold('anger_rising', cooldown=dynamic_cooldown_rising)):
            
            self._mark_threshold_crossed('anger_rising')
            
            # 🌊 DİNAMİK SEVERITY (Cumulative damage'e göre!)
            dynamic_severity = 0.15 + (0.3 * min(1.0, self.cumulative_damage / 50.0))
            dynamic_affected = 0.10 + (0.2 * min(1.0, self.cumulative_damage / 50.0))
            
            return {
                'type': 'minor_shake',
                'trigger': 'fluid_anger_rising',
                'message': f'🌱 Hafif sarsıntı (Maç #{self.match_count}, Anger>{anger_rising_threshold:.2f})',
                'severity': dynamic_severity,  # 🌊 DİNAMİK!
                'affected_ratio': dynamic_affected  # 🌊 DİNAMİK!
            }
        
        # 🌊 DİNAMİK THRESHOLD'LARI HESAPLA!
        health_medium_threshold = self._calculate_dynamic_health_threshold('medium')
        dynamic_cooldown_health = self._calculate_dynamic_cooldown(8, population_size, lazarus_avg)
        
        # 1B) STRES DALGASI (🌊 DİNAMİK!)
        if (nature_state.health < health_medium_threshold and 
            self.match_count >= 8 and
            self._is_fresh_threshold('health_medium', cooldown=dynamic_cooldown_health)):
            
            self._mark_threshold_crossed('health_medium')
            
            # 🌊 DİNAMİK SEVERITY!
            dynamic_severity = 0.20 + (0.25 * min(1.0, self.cumulative_damage / 50.0))
            
            return {
                'type': 'stress_wave',
                'trigger': 'fluid_health_medium',
                'message': f'💨 Stres dalgası (Maç #{self.match_count}, Health<{health_medium_threshold:.2f})',
                'severity': dynamic_severity,  # 🌊 DİNAMİK!
                'affected_ratio': dynamic_severity * 0.8
            }
        
        # ============================================
        # SEVİYE 2: ORTA TEPKİLER (🌊 DİNAMİK!)
        # ============================================
        
        # 🌊 DİNAMİK THRESHOLD'LAR!
        anger_high_threshold = self._calculate_dynamic_anger_threshold('high')
        dynamic_cooldown_quake = self._calculate_dynamic_cooldown(30, population_size, lazarus_avg)
        
        # 2A) DEPREM (🌊 DİNAMİK!)
        if (nature_state.anger > anger_high_threshold and 
            self.match_count >= 30 and
            self._is_fresh_threshold('anger_high', cooldown=dynamic_cooldown_quake)):
            
            self._mark_threshold_crossed('anger_high')
            
            # 🌊 DİNAMİK SEVERITY!
            dynamic_severity = 0.40 + (0.30 * min(1.0, self.cumulative_damage / 80.0))
            
            return {
                'type': 'quake',
                'trigger': 'fluid_anger_high',
                'message': f'🌍 Deprem! (Maç #{self.match_count}, Anger>{anger_high_threshold:.2f})',
                'severity': dynamic_severity,  # 🌊 DİNAMİK!
                'affected_ratio': dynamic_severity * 0.7
            }
        
        # 🌊 DİNAMİK HEALTH THRESHOLD!
        health_low_threshold = self._calculate_dynamic_health_threshold('low')
        dynamic_cooldown_health_low = self._calculate_dynamic_cooldown(40, population_size, lazarus_avg)
        
        # 2B) SAĞLIK DÜŞÜK (🌊 DİNAMİK!)
        if (nature_state.health < health_low_threshold and 
            self.match_count >= 40 and
            self._is_fresh_threshold('health_low', cooldown=dynamic_cooldown_health_low)):
            
            self._mark_threshold_crossed('health_low')
            
            # 🌊 DİNAMİK SEVERITY!
            dynamic_severity = 0.35 + (0.30 * min(1.0, self.cumulative_damage / 80.0))
            
            return {
                'type': 'health_crisis',
                'trigger': 'fluid_health_low',
                'message': f'🩹 Sağlık krizi (Maç #{self.match_count}, Health<{health_low_threshold:.2f})',
                'severity': dynamic_severity,  # 🌊 DİNAMİK!
                'affected_ratio': dynamic_severity * 0.6
            }
        
        # ============================================
        # SEVİYE 3: BÜYÜK TEPKİLER
        # ============================================
        
        # 🌊 DİNAMİK THRESHOLD!
        anger_explosive_threshold = self._calculate_dynamic_anger_threshold('explosive')
        dynamic_cooldown_explosive = self._calculate_dynamic_cooldown(100, population_size, lazarus_avg)
        
        # 3A) BÜYÜK DEPREM (🌊 DİNAMİK!)
        if (nature_state.anger > anger_explosive_threshold and 
            self.match_count >= 100 and
            self._is_fresh_threshold('anger_explosive', cooldown=dynamic_cooldown_explosive)):
            
            self._mark_threshold_crossed('anger_explosive')
            
            # 🌊 DİNAMİK SEVERITY VE KILL RATIO!
            dynamic_severity = 0.60 + (0.30 * min(1.0, self.cumulative_damage / 100.0))
            # Kill ratio - LİMİT YOK! Sigmoid doğal limitini kullanır
            excess_anger = (nature_state.anger - anger_explosive_threshold) / anger_explosive_threshold
            dynamic_kill_ratio = 1 - math.exp(-excess_anger * 2)  # LİMİT YOK!
            
            return {
                'type': 'major_quake',
                'trigger': 'fluid_anger_explosive',
                'message': f'🌍🌍 BÜYÜK DEPREM! (Maç #{self.match_count}, Anger>{anger_explosive_threshold:.2f}, Kill:{dynamic_kill_ratio*100:.0f}%)',
                'severity': dynamic_severity,  # 🌊 DİNAMİK!
                'kill_ratio': dynamic_kill_ratio  # 🌊 LİMİT YOK!
            }
        
        # ============================================
        # SEVİYE 4: KARA VEBA (TARİHTE 1 KEZ!) (🌊 DİNAMİK!)
        # ============================================
        # 🌊 DİNAMİK KOŞULLAR:
        # 1. MEDENİYET: population >= dynamic_threshold * 4 (Çok büyük!)
        # 2. HEALTH: < dynamic_critical_threshold
        # 3. ANGER: > dynamic_veba_threshold  
        # 4. UZUN SÜRE: match >= 500
        # 5. COOLDOWN: dynamic (recovery capacity'ye göre!)
        
        health_critical_threshold = self._calculate_dynamic_health_threshold('critical')
        anger_veba_threshold = self._calculate_dynamic_anger_threshold('veba')
        veba_population_threshold = self.dynamic_population_threshold * 4  # Eşiğin 4 katı!
        dynamic_cooldown_veba = self._calculate_dynamic_cooldown(500, population_size, lazarus_avg)
        
        if (population_size >= veba_population_threshold and
            nature_state.health < health_critical_threshold and
            nature_state.anger > anger_veba_threshold and
            self.match_count >= 500):
            
            # Daha önce oldu mu? (🌊 DİNAMİK COOLDOWN!)
            last_kara_veba = self.thresholds['health_critical'].last_cross_match
            if last_kara_veba > 0 and self.match_count - last_kara_veba < dynamic_cooldown_veba:
                # ÇOK YAKINDA OLDU!
                return None
            
            # Cross count kontrol - SADECE 1 KEZ!
            if self.thresholds['health_critical'].cross_count >= 1:
                # ZATEN 1 KEZ OLDU, BİR DAHA OLMAMALI!
                return None
            
            self.thresholds['health_critical'].crossed = True
            self.thresholds['health_critical'].cross_count += 1
            self.thresholds['health_critical'].last_cross_match = self.match_count
            
            # 🌊 DİNAMİK SURVIVAL RATE (Lazarus Lambda'ya göre!)
            # Yüksek Lazarus → Daha fazla hayatta kalır (öğrenme kapasitesi yüksek)
            base_survival = 0.10
            lazarus_bonus = lazarus_avg * 0.15  # Max +%15
            dynamic_survival_rate = base_survival + lazarus_bonus
            
            return {
                'type': 'kara_veba',
                'trigger': 'fluid_civilization_collapse',
                'message': f'☠️☠️☠️ KARA VEBA! (Maç #{self.match_count}) Pop:{population_size}>{veba_population_threshold:.0f}, Health:{nature_state.health:.2f}<{health_critical_threshold:.2f}, Survival:{dynamic_survival_rate*100:.0f}%',
                'severity': 0.95,
                'survival_rate': dynamic_survival_rate  # 🌊 DİNAMİK! %10-25 arası
            }
        
        # 2) ÖFKE PATLAYICI (> 0.80) → DEPREM!
        # ⚠️ YENİ: MİNİMUM MAÇ SAYISI! (İlk 50 maçta olmasın!)
        if (nature_state.anger > 0.80 and 
            self.match_count >= 50 and  # ✅ EN AZ 50 MAÇ GEREKLİ!
            not self.thresholds['anger_explosive'].crossed):
            
            # ✅ COOLDOWN KONTROLÜ (Son depremden 50 maç geçmeli!)
            last_quake = self.thresholds['anger_explosive'].last_cross_match
            if last_quake > 0 and self.match_count - last_quake < 50:
                return None
            
            self.thresholds['anger_explosive'].crossed = True
            self.thresholds['anger_explosive'].cross_count += 1
            self.thresholds['anger_explosive'].last_cross_match = self.match_count
            
            return {
                'type': 'major_quake',
                'trigger': 'anger_explosive',
                'message': f'🌍 BÜYÜK DEPREM! (Maç #{self.match_count}) Doğanın öfkesi patladı!',
                'severity': 0.85,
                'affected_ratio': 0.70
            }
        
        # 3A) KRİTİK NÜFUS (> 300) + VERİMLİLİK KONTROLÜ → KİTLESEL ÖLÜM!
        # VERİMLİLİK = Başarı / (Nüfus / 250)  # 🌊 250 LoRA BASE!
        if population_size > 300 and self.match_count >= 200:
            # Verimlilik hesapla
            efficiency = self._calculate_efficiency(population_size)
            
            # Verimsizse müdahale!
            if efficiency < 0.15:  # Çok düşük verimlilik!
                
                # COOLDOWN (Son felaketten 200 maç geçmeli!)
                last_disaster = self.thresholds.get('population_critical', Threshold('temp', 300)).last_cross_match
                if last_disaster > 0 and self.match_count - last_disaster < 200:
                    return None
                
                if 'population_critical' in self.thresholds:
                    self.thresholds['population_critical'].crossed = True
                    self.thresholds['population_critical'].cross_count += 1
                    self.thresholds['population_critical'].last_cross_match = self.match_count
                
                return {
                    'type': 'mass_extinction',
                    'trigger': 'inefficient_overpopulation',
                    'message': f'💀 VERİMSİZ NÜFUS! (Maç #{self.match_count}) {population_size} LoRA ama verimlilik: {efficiency:.1%}',
                    'severity': 0.90,
                    'kill_ratio': 0.60  # %60 ölür!
                }
        
        # 3B) 🌊 AKIŞKAN NÜFUS KONTROLÜ! (🌌 EVREN GENİŞLETİLDİ: İlk 250 maç yok!)
        if self.match_count >= 250:
            # DİNAMİK EŞİĞİ HESAPLA!
            dynamic_threshold = self.calculate_dynamic_population_threshold(
                nature_state, 
                population_entropy, 
                lazarus_avg
            )
            
            # Eşiği aştı mı?
            if population_size > dynamic_threshold:
                # Verimlilik kontrol
                efficiency = self._calculate_efficiency(population_size)
                
                # Verimsizse müdahale! (🌌 EVREN GENİŞLETİLDİ: Çok daha toleranslı!)
                if efficiency < 0.08:  # ÇOK düşük verimlilik!
                    
                    # COOLDOWN (Son felaketten 150 maç geçmeli!)
                    last_disaster = self.thresholds.get('population_threshold', Threshold('temp', 200)).last_cross_match
                    if last_disaster > 0 and self.match_count - last_disaster < 150:
                        return None
                    
                    # DİNAMİK KILL RATIO HESAPLA!
                    dynamic_kill_ratio = self.calculate_dynamic_kill_ratio(population_size, dynamic_threshold)
                    
                    if 'population_threshold' in self.thresholds:
                        self.thresholds['population_threshold'].crossed = True
                        self.thresholds['population_threshold'].cross_count += 1
                        self.thresholds['population_threshold'].last_cross_match = self.match_count
                    
                    return {
                        'type': 'overpopulation_purge',
                        'trigger': 'fluid_inefficient_threshold',
                        'message': f'🌊 AKIŞKAN MÜDAHALE! (Maç #{self.match_count}) {population_size} LoRA > Eşik:{dynamic_threshold:.0f}, Verimlilik:{efficiency:.1%}, Kill:{dynamic_kill_ratio*100:.0f}%',
                        'severity': 0.80,
                        'kill_ratio': dynamic_kill_ratio,  # 🌊 DİNAMİK!
                        'dynamic_threshold': dynamic_threshold,  # Log için
                        'population_entropy': population_entropy,  # Log için
                        'lazarus_avg': lazarus_avg  # Log için
                    }
        
        # 4) 🌊 YÜKSEK NÜFUS + DÜŞÜK SAĞLIK → TEDRİCİ ÖLÜM (AKIŞKAN!)
        # Dinamik eşiği kullan
        if self.match_count >= 80:  # En az 80 maç geçmeli
            # Dinamik eşiği hesapla
            dynamic_threshold_health = self.calculate_dynamic_population_threshold(
                nature_state, 
                population_entropy, 
                lazarus_avg
            )
            
            # Eşiğin %80'inden fazla ve sağlık düşükse
            if (population_size > dynamic_threshold_health * 0.8 and 
                nature_state.health < 0.40 and
                self._is_fresh_threshold('population_warning')):
                
                self._mark_threshold_crossed('population_warning')
                
                # Dinamik kill ratio (daha yumuşak)
                excess_ratio = (population_size - dynamic_threshold_health * 0.8) / (dynamic_threshold_health * 0.8)
                slow_kill_ratio = min(0.3, excess_ratio * 0.15)  # Max %30
                
                return {
                    'type': 'slow_purge',
                    'trigger': 'fluid_population_health',
                    'message': f'🦠 Tedricî ölüm (Nüfus: {population_size} > Eşik %80:{dynamic_threshold_health*0.8:.0f}, Sağlık: {nature_state.health:.2f})',
                    'severity': 0.60,
                    'kill_ratio': slow_kill_ratio  # 🌊 DİNAMİK!
                }
        
        # 5) BİRİKİMLİ ZARAR YÜKSEK (> 50) → DOĞA UYANIR
        if (self.cumulative_damage > 50 and 
            self._is_fresh_threshold('cumulative_damage_50')):
            
            self._mark_threshold_crossed('cumulative_damage_50')
            
            return {
                'type': 'nature_awakens',
                'trigger': 'cumulative_damage',
                'message': f'🌪️ Doğa uyanıyor! Toplam zarar: {self.cumulative_damage:.1f}',
                'severity': 0.70,
                'affected_ratio': 0.50
            }
        
        # 6) UZUN HATA STREAKİ (> 30) → KAOS ARTAR
        if (self.mistake_streak > 30 and 
            self._is_fresh_threshold(f'mistake_streak_30')):
            
            self._mark_threshold_crossed(f'mistake_streak_30')
            
            return {
                'type': 'chaos_surge',
                'trigger': 'mistake_streak',
                'message': f'⚡ {self.mistake_streak} ardışık hata! Kaos patladı!',
                'severity': 0.50,
                'chaos_boost': 0.30
            }
        
        # 7) KAOS + ÖFKE YÜKSEK → KOMBİNE OLAY
        if (nature_state.chaos_index > 0.70 and 
            nature_state.anger > 0.60 and
            self._is_fresh_threshold('chaos_anger_combo')):
            
            self._mark_threshold_crossed('chaos_anger_combo')
            
            return {
                'type': 'perfect_storm',
                'trigger': 'chaos + anger',
                'message': '🌀 Mükemmel Fırtına! Kaos ve öfke birleşti!',
                'severity': 0.80,
                'affected_ratio': 0.60,
                'chaos_reset': True  # Kaos sıfırlanır
            }
        
        # 8) RESET MEKANİZMASI: Doğa iyileşirse eşikler sıfırlanır
        self._check_threshold_resets(nature_state, population_size)
        
        # 9) DOĞAL GÜRÜLTÜ (Her zaman var, ama çok hafif)
        # Bu da eşik bazlı: Kaos > 0.2 ise küçük sallantı olabilir
        if nature_state.chaos_index > 0.20:
            # Kaos seviyesine göre olasılık
            tremor_chance = (nature_state.chaos_index - 0.20) * 0.10  # Max %8
            
            if random.random() < tremor_chance:
                return {
                    'type': 'natural_tremor',
                    'trigger': 'background_chaos',
                    'message': f'⚡ Doğal titreşim (kaos: {nature_state.chaos_index:.2f})',
                    'severity': random.uniform(0.05, 0.15),
                    'affected_ratio': random.uniform(0.05, 0.15)
                }
        
        return None
    
    def _check_threshold_resets(self, nature_state, population_size: int):
        """
        Doğa iyileşirse eşikler sıfırlanır
        Böylece aynı eşik tekrar tetiklenebilir!
        """
        
        # Sağlık iyileşti mi?
        if nature_state.health > 0.60:
            if self.thresholds['health_critical'].crossed:
                self.thresholds['health_critical'].crossed = False
                print("  💚 Doğa sağlığı iyileşti, kritik eşik sıfırlandı")
            
            if self.thresholds['health_low'].crossed:
                self.thresholds['health_low'].crossed = False
        
        # Öfke azaldı mı?
        if nature_state.anger < 0.40:
            if self.thresholds['anger_explosive'].crossed:
                self.thresholds['anger_explosive'].crossed = False
                print("  😌 Doğa sakinleşti, öfke eşiği sıfırlandı")
            
            if self.thresholds['anger_high'].crossed:
                self.thresholds['anger_high'].crossed = False
        
        # Nüfus azaldı mı? (YENİ EŞİKLER!)
        if population_size < 150:
            # Eski eşikler kaldırıldı, yeni eşikler kullan
            if 'population_critical' in self.thresholds and self.thresholds['population_critical'].crossed:
                self.thresholds['population_critical'].crossed = False
                print("  👥 Nüfus çok düştü (<150), kritik eşik sıfırlandı")
            
            if 'population_threshold' in self.thresholds and self.thresholds['population_threshold'].crossed:
                self.thresholds['population_threshold'].crossed = False
                print("  👥 Nüfus normale döndü (<150), eşik sıfırlandı")
            
            if 'population_warning' in self.thresholds and self.thresholds['population_warning'].crossed:
                self.thresholds['population_warning'].crossed = False
        
        # Kaos azaldı mı?
        if nature_state.chaos_index < 0.30:
            if self.thresholds['chaos_extreme'].crossed:
                self.thresholds['chaos_extreme'].crossed = False
                print("  🌊 Kaos normale döndü, ekstrem eşik sıfırlandı")
    
    def calculate_dynamic_population_threshold(self, nature_state, population_entropy, lazarus_avg):
        """
        🌊 AKIŞKAN NÜFUS EŞİĞİ
        
        Hiçbir sabit sayı yok! Her şey anlık duruma göre hesaplanıyor:
        - Nature's temperature (sıcak → tolerans düşük)
        - Population entropy (çeşitlilik → tolerans yüksek)
        - Lazarus Lambda ortalaması (potansiyel → tolerans yüksek)
        
        Returns:
            dynamic_threshold (float): Anlık nüfus eşiği
        """
        # 1) BASE: Nature's temperature'a göre (🌌 EVREN GENİŞLETİLDİ!)
        if self.nature_thermostat:
            temp = self.nature_thermostat.temperature
        else:
            temp = 0.5  # Default
        
        base_threshold = 150 + (250 * (1 - temp))
        # Sıcaklık 0 (soğuk, öngörülebilir) → 400 LoRA tolere edilir! 🌌
        # Sıcaklık 1 (sıcak, kaotik) → 150 LoRA tolere edilir
        
        # 2) ENTROPY FAKTÖRÜ: Çeşitlilik azsa tolerans düşük
        entropy_factor = 0.5 + (population_entropy * 0.5)
        # Entropi 0 (herkes aynı) → 0.5x, eşik düşer
        # Entropi 1 (tam çeşitlilik) → 1.0x, eşik artar
        
        # 3) POTENTIAL FAKTÖRÜ: Lazarus Lambda ortalaması yüksekse tolerans yüksek
        potential_factor = 0.7 + (lazarus_avg * 0.6)
        # Lazarus 0 → 0.7x
        # Lazarus 1 → 1.3x (Yüksek potansiyel → daha çok LoRA yaşayabilir)
        
        # 4) HEALTH FAKTÖRÜ: Doğa sağlığı düşükse tolerans düşük
        health_factor = 0.6 + (nature_state.health * 0.4)
        # Health 0 → 0.6x (hasta doğa → az tolere eder)
        # Health 1 → 1.0x
        
        # 5) FİNAL THRESHOLD (🌊 TAM AKIŞKAN - HİÇBİR LİMİT YOK!)
        dynamic_threshold = base_threshold * entropy_factor * potential_factor * health_factor
        
        # LİMİT YOK! Formül ne diyorsa o! 🌊
        # Eğer formül 1000 diyorsa → 1000 LoRA tolere edilir
        # Eğer formül 10 diyorsa → Sadece 10 LoRA tolere edilir
        # TAM AKIŞKANLIK!
        
        # Kaydet (log için)
        self.dynamic_population_threshold = dynamic_threshold
        self.last_population_entropy = population_entropy
        self.last_lazarus_avg = lazarus_avg
        
        return dynamic_threshold
    
    def calculate_dynamic_kill_ratio(self, population_size, dynamic_threshold):
        """
        🌊 AKIŞKAN ÖLDÜRME ORANI
        
        Sigmoid benzeri: Fazla nüfus arttıkça agresif ölçeklenir
        
        Returns:
            kill_ratio (float): 0-0.6 arası
        """
        if population_size <= dynamic_threshold:
            return 0  # Eşiğin altında, müdahale yok
        
        # Fazla nüfus oranı
        excess_ratio = (population_size - dynamic_threshold) / dynamic_threshold
        
        # Sigmoid benzeri: Fazla nüfus arttıkça agresif ölçeklenir
        # 1 - exp(-x) formülü kullanıyoruz
        kill_ratio = 1 - math.exp(-excess_ratio)
        
        # 🌊 LİMİT YOK! Formül ne derse o!
        # Eğer excess çok yüksekse, %99 bile ölürebilir!
        # Bu doğanın gerçek gücü!
        
        return kill_ratio  # 🌊 TAM AKIŞKAN!
    
    def _is_fresh_threshold(self, threshold_name: str, cooldown: int = 100) -> bool:
        """
        Bu eşik daha önce geçilmedi mi?
        Veya geçildiyse çok uzun zaman geçti mi?
        
        Args:
            threshold_name: Eşik adı
            cooldown: Cooldown süresi (maç sayısı)
        """
        if threshold_name not in self.thresholds:
            # Dinamik eşikler için (streak'ler vs.)
            # Son cooldown maçta bu olay gerçekleşti mi?
            recent_matches = [e for e in self.recent_events 
                            if self.match_count - e.get('match', 0) < cooldown]
            
            for event in recent_matches:
                if event.get('trigger') == threshold_name:
                    return False  # Çok yakın zamanda oldu
            
            return True
        
        threshold = self.thresholds[threshold_name]
        
        # Hiç geçilmediyse
        if not threshold.crossed:
            return True
        
        # Geçildiyse ama 200+ maç geçtiyse tekrar tetiklenebilir
        if self.match_count - threshold.last_cross_match > 200:
            return True
        
        return False
    
    def _mark_threshold_crossed(self, threshold_name: str):
        """Eşik geçildi olarak işaretle"""
        if threshold_name in self.thresholds:
            self.thresholds[threshold_name].crossed = True
            self.thresholds[threshold_name].cross_count += 1
            self.thresholds[threshold_name].last_cross_match = self.match_count
        
        # Recent events'e ekle
        self.recent_events.append({
            'trigger': threshold_name,
            'match': self.match_count
        })
        
        # Eski eventleri temizle (son 500 maç)
        self.recent_events = [e for e in self.recent_events 
                             if self.match_count - e['match'] < 500]
    
    def get_status(self) -> Dict:
        """Sistem durumu"""
        active_thresholds = {
            name: t for name, t in self.thresholds.items() 
            if t.crossed
        }
        
        return {
            'match': self.match_count,
            'cumulative_damage': self.cumulative_damage,
            'cumulative_mistakes': self.cumulative_mistakes,
            'mistake_streak': self.mistake_streak,
            'success_streak': self.success_streak,
            'active_thresholds': len(active_thresholds),
            'active_threshold_names': list(active_thresholds.keys())
        }
    
    def print_status(self):
        """Durum yazdır"""
        status = self.get_status()
        
        print(f"\n{'='*70}")
        print(f"⚡ DOĞAL TETİKLEYİCİLER (Maç #{status['match']})")
        print(f"{'='*70}")
        print(f"  💥 Birikimli Zarar: {status['cumulative_damage']:.1f}")
        print(f"  ❌ Toplam Hata: {status['cumulative_mistakes']}")
        print(f"  📉 Hata Streaki: {status['mistake_streak']}")
        print(f"  📈 Başarı Streaki: {status['success_streak']}")
        print(f"  🚨 Aktif Eşikler: {status['active_thresholds']}")
        
        if status['active_threshold_names']:
            print(f"     → {', '.join(status['active_threshold_names'])}")
        
        print(f"{'='*70}\n")


# Kullanım örneği
if __name__ == "__main__":
    from nature_entropy_system import NatureState
    
    trigger_system = NaturalTriggerSystem()
    nature = NatureState()
    population = 20
    
    # Simülasyon
    for match in range(1, 501):
        # Her maç bir şeyler oluyor
        match_success = random.random() > 0.4  # %60 başarı
        
        if not match_success:
            nature.health -= 0.01
            nature.anger += 0.02
            mistake_severity = random.uniform(0.1, 0.3)
        else:
            nature.health = min(1.0, nature.health + 0.005)
            nature.anger = max(0.0, nature.anger - 0.01)
            mistake_severity = 0.0
        
        # Nüfus değişir
        if random.random() < 0.1:
            population += 1
        if random.random() < 0.05:
            population -= 1
        
        # Tetikleyicileri kontrol et
        event = trigger_system.update_state(
            nature, population, match_success, mistake_severity
        )
        
        if event:
            print(f"\n🌍 MAÇ #{match}: {event['message']}")
            print(f"   Tetikleyici: {event['trigger']}")
            print(f"   Şiddet: {event['severity']:.2f}")
        
        # Her 50 maçta durum
        if match % 50 == 0:
            trigger_system.print_status()

