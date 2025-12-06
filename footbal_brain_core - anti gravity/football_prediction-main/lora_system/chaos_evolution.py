"""
🌪️ KAOS EVRİM MOTORU (🌊 TAM AKIŞKAN!)
=========================================

Tamamen doğal seleksiyon + Bilimsel teori:
- AI müdahale etmiyor, sistem kendi dengesini buluyor
- Herkes herkesle çiftleşebilir (kaotik kombinasyonlar)
- Gürültü her yerde (beklenmedik keşifler)
- Spontane doğum, şanslı kurtuluş, şok mutasyonlar

🌊 AKIŞKAN PARAMETRELER:
- Üreme şansı → Genetic diversity'ye göre
- Mutasyon şansı → Population variance'a göre
- Partner selection → Population health'e göre
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from copy import deepcopy

from .lora_adapter import LoRAAdapter


class ChaosEvolutionManager:
    """
    Kaotik evrimsel yönetici
    """
    
    def __init__(self, config: Dict, device='cpu'):
        self.config = config
        self.device = device  # ✅ Device bilgisi
        self.population: List[LoRAAdapter] = []
        self.match_count = 0
        self.evolution_log = []
        
        # 🛡️ DENEYİM DİRENCİ SİSTEMİ (Dışarıdan set edilecek!)
        self.experience_resistance = None
        
        # 💕 ULTRA KAOTİK ÇİFTLEŞME (Dışarıdan set edilecek!)
        self.ultra_mating = None
        
        # 🌡️ NATURE THERMOSTAT (Dışarıdan set edilecek!)
        self.nature_thermostat = None
        
        # Parametreler
        self.min_population = config.get('population', {}).get('min_population', 5)
        self.max_population = config.get('population', {}).get('max_population', None)
        self.death_threshold = config.get('death', {}).get('threshold', 0.05)  # ✅ Doğru key!
        self.reproduction_threshold = config.get('reproduction', {}).get('fitness_threshold', 0.60)
        # 🌊 BASE değerler (akışkan hesaplama için!)
        self.base_reproduction_chance = config.get('reproduction', {}).get('chance_per_match', 0.06)
        self.base_mutation_chance = config.get('noise', {}).get('spontaneous_birth', {}).get('chance_per_match', 0.04)
        self.lucky_survival_chance = config.get('death', {}).get('lucky_survival_chance', 0.50)  # ✅ 0.50!
        
        # Çiftleşme olasılıkları (BASE değerler - akışkan olacak!)
        partner_sel = config.get('reproduction', {}).get('partner_selection', {})
        self.base_partner_random = partner_sel.get('random', 0.30)
        self.base_partner_strongest = partner_sel.get('strongest', 0.30)
        self.base_partner_weakest = partner_sel.get('weakest', 0.20)
        self.base_partner_complementary = partner_sel.get('complementary', 0.20)
        
        # Gürültü parametreleri (BASE değerler - akışkan olacak!)
        noise = config.get('noise', {})
        self.crossover_noise_max = noise.get('crossover', {}).get('base_noise_max', 0.3)
        self.mega_noise_chance = noise.get('crossover', {}).get('mega_noise_chance', 0.10)
        self.base_param_mutation_chance = noise.get('mutation', {}).get('param_mutation_chance', 0.15)
        self.base_shock_mutation_chance = noise.get('mutation', {}).get('shock_mutation_chance', 0.05)
        
        print(f"🌪️ Kaos Evrim Motoru başlatıldı!")
        print(f"   Min/Max Populasyon: {self.min_population}/{self.max_population}")
        print(f"   Ölüm Eşiği: {self.death_threshold}")
        print(f"   Üreme Eşiği: {self.reproduction_threshold}")
    
    def initialize_population(self, size: int, input_dim: int = 63, hidden_dim: int = 128, device='cpu'):
        """İlk popülasyonu oluştur"""
        print(f"🐣 İlk popülasyon oluşturuluyor: {size} LoRA...")
        
        for i in range(size):
            lora = LoRAAdapter(input_dim=input_dim, hidden_dim=hidden_dim, rank=16, alpha=16.0)
            lora = lora.to(device)  # Device'a taşı
            lora.name = f"LoRA_Gen0_{i:03d}"
            lora.generation = 0
            lora.birth_match = 0
            self.population.append(lora)
        
        print(f"✅ {len(self.population)} LoRA hazır!")
    
    def _calculate_genetic_diversity(self) -> float:
        """
        🌊 Genetik çeşitliliği hesapla (Temperament variance)
        
        Returns:
            0-1 arası diversity score
            0 = Hepsi aynı (klonlar)
            1 = Çok farklı (maksimum çeşitlilik)
        """
        if len(self.population) < 2:
            return 0.5  # Default
        
        # 15 temperament özelliği için sabit sıralama
        traits = sorted(list(self.population[0].temperament.keys()))
        
        all_values = []
        for lora in self.population:
            vec = [lora.temperament.get(t, 0.5) for t in traits]
            all_values.append(vec)
        
        # Varyans hesapla
        all_values = np.array(all_values)
        variances = np.var(all_values, axis=0)
        mean_variance = np.mean(variances)
        
        # Normalize (0.08 varyans genelde çok yüksektir)
        diversity = min(1.0, mean_variance / 0.08)
        
        return diversity

    def _calculate_natural_reproduction_chance(self, lora, population_size: int, alarm_info: dict = None) -> float:
        """
        Doğal üreme şansı hesapla (GERÇEK DÜNYA GİBİ!)
        
        Faktörler:
        1. Sosyal bağ gücü (40%) - En güçlü bağ ne kadar?
        2. Fitness (30%) - Sağlıklı mı?
        3. Hırs (15%) - İstekli mi?
        4. Nüfus faktörü (15%) - Dünya gibi: nüfus artar!
        5. 🌡️ TEMPERATURE (Akışkan!) - Sıcaksa zor, soğuksa kolay!
        
        Returns:
            0-1 arası şans
        """
        # 1. Sosyal Bağ (En iyi arkadaşı var mı?)
        social_factor = 0.0
        if hasattr(lora, 'social_bonds') and lora.social_bonds:
            max_bond = max(lora.social_bonds.values()) if lora.social_bonds else 0.0
            social_factor = max_bond  # 0-1 arası
        
        # 2. Fitness
        fitness_factor = lora.get_recent_fitness()
        
        # 3. Hırs (Ambition)
        ambition_factor = lora.temperament.get('ambition', 0.5)
        
        # 4. Nüfus Baskısı (Ters orantı: çok kalabalıksa üreme azalır)
        # Ama alarm varsa artar!
        if alarm_info and alarm_info['level'] != 'GREEN':
            population_factor = 1.0  # Kriz anında nüfus baskısı yok sayılır!
        else:
            # Normal durum: Nüfus arttıkça üreme isteği azalır (kaynak kıtlığı)
            max_pop = self.max_population if self.max_population else 1000
            population_factor = 1.0 - (population_size / max_pop)
            population_factor = max(0.1, population_factor)
        
        # Ağırlıklı toplam
        base_chance = (
            (social_factor * 0.40) +
            (fitness_factor * 0.30) +
            (ambition_factor * 0.15) +
            (population_factor * 0.15)
        )
        
        # Base scale
        final_chance = base_chance * self.base_reproduction_chance * 5.0  # Scale up
        
        # 🌡️ TEMPERATURE ETKİSİ (AKIŞKAN!)
        if self.nature_thermostat:
            temp = self.nature_thermostat.temperature
            # Sıcak (0.8) → Zorluk artar → Şans x0.7
            # Soğuk (0.2) → Kolaylık artar → Şans x1.3
            temp_modifier = 1.0 - ((temp - 0.5) * 0.6)
            final_chance *= temp_modifier
        
        return min(0.90, final_chance)

    def _determine_death_reason(self, lora, fitness: float, current_match: int = None) -> str:
        """
        Ölüm sebebini belirle
        
        Returns:
            Sebep metni (örn: "Düşük fitness (0.02)", "Hedefsizlik", vs)
        """
        # 1. Fitness çok düşük
        if fitness < 0.05:
            return f"Açlık (Fitness: {fitness:.3f})"
        
        # 2. Yaşlılık (Eğer çok yaşlıysa ve fitness düşüyorsa)
        if current_match and (current_match - lora.birth_match > 500) and fitness < 0.3:
            return f"Yaşlılık (Yaş: {current_match - lora.birth_match})"
        
        # 3. Yalnızlık (Sosyal bağ yoksa)
        if hasattr(lora, 'social_bonds') and not lora.social_bonds and fitness < 0.2:
            return "Yalnızlık (Sosyal bağ yok)"
            
        # 4. Stres (Stress tolerance düşükse)
        if lora.temperament.get('stress_tolerance', 0.5) < 0.2 and fitness < 0.25:
            return "Stres (Dayanıksızlık)"
            
        return f"Doğal Seleksiyon (Fit: {fitness:.3f})"
    
    def select_partner(self, lora: LoRAAdapter) -> Optional[LoRAAdapter]:
        """
        AKIŞKAN PARTNER SEÇİMİ!
        
        Kodlanmış %30-%30 YOK artık!
        Ultra kaotik sistem kullanılıyor!
        """
        if not self.population or len(self.population) < 2:
            return None
            
        # Kendisi hariç adaylar
        candidates = [l for l in self.population if l.id != lora.id]
        if not candidates:
            return None
            
        # 🌡️ TEMPERATURE ETKİSİ (Partner seçiminde!)
        temp = 0.5
        if self.nature_thermostat:
            temp = self.nature_thermostat.temperature
        
        # Sıcaklık yüksekse → Güçlüye yönelim artar (Hayatta kalma içgüdüsü!)
        # Sıcaklık düşükse → Rastgelelik artar (Rahatlık)
        
        # Seçim stratejisi belirle (Rulet)
        rand = random.random()
        
        # Dinamik olasılıklar
        prob_strongest = self.base_partner_strongest + (temp - 0.5) * 0.4  # Sıcaksa artar
        prob_strongest = max(0.1, min(0.8, prob_strongest))
        
        prob_random = self.base_partner_random - (temp - 0.5) * 0.3 # Sıcaksa azalır
        prob_random = max(0.1, min(0.8, prob_random))
        
        if rand < prob_strongest:
            # En güçlüyle (Fitness)
            partner = max(candidates, key=lambda l: l.get_recent_fitness())
            return partner
            
        elif rand < (prob_strongest + prob_random):
            # Rastgele
            return random.choice(candidates)
            
        elif rand < (prob_strongest + prob_random + self.base_partner_complementary):
            # Tamamlayıcı (En farklı olan)
            return self._find_complementary(lora, candidates)
            
        else:
            # En zayıfla (Merhamet veya sömürü?)
            partner = min(candidates, key=lambda l: l.get_recent_fitness())
            return partner
    
    def _find_complementary(self, lora: LoRAAdapter, others: List[LoRAAdapter]) -> LoRAAdapter:
        """En farklı LoRA'yı bul"""
        best_partner = others[0]
        max_dist = -1.0
        
        my_params = lora.temperament
        
        for other in others:
            dist = 0
            for k, v in my_params.items():
                dist += abs(v - other.temperament.get(k, 0.5))
            
            if dist > max_dist:
                max_dist = dist
                best_partner = other
                
        return best_partner
    
    def _param_distance(self, params1: Dict, params2: Dict) -> float:
        """İki LoRA arasındaki parametre mesafesi"""
        dist = 0.0
        count = 0
        
        for k in params1:
            if k in params2:
                d = abs(params1[k] - params2[k])
                dist += d
                count += 1
                
        return dist / max(1, count)
    
    def chaotic_crossover(self, parent1: LoRAAdapter, parent2: LoRAAdapter) -> LoRAAdapter:
        """
        KAOTİK ÇİFTLEŞME:
        - Her parametrede farklı gürültü
        - Bazen mega gürültü
        - Öngörülemez kombinasyonlar
        - KİŞİLİK genetik olarak geçer!
        """
        # Yeni LoRA oluştur
        child = LoRAAdapter(
            input_dim=parent1.fc1.in_features,
            hidden_dim=parent1.fc1.out_features,
            rank=parent1.rank,
            alpha=parent1.alpha,
            device=self.device
        )
        
        # İsimlendirme (Genetik soy takibi)
        gen = max(parent1.generation, parent2.generation) + 1
        child.name = f"LoRA_Gen{gen}_{child.id[:6]}"
        child.generation = gen
        child.birth_match = self.match_count
        child.parents = [parent1.id, parent2.id]
        
        # Parametreleri karıştır
        p1_params = parent1.get_all_lora_params()
        p2_params = parent2.get_all_lora_params()
        child_params = child.get_all_lora_params()
        
        for layer in ['fc1', 'fc2', 'fc3']:
            for matrix in ['lora_A', 'lora_B']:
                t1 = p1_params[layer][matrix]
                t2 = p2_params[layer][matrix]
                
                # 🌊 AKIŞKAN CROSSOVER RATE
                # Sabit 0.5 değil! Anneye mi babaya mı çekecek?
                # Fitness'ı yüksek olana çekme ihtimali artar!
                f1 = parent1.get_recent_fitness()
                f2 = parent2.get_recent_fitness()
                total_f = f1 + f2 + 1e-6
                p1_ratio = f1 / total_f
                
                # Maske oluştur (Hangi gen kimden?)
                mask = (torch.rand_like(t1) < p1_ratio).float()
                
                # Karışım
                mixed = (t1 * mask) + (t2 * (1 - mask))
                
                # Gürültü ekle (Mutasyon)
                noise_scale = random.uniform(0.01, self.crossover_noise_max)
                
                # Mega gürültü şansı?
                if random.random() < self.mega_noise_chance:
                    noise_scale *= 3.0  # Şok değişim!
                
                noise = torch.randn_like(mixed) * noise_scale
                mixed += noise
                
                child_params[layer][matrix] = mixed
        
        child.set_all_lora_params(child_params)
        
        # Mizaç aktarımı
        child.temperament = self._inherit_temperament(parent1, parent2)
        
        return child
    
    def _inherit_temperament(self, parent1: LoRAAdapter, parent2: LoRAAdapter) -> Dict:
        """
        Anne + Baba kişiliklerinden çocuk kişiliği oluştur
        
        GENETİK MANTIK:
        - %50 anneden, %50 babadan (ortalama)
        - ±%20 mutasyon (yeni varyasyon!)
        - Nadir: Tam yeni kişilik (%5 şans)
        """
        # Nadir: Tamamen yeni kişilik (Alien geni!)
        if random.random() < 0.05:
            return self.spawn_random_lora(device=self.device).temperament
            
        child_temp = {}
        
        for trait in parent1.temperament.keys():
            p1_val = parent1.temperament.get(trait, 0.5)
            p2_val = parent2.temperament.get(trait, 0.5)
            
            # Ortalama
            avg = (p1_val + p2_val) / 2
            
            # ±%20 mutasyon
            mutation = random.uniform(-0.2, 0.2)
            final_val = avg + mutation
            
            # 0-1 arasında sınırla
            final_val = max(0.0, min(1.0, final_val))
            
            child_temp[trait] = final_val
        
        return child_temp
    
    def mutate(self, lora: LoRAAdapter):
        """
        MUTASYON:
        - %15 her parametre mutasyona uğrayabilir
        - %5 şok mutasyon (tamamen yeni değer)
        """
        params = lora.get_all_lora_params()
        
        # 🌡️ TEMPERATURE ETKİSİ (Mutasyonda!)
        temp = 0.5
        if self.nature_thermostat:
            temp = self.nature_thermostat.temperature
        
        # Sıcak (0.8) → Mutasyon artar (x1.3) → Adaptasyon zorlanır
        # Soğuk (0.2) → Mutasyon azalır (x0.7) → Stabilite
        temp_modifier = 1.0 + ((temp - 0.5) * 1.0)  # 0.5-1.5 arası
        
        for layer in ['fc1', 'fc2', 'fc3']:
            for matrix in ['lora_A', 'lora_B']:
                param = params[layer][matrix]
                
                # 🌊 DİNAMİK MUTASYON ŞANSI (Genetic diversity'ye göre!)
                genetic_diversity = self._calculate_genetic_diversity()
                
                # Diversity düşük → Daha fazla mutasyon (radikal değişim!)
                # Diversity yüksek → Daha az mutasyon (stabil)
                fluid_param_mutation = self.base_param_mutation_chance * (1.8 - genetic_diversity) * temp_modifier
                fluid_shock_mutation = self.base_shock_mutation_chance * (2.0 - genetic_diversity) * temp_modifier
                # Diversity 0 → %27 normal, %10 shock (radikal!)
                # Diversity 1 → %12 normal, %5 shock (stabil)
                
                # Normal mutasyon
                if random.random() < fluid_param_mutation:
                    mutation_strength = random.uniform(0.01, 0.3)
                    noise = torch.randn_like(param) * mutation_strength
                    param += noise
                
                # ŞOK MUTASYON (🌊 DİNAMİK!)
                if random.random() < fluid_shock_mutation:
                    param = torch.randn_like(param) * 0.5
                
                params[layer][matrix] = param
        
        lora.set_all_lora_params(params)
    
    def spawn_random_lora(self, device='cpu') -> LoRAAdapter:
        """
        Spontane doğum: Hiçlikten bir LoRA doğar! 👽
        
        Alien LoRA'lar genelde FARKLI kişilik yapısına sahiptir!
        """
        lora = LoRAAdapter(input_dim=78, hidden_dim=128, rank=16, alpha=16.0, device=device)  # __init__ içinde .to(device) çağrılıyor
        lora.name = f"LoRA_Alien_{lora.id}"
        lora.generation = 0
        lora.birth_match = self.match_count
        lora.parents = []
        
        # 👽 ALIEN KİŞİLİK: Daha ekstrem değerler!
        lora.temperament = {
            'independence': random.uniform(0.7, 1.0),        # Çok bağımsız!
            'social_intelligence': random.uniform(0.0, 0.5), # Sosyal zeka düşük
            'herd_tendency': random.uniform(0.0, 0.3),       # Sürüye uymaz!
            'contrarian_score': random.uniform(0.5, 1.0),    # Çok karşıt!
            'confidence_level': random.uniform(0.6, 1.0),    # Aşırı özgüvenli
            'risk_appetite': random.uniform(0.7, 1.0),       # Risk sever!
            'patience': random.uniform(0.1, 0.5),            # Sabırsız
            'impulsiveness': random.uniform(0.6, 1.0),       # Dürtüsel
            'stress_tolerance': random.uniform(0.3, 0.8)
        }
        
        return lora
    
    def evolution_step(self, alarm_info: Dict = None):
        """
        HER MAÇ SONRASI: Evrim adımı
        - Ölümler (fitness < threshold)
        - Üremeler (fitness > threshold + şans) - ALARM'a göre artar!
        - Spontane doğumlar
        
        Args:
            alarm_info: Popülasyon alarm bilgisi (soy azalırsa üreme artar!)
        """
        events = []
        
        # ⚠️ ALARM ÇARPANI (Soy azalırsa üreme artar!)
        repro_multiplier = 1.0
        if alarm_info:
            repro_multiplier = alarm_info.get('reproduction_multiplier', 1.0)
            if alarm_info['level'] != 'GREEN':
                print(f"\n⚠️ ALARM: {alarm_info['message']}")
                print(f"   Üreme şansı: x{repro_multiplier:.1f}")
        
        # 🌊 AKIŞKAN ÜREME ŞANSI (Genetic diversity'ye göre hesapla!)
        genetic_diversity = self._calculate_genetic_diversity()
        
        # Diversity düşük → Daha fazla üreme
        fluid_base_chance = self.base_reproduction_chance * (1.5 - (genetic_diversity * 0.8))
        
        # Alarm multiplier ile çarp
        reproduction_chance = fluid_base_chance * repro_multiplier
        # 🌊 LİMİT YOK! İhtiyaç varsa %200+ bile olabilir!
        
        # 1) ÖLÜMLER (FİZİK BAZLI! - LIFE ENERGY!)
        survivors = []
        for lora in self.population:
            fitness = lora.get_recent_fitness()
            
            # ⚡ LIFE ENERGY KONTROLÜ! (Fizik bazlı ölüm!)
            life_energy = getattr(lora, 'life_energy', 1.0)
            
            # FİZİK BAZLI ÖLÜM:
            # Life energy <= 0 → DOĞAL ÖLÜM! (Sönümlenme!)
            if life_energy <= 0:
                death_reason = f"Yaşam enerjisi tükendi (Energy: {life_energy:.3f})"
                
                events.append({
                    'type': 'death',
                    'lora': lora.name,
                    'lora_obj': lora,
                    'fitness': fitness,
                    'age': self.match_count - lora.birth_match,
                    'death_reason': death_reason,
                    'death_type': 'natural_energy_depletion'  # Fizik bazlı!
                })
                continue  # Bu LoRA öldü
            
            # 🛡️ YEDEK: Klasik threshold kontrolü (Life energy > 0 ama fitness çok düşük)
            if self.experience_resistance:
                dynamic_threshold = self.experience_resistance.calculate_death_threshold(
                    lora, base_threshold=self.death_threshold
                )
            else:
                dynamic_threshold = self.death_threshold
            
            if fitness < dynamic_threshold and life_energy < 0.5:  # Hem fitness hem energy düşük!
                # Ölmesi lazım, ama şanslı kurtuluş!
                if random.random() < self.lucky_survival_chance:
                    survivors.append(lora)
                    
                    # 🍀 Şanslı kurtuluş sayacını artır!
                    if not hasattr(lora, 'lucky_survivals'):
                        lora.lucky_survivals = 0
                    lora.lucky_survivals += 1
                    
                    # 🛡️ KRİTİK DURUMDAN KURTULDU! DİRENÇ KAZAN!
                    if self.experience_resistance:
                        self.experience_resistance.add_critical_survival(lora.id, fitness)
                    
                    events.append({
                        'type': 'lucky_survival',
                        'lora': lora.name,
                        'lora_obj': lora,
                        'fitness': fitness,
                        'survival_count': lora.lucky_survivals,
                        'dynamic_threshold': dynamic_threshold  # Dinamik eşik kaydet!
                    })
                else:
                    # ÖLÜM SEBEBİ BELİRLE (YAŞ DAHİL!)
                    death_reason = self._determine_death_reason(lora, fitness, current_match=self.match_count)
                    
                    events.append({
                        'type': 'death',
                        'lora': lora.name,
                        'lora_obj': lora,  # ✅ LoRA objesini de kaydet (mucize kontrolü için!)
                        'fitness': fitness,
                        'age': self.match_count - lora.birth_match,
                        'death_reason': death_reason  # 💀 ÖLÜM SEBEBİ!
                    })
            else:
                survivors.append(lora)
        
        self.population = survivors
        
        # 2) ÜREMELER (DOĞAL SİSTEM - BAĞ BAZLI!)
        new_borns = []
        for lora in self.population:
            fitness = lora.get_recent_fitness()
            
            # DOĞAL ÜREME ŞANSI HESAPLA
            natural_reproduction_chance = self._calculate_natural_reproduction_chance(
                lora, 
                len(self.population),
                alarm_info
            )
            
            # Fitness yeterli + Doğal şans
            if fitness > self.reproduction_threshold:
                if random.random() < natural_reproduction_chance:
                    partner = self.select_partner(lora)
                    
                    if partner is not None:
                        # Çiftleşme! (Deep Neural Crossover)
                        # Artık LoRA'nın kendi crossover metodunu kullanıyoruz!
                        child = lora.crossover(partner)
                        
                        # Ebeveyn çocuk sayısını artır
                        if not hasattr(lora, 'children_count'): lora.children_count = 0
                        if not hasattr(partner, 'children_count'): partner.children_count = 0
                        lora.children_count += 1
                        partner.children_count += 1
                        
                        # Mutasyon şansı
                        if random.random() < 0.3:  # %30 çocuklar mutasyona uğrar
                            self.mutate(child)
                        
                        new_borns.append(child)
                        events.append({
                            'type': 'birth',
                            'child': child.name,
                            'parent1': lora.name,
                            'parent2': partner.name,
                            'generation': child.generation
                        })
        
        self.population.extend(new_borns)
        
        # 3) SPONTANE DOĞUM (Alien LoRA!) (🌊 DİNAMİK!)
        
        # 🌊 DİNAMİK ALIEN ŞANSI (Genetic diversity'ye göre!)
        genetic_diversity = self._calculate_genetic_diversity()
        
        # Diversity düşük → Daha fazla alien (yeni gen havuzu lazım!)
        fluid_alien_chance = self.base_mutation_chance * (2.5 - (genetic_diversity * 1.5))
        
        if random.random() < fluid_alien_chance:
            alien = self.spawn_random_lora(device=self.device)
            self.population.append(alien)
            events.append({
                'type': 'spontaneous_birth',
                'lora': alien.name,
                'genetic_diversity': genetic_diversity,
                'alien_chance': fluid_alien_chance
            })
        
        # 4) SOY TÜKENMESİ KONTROLÜ
        # MANUEL DİRİLTME! Otomatik spawn YOK!
        if len(self.population) == 0:
            print(f"\n{'💀'*40}")
            print(f"💀 SOY TÜKENDİ! TÜM LoRA'LAR ÖLDÜ!")
            print(f"{'💀'*40}")
            print(f"\n⚡ DİRİLTME KOMUTU:")
            print(f"   python run_evolutionary_learning.py --resurrect")
            print(f"\n📚 ORTAK HAFIZA KORUNDU! Bilgi kaybolmadı!")
            print(f"{'💀'*40}\n")
            
            # Evrim durdur (diriltme bekle)
            events.append({
                'type': 'extinction',
                'message': 'Soy tükendi, diriltme bekleniyor'
            })
        
        # ❌ ÜST LİMİT YOK! Doğa kendi dengesini kuracak!
        
        return events
    
    def post_match_update(self, alarm_info: Dict = None):
        """
        Her maç sonrası çağrılır
        
        Args:
            alarm_info: Popülasyon alarm bilgisi (soy azalırsa üreme artar!)
        """
        self.match_count += 1
        
        # Her maçta evrim adımı (alarm bilgisiyle!)
        events = self.evolution_step(alarm_info=alarm_info)
        
        if len(events) > 0:
            self.evolution_log.append({
                'match': self.match_count,
                'population': len(self.population),
                'events': events
            })
        
        return events
    
    def get_population_stats(self) -> Dict:
        """Popülasyon istatistikleri"""
        if len(self.population) == 0:
            return {}
        
        fitnesses = [lora.get_recent_fitness() for lora in self.population]
        generations = [lora.generation for lora in self.population]
        ages = [self.match_count - lora.birth_match for lora in self.population]
        
        return {
            'size': len(self.population),
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'avg_generation': np.mean(generations),
            'max_generation': np.max(generations),
            'avg_age': np.mean(ages),
            'match_count': self.match_count
        }
    
    def print_status(self):
        """Durum yazdır"""
        stats = self.get_population_stats()
        
        print(f"\n{'='*60}")
        print(f"🌪️ KAOS EVRİM DURUMU (Maç: {stats.get('match_count', 0)})")
        print(f"{'='*60}")
        print(f"  Popülasyon: {stats.get('size', 0)}")
        print(f"  Avg Fitness: {stats.get('avg_fitness', 0):.3f}")
        print(f"  Min/Max Fitness: {stats.get('min_fitness', 0):.3f} / {stats.get('max_fitness', 0):.3f}")
        print(f"  Avg Generation: {stats.get('avg_generation', 0):.1f}")
        print(f"  Max Generation: {stats.get('max_generation', 0)}")
        print(f"  Avg Age: {stats.get('avg_age', 0):.1f} maç")
        print(f"{'='*60}\n")

