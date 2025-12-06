"""
🎯 GELİŞMİŞ MEKANİKLER
=======================

1. Elit Direnci (Zırh sistemi)
2. Sağ Kalan Sendromu (Survivor's Guilt)
3. Kan Uyuşmazlığı (Anti-Inbreeding)
4. Kış Uykusu (Hibernation)
5. Pozitif Geri Besleme Freni (Cooldown)
"""

import numpy as np
import torch
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class SurvivorGuilt:
    """Sağ kalan sendromu"""
    survived_event: str      # 'kara_veba', 'mass_extinction', etc.
    match: int
    guilt_severity: float    # 0-1 arası
    fitness_penalty: float   # Fitness kaybı
    trauma_gain: float       # Travma artışı


class EliteResistance:
    """
    ELİT DİRENCİ (Zırh Sistemi)
    
    Elite LoRA'lar = Top 50 listesinde VEYA Mucize olanlar
    
    Elite'ler zırh kazanır.
    Ama %100 koruma YOK! Max %60 korumalı.
    """
    
    @staticmethod
    def is_elite(lora, all_loras_ever: dict = None, miracle_system = None) -> bool:
        """
        LoRA elite mi kontrol et
        
        Elite = Top 50'de VEYA Mucize
        
        Args:
            lora: LoRAAdapter
            all_loras_ever: Tüm zamanlar LoRA kayıtları
            miracle_system: MiracleSystem instance
        
        Returns:
            True if elite, False otherwise
        """
        # Mucize mi kontrol et
        if miracle_system:
            try:
                miracle_list = miracle_system.get_all_miracle_ids()
                if lora.id in miracle_list:
                    return True  # 🏆 MUCİZE = ELİTE!
            except:
                pass
        
        # Top 50'de mi kontrol et
        if all_loras_ever:
            # Fitness'a göre sırala
            sorted_loras = sorted(
                all_loras_ever.items(),
                key=lambda x: x[1].get('final_fitness', 0),
                reverse=True
            )
            
            top_50_ids = [lora_id for lora_id, _ in sorted_loras[:50]]
            if lora.id in top_50_ids:
                return True  # ⭐ TOP 50 = ELİTE!
        
        return False  # Elite değil
    
    @staticmethod
    def calculate_armor(fitness: float, is_elite: bool = False) -> float:
        """
        Zırh hesapla
        
        Args:
            fitness: 0-1 arası
            is_elite: Elite mi? (Top 50 veya Mucize)
        
        Returns:
            armor: 0-0.60 arası (Max %60 koruma)
        """
        # Elite değilse sıradan hesaplama
        if not is_elite:
            if fitness < 0.50:
                return 0.0  # Zayıflar korumasız
            # Linear scaling: 0.50 → 0%, 1.00 → 60%
            armor = (fitness - 0.50) * 1.2  # 0.50 fark × 1.2 = 0.60 max
            return min(armor, 0.60)  # Asla %60'ı geçmez
        else:
            # 🏆 ELITE BONUS! +20% zırh
            base_armor = max(0, (fitness - 0.50) * 1.2)
            elite_bonus = 0.20  # +20% bonus
            total_armor = base_armor + elite_bonus
            return min(total_armor, 0.60)  # Yine max %60
    
    @staticmethod
    def should_survive_with_armor(fitness: float) -> Tuple[bool, Optional[SurvivorGuilt]]:
        """
        Zırh ile hayatta kalacak mı?
        
        Returns:
            (survived, guilt)
        """
        armor = EliteResistance.calculate_armor(fitness)
        
        # Zırh şansı
        if random.random() < armor:
            # ZIRH KORUDI! Ama bedeli var...
            
            # Sağ kalan sendromu
            guilt = SurvivorGuilt(
                survived_event='disaster_with_armor',
                match=0,  # Dışarıdan set edilecek
                guilt_severity=random.uniform(0.4, 0.8),
                fitness_penalty=armor * 0.3,  # Zırh ne kadar güçlüyse, suçluluk o kadar ağır
                trauma_gain=armor * 0.5
            )
            
            return True, guilt
        else:
            # Zırh yetmedi, öldü
            return False, None


class AntiInbreeding:
    """
    KAN UYUŞMAZLIĞI (Anti-Inbreeding)
    
    Genetik darboğazı önler.
    Çok benzer LoRA'lar çiftleşemez.
    """
    
    @staticmethod
    def calculate_genetic_similarity(lora1, lora2) -> float:
        """
        İki LoRA'nın genetik benzerliği
        
        Returns:
            similarity: 0-1 arası (1 = aynı)
        """
        params1 = lora1.get_all_lora_params()
        params2 = lora2.get_all_lora_params()
        
        similarities = []
        
        for layer in ['fc1', 'fc2', 'fc3']:
            for matrix in ['lora_A', 'lora_B']:
                p1 = params1[layer][matrix].flatten()
                p2 = params2[layer][matrix].flatten()
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    p1.unsqueeze(0), 
                    p2.unsqueeze(0)
                ).item()
                
                similarities.append(cos_sim)
        
        return np.mean(similarities)
    
    @staticmethod
    def can_mate(lora1, lora2, threshold: float = 0.95) -> Tuple[bool, str]:
        """
        Çiftleşebilirler mi?
        
        Returns:
            (can_mate, reason)
        """
        similarity = AntiInbreeding.calculate_genetic_similarity(lora1, lora2)
        
        if similarity > threshold:
            return False, f"Genetik benzerlik çok yüksek (%{similarity*100:.1f})"
        
        return True, "Uyumlu"
    
    @staticmethod
    def force_mutant_birth(lora1, lora2):
        """
        Çok benzer ebeveynlerden %100 mutant çocuk doğar
        (Genetik çeşitliliği koru!)
        """
        from .lora_adapter import LoRAAdapter
        
        # Tamamen yeni LoRA (mutant)
        # 🔧 DOĞRU BOYUT! (63 = 60 features + 3 base_proba)
        mutant = LoRAAdapter(input_dim=78, hidden_dim=128, rank=16, alpha=16.0, device=device)
        mutant.name = f"LoRA_MUTANT_{mutant.id}"
        mutant.generation = max(lora1.generation, lora2.generation) + 1
        mutant.parents = [lora1.id, lora2.id]
        mutant.is_mutant = True  # Özel işaret
        
        return mutant


class Hibernation:
    """
    KIŞ UYKUSU (Hibernation)
    
    Aktif olmayan LoRA'ları diske yaz, RAM'den sil.
    Gerektiğinde geri yükle.
    """
    
    def __init__(self, hibernate_dir: str = "hibernated_loras"):
        self.hibernate_dir = hibernate_dir
        os.makedirs(hibernate_dir, exist_ok=True)
        
        self.hibernated_loras: Dict[str, str] = {}  # lora_id -> file_path
    
    def check_and_hibernate(self, population: List, attention_weights: List, match_idx: int = 0) -> int:
        """
        Popülasyonu kontrol et, uyutulması gerekenleri uyut
        
        ⚠️ KRİTİK KURAL: ASLA %20'den fazla uyumasın!
        Amaç: Evrim ve gelişim, yük taşımak değil!
        
        Args:
            population: LoRA listesi
            attention_weights: Meta-LoRA ağırlıkları
            match_idx: Maç numarası (debug için)
        
        Returns:
            Uyutulan LoRA sayısı
        """
        population_size = len(population)
        total_population = population_size + len(self.hibernated_loras)  # Aktif + Uyuyan
        
        # 🔍 DEBUG: Mevcut durum
        current_hibernated_ratio = len(self.hibernated_loras) / total_population if total_population > 0 else 0
        max_allowed_hibernated = int(total_population * 0.20)  # %20 limit!
        current_hibernated = len(self.hibernated_loras)
        remaining_hibernation_slots = max(0, max_allowed_hibernated - current_hibernated)
        
        if match_idx % 10 == 0:
            print(f"\n   😴 HİBERNATION DEBUG (Maç #{match_idx}):")
            print(f"      • Toplam Popülasyon: {total_population} (Aktif: {population_size}, Uyuyan: {current_hibernated})")
            print(f"      • Şu An Uyuma Oranı: {current_hibernated_ratio*100:.1f}%")
            print(f"      • Maksimum İzin: 20.0% ({max_allowed_hibernated} LoRA)")
            print(f"      • Kalan Slot: {remaining_hibernation_slots} LoRA")
        
        if population_size <= 100:
            if match_idx % 10 == 0:
                print(f"      ⏸️  Uyutma yapılmıyor (Nüfus ≤ 100)")
            return 0  # Nüfus az, uyutma yapma
        
        # ⚠️ LİMİT KONTROLÜ! (%20'den fazla uyumasın!)
        if current_hibernated >= max_allowed_hibernated:
            if match_idx % 10 == 0:
                print(f"      🛑 LİMİT AŞILDI! Uyutma DURDURULDU (%20 limit: {max_allowed_hibernated})")
            return 0  # Limit aşıldı, daha fazla uyutma!
        
        # Uyutulabilir adayları bul
        candidates = []
        for i, lora in enumerate(population):
            weight = attention_weights[i] if i < len(attention_weights) else 0.0
            if self.should_hibernate(lora, weight, population_size):
                candidates.append((lora, weight))
        
        # Adayları fitness'a göre sırala (düşük fitness önce uyusun)
        candidates.sort(key=lambda x: x[0].get_recent_fitness())
        
        # Limit'e kadar uyut!
        hibernated_count = 0
        to_remove = []
        
        for lora, weight in candidates:
            if hibernated_count >= remaining_hibernation_slots:
                break  # Limit doldu!
            
            # UYUT!
            file_path = self.hibernate_lora(lora)
            to_remove.append(lora)
            hibernated_count += 1
        
        # Popülasyondan çıkar
        for lora in to_remove:
            population.remove(lora)
        
        if hibernated_count > 0:
            new_ratio = (current_hibernated + hibernated_count) / total_population
            print(f"   😴 {hibernated_count} LoRA uyutuldu (Yeni oran: {new_ratio*100:.1f}%)")
            if new_ratio > 0.18:
                print(f"      ⚠️  UYARI: Limit yakın! (%20'ye yaklaşıyor)")
        
        return hibernated_count
    
    def should_hibernate(self, lora, meta_attention_weight: float, 
                         population_size: int) -> bool:
        """
        Bu LoRA uyumalı mı?
        
        Kriterler:
        - Nüfus > 100
        - Meta-LoRA düşük ağırlık veriyor (< %2)
        - Fitness orta (0.40-0.60, çok kötü değil ama iyi de değil)
        """
        if population_size <= 100:
            return False  # Nüfus az, uyutma
        
        if meta_attention_weight > 0.02:
            return False  # Aktif kullanılıyor
        
        fitness = lora.get_recent_fitness()
        if fitness < 0.40 or fitness > 0.70:
            return False  # Çok kötü veya çok iyi, uyutma
        
        # Orta şeker LoRA, nüfus fazla, kullanılmıyor → UYUT!
        return True
    
    def hibernate_lora(self, lora) -> str:
        """
        LoRA'yı diske yaz, RAM'den sil
        
        Returns:
            file_path
        """
        file_path = os.path.join(self.hibernate_dir, f"{lora.name}.pt")
        
        # Tüm durumu kaydet (TES + Energy!)
        state = {
            'params': lora.get_all_lora_params(),
            'metadata': {
                'id': lora.id,
                'name': lora.name,
                'generation': lora.generation,
                'birth_match': lora.birth_match,
                'fitness_history': lora.fitness_history,
                'specialization': getattr(lora, 'specialization', None),
                'parents': lora.parents,
                'life_energy': getattr(lora, 'life_energy', 1.0),  # ⚡ Energy!
                'temperament': getattr(lora, 'temperament', {}),
                '_last_kl': getattr(lora, '_last_kl', 0.0)  # Einstein için
            }
        }
        
        torch.save(state, file_path)
        
        self.hibernated_loras[lora.id] = file_path
        
        return file_path
    
    def wake_up_lora(self, lora_id: str, device='cuda'):
        """
        LoRA'yı diskten yükle
        
        Args:
            device: Hedef device (CUDA/CPU)
        
        Returns:
            LoRA instance
        """
        if lora_id not in self.hibernated_loras:
            return None
        
        file_path = self.hibernated_loras[lora_id]
        
        if not os.path.exists(file_path):
            return None
        
        # Yükle (CPU'ya)
        state = torch.load(file_path, map_location='cpu')
        
        from .lora_adapter import LoRAAdapter
        # 🔧 DOĞRU DEVICE VE BOYUT! (63 = 60 features + 3 base_proba)
        lora = LoRAAdapter(input_dim=78, hidden_dim=128, rank=16, alpha=16.0, device=device)
        lora.set_all_lora_params(state['params'])
        
        # Metadata'yı geri yükle
        for key, value in state['metadata'].items():
            setattr(lora, key, value)
        
        # ⚡ UYURKEN ENERJİ AZALIR! (Yavaş tükenme!)
        # Her maç: -0.01 energy
        # Eğer çok uyuduysa enerji düşmüş olabilir!
        # (Şimdilik sadece yükle, sonra pasif tükenme ekleriz)
        
        # Hibernated listesinden çıkar
        del self.hibernated_loras[lora_id]
        
        return lora
    
    def get_hibernation_stats(self) -> Dict:
        """Uyku durumu istatistikleri"""
        return {
            'hibernated_count': len(self.hibernated_loras),
            'hibernated_loras': list(self.hibernated_loras.keys())
        }
    
    def wake_up_best_hibernated(self, population: List, target_count: int, device='cpu') -> List:
        """
        ⏰ AKILLI UYANMA: En iyi uyuyanları uyandır!
        
        Popülasyon düşükse veya güçlendirme gerekiyorsa kullan.
        
        Args:
            population: Mevcut popülasyon (kontrol için)
            target_count: Kaç LoRA uyandırılacak?
            device: CUDA/CPU
        
        Returns:
            Uyanan LoRA listesi
        """
        if len(self.hibernated_loras) == 0:
            print("   ⚠️ Uyuyan LoRA yok!")
            return []
        
        # Tüm uyuyanları fitness'a göre sırala
        hibernated_data = []
        
        for lora_id, file_path in self.hibernated_loras.items():
            if not os.path.exists(file_path):
                continue
            
            # Metadata'yı oku (fitness için)
            try:
                state = torch.load(file_path, map_location='cpu')
                metadata = state.get('metadata', {})
                fitness_history = metadata.get('fitness_history', [])
                
                if len(fitness_history) > 0:
                    recent_fitness = np.mean(fitness_history[-50:])
                else:
                    recent_fitness = 0.5
                
                hibernated_data.append({
                    'lora_id': lora_id,
                    'file_path': file_path,
                    'fitness': recent_fitness,
                    'name': metadata.get('name', 'Unknown')
                })
            except Exception as e:
                print(f"   ⚠️ {lora_id} yüklenemedi: {e}")
                continue
        
        # Fitness'a göre sırala (en iyi önce)
        hibernated_data.sort(key=lambda x: x['fitness'], reverse=True)
        
        # İlk N tanesini uyandır
        awakened = []
        wake_count = min(target_count, len(hibernated_data))
        
        print(f"\n⏰ AKILLI UYANMA BAŞLIYOR!")
        print(f"   📊 Uyuyan LoRA: {len(hibernated_data)}")
        print(f"   🎯 Hedef: {wake_count} LoRA uyandır")
        print(f"   💤 En iyi {wake_count} uyuyan seçiliyor...")
        
        for i in range(wake_count):
            data = hibernated_data[i]
            lora_id = data['lora_id']
            
            # Uyandır! (🔧 DOĞRU DEVICE'DA!)
            lora = self.wake_up_lora(lora_id, device=device)
            
            if lora:
                # Artık doğru device'da yaratılıyor, .to(device) gereksiz
                awakened.append(lora)
                print(f"   ⏰ {data['name']} uyandı! (Fitness: {data['fitness']:.3f})")
        
        print(f"\n✅ {len(awakened)} LoRA uyandırıldı!")
        
        return awakened
    
    def should_wake_up_loras(self, population_size: int, threshold: int = 40) -> Tuple[bool, int]:
        """
        LoRA'lar uyandırılmalı mı? (BASIT VERSİYON)
        
        Args:
            population_size: Mevcut popülasyon
            threshold: Eşik değer (altındaysa uyandır)
        
        Returns:
            (should_wake, how_many)
        """
        if len(self.hibernated_loras) == 0:
            return False, 0  # Uyuyan yok
        
        if population_size >= threshold:
            return False, 0  # Popülasyon yeterli
        
        # Kaç LoRA uyandırılmalı?
        deficit = threshold - population_size
        wake_count = min(deficit, len(self.hibernated_loras))
        
        return True, wake_count
    
    def intelligent_wake_up(self, population: List, match_data: dict = None, 
                           attention_weights: List = None, recent_disaster: bool = False) -> Tuple[List, str]:
        """
        ⏰ AKILLI UYANMA SİSTEMİ (5 FAKTÖR!)
        
        En zeki sistem: Her duruma göre en uygun LoRA'ları uyandır!
        
        FAKTÖRLER:
        1. POPÜLASYON DÜŞÜKLÜĞÜ (< 40)
        2. UZMAN EKSİKLİĞİ (Pattern bazlı)
        3. META-LoRA DİKKAT DAĞILIMI (Herkes eşit → yeni kan lazım)
        4. DOĞAL FELAKET SONRASI (Güçlendirme)
        5. MİZAÇ DENGESİ (Çeşitlilik için)
        
        Args:
            population: Mevcut popülasyon
            match_data: Sıradaki maç verisi (pattern tespiti için)
            attention_weights: Meta-LoRA dikkat ağırlıkları
            recent_disaster: Son zamanda felaket oldu mu?
        
        Returns:
            (awakened_loras, reason)
        """
        if len(self.hibernated_loras) == 0:
            return [], "Uyuyan LoRA yok"
        
        population_size = len(population)
        wake_reasons = []
        total_wake_count = 0
        awakened = []
        
        print(f"\n🧠 AKILLI UYANMA ANALİZİ:")
        print(f"   📊 Popülasyon: {population_size}")
        print(f"   💤 Uyuyan: {len(self.hibernated_loras)}")
        
        # ============================================
        # FAKTÖR 1: POPÜLASYON DÜŞÜKLÜĞÜ
        # ============================================
        if population_size < 40:
            deficit = 40 - population_size
            wake_count_1 = min(deficit, len(self.hibernated_loras))
            total_wake_count += wake_count_1
            wake_reasons.append(f"Popülasyon düşük ({population_size}<40)")
            print(f"   ⚠️ FAKTÖR 1: Popülasyon düşük! +{wake_count_1} uyandır")
        
        # ============================================
        # FAKTÖR 2: UZMAN EKSİKLİĞİ
        # ============================================
        if match_data:
            # Pattern tespiti (basit)
            is_derby = match_data.get('is_derby', False) if isinstance(match_data, dict) else False
            high_hype = match_data.get('total_tweets', 0) > 50000 if isinstance(match_data, dict) else False
            
            # Popülasyonda uzman var mı kontrol et
            has_specialist = False
            for lora in population:
                spec = getattr(lora, 'specialization', None)
                if is_derby and spec == 'derbi_expert':
                    has_specialist = True
                    break
                if high_hype and spec == 'hype_expert':
                    has_specialist = True
                    break
            
            if (is_derby or high_hype) and not has_specialist:
                wake_count_2 = min(2, len(self.hibernated_loras))
                total_wake_count += wake_count_2
                pattern_type = "Derbi" if is_derby else "Hype"
                wake_reasons.append(f"{pattern_type} uzmanı gerekli")
                print(f"   🎯 FAKTÖR 2: {pattern_type} uzmanı lazım! +{wake_count_2} uyandır")
        
        # ============================================
        # FAKTÖR 3: META-LoRA DİKKAT DAĞILIMI
        # ============================================
        if isinstance(attention_weights, (list, np.ndarray)) and len(attention_weights) > 0:
            # Dikkat eşit dağılmış mı? (Herkes eşit → yeni kan lazım!)
            attention_variance = np.var(attention_weights)
            
            if attention_variance < 0.01:  # Çok eşit dağılmış!
                wake_count_3 = min(3, len(self.hibernated_loras))
                total_wake_count += wake_count_3
                wake_reasons.append("Dikkat eşit dağılmış (yeni kan)")
                print(f"   🎲 FAKTÖR 3: Dikkat çok eşit! Yeni kan lazım! +{wake_count_3} uyandır")
        
        # ============================================
        # FAKTÖR 4: DOĞAL FELAKET SONRASI
        # ============================================
        if recent_disaster:
            wake_count_4 = min(5, len(self.hibernated_loras))
            total_wake_count += wake_count_4
            wake_reasons.append("Felaket sonrası güçlendirme")
            print(f"   🌪️ FAKTÖR 4: Felaket sonrası! Güçlendir! +{wake_count_4} uyandır")
        
        # ============================================
        # FAKTÖR 5: MİZAÇ DENGESİ
        # ============================================
        # Popülasyonda mizaç dağılımı dengesiz mi?
        if len(population) > 5:
            # Hırs ortalaması
            avg_ambition = np.mean([lora.temperament.get('ambition', 0.5) for lora in population])
            avg_resilience = np.mean([lora.temperament.get('resilience', 0.5) for lora in population])
            
            # Çok düşükse (< 0.4) veya çok yüksekse (> 0.7) dengesiz
            if avg_ambition < 0.4 or avg_ambition > 0.8:
                wake_count_5 = min(2, len(self.hibernated_loras))
                total_wake_count += wake_count_5
                wake_reasons.append("Mizaç dengesi bozuk")
                print(f"   🎭 FAKTÖR 5: Mizaç dengesiz! Çeşitlilik lazım! +{wake_count_5} uyandır")
        
        # ============================================
        # UYANMA KARAR
        # ============================================
        if total_wake_count == 0:
            print(f"   ✅ Uyanma gerekmiyor, sistem dengeli!")
            return [], "Sistem dengeli"
        
        # En iyi uyuyanları uyandır
        final_wake_count = min(total_wake_count, len(self.hibernated_loras))
        print(f"\n   🎯 TOPLAM: {final_wake_count} LoRA uyandırılacak")
        print(f"   📋 Sebepler: {', '.join(wake_reasons)}")
        
        # 🔧 DOĞRU DEVICE KULLAN! (Population'dan al)
        target_device = next(population[0].parameters()).device if len(population) > 0 else 'cuda'
        awakened = self.wake_up_best_hibernated(population, final_wake_count, device=target_device)
        
        reason_text = f"{final_wake_count} LoRA uyandırıldı: {', '.join(wake_reasons)}"
        
        return awakened, reason_text


class PositiveFeedbackBrake:
    """
    POZİTİF GERİ BESLEME FRENİ
    
    Doğanın sonsuz öfke döngüsüne girmesini engeller.
    Her felaketten sonra soğuma süresi.
    """
    
    def __init__(self, cooldown_matches: int = 20):
        self.cooldown_matches = cooldown_matches
        self.last_major_event_match = -1000  # Çok eskiden
        self.event_history = []
        
        # Doygunluk (Saturation)
        self.saturation_threshold = 3  # 20 maç içinde 3 olay → doygunluk
    
    def can_trigger_event(self, current_match: int, event_severity: float) -> Tuple[bool, str]:
        """
        Yeni olay tetiklenebilir mi?
        
        Returns:
            (can_trigger, reason)
        """
        # 1) SOĞUMA SÜRESİ
        matches_since_last = current_match - self.last_major_event_match
        
        if event_severity > 0.7:  # Büyük olaylar için
            if matches_since_last < self.cooldown_matches:
                return False, f"Soğuma süresi (son olaydan {matches_since_last} maç geçti, min {self.cooldown_matches})"
        
        # 2) DOYGUNLUK KONTROLÜ
        recent_events = [e for e in self.event_history 
                        if current_match - e['match'] < 20]
        
        major_recent = [e for e in recent_events if e['severity'] > 0.6]
        
        if len(major_recent) >= self.saturation_threshold:
            return False, f"Doğa doygunluğa ulaştı ({len(major_recent)} olay 20 maçta)"
        
        # 3) DOĞA ENERJİSİ
        # Çok fazla olay olduysa, doğanın enerjisi azalır
        total_severity = sum(e['severity'] for e in recent_events)
        
        if total_severity > 3.0:  # Toplam şiddet > 3
            return False, "Doğanın enerjisi tükendi, dinleniyor"
        
        return True, "Uygun"
    
    def register_event(self, match: int, event_type: str, severity: float):
        """Olayı kaydet"""
        
        self.event_history.append({
            'match': match,
            'type': event_type,
            'severity': severity
        })
        
        if severity > 0.7:
            self.last_major_event_match = match
        
        # Eski olayları temizle (son 100 maç)
        self.event_history = [e for e in self.event_history 
                             if match - e['match'] < 100]
    
    def get_nature_energy(self, current_match: int) -> float:
        """
        Doğanın mevcut enerjisi
        
        Returns:
            energy: 0-1 arası (1 = tam enerji)
        """
        recent_events = [e for e in self.event_history 
                        if current_match - e['match'] < 50]
        
        if len(recent_events) == 0:
            return 1.0  # Tam enerji
        
        # Son 50 maçtaki toplam şiddet
        total_severity = sum(e['severity'] for e in recent_events)
        
        # Her 1.0 şiddet = %20 enerji kaybı
        energy_loss = min(total_severity * 0.2, 0.9)  # Max %90 kayıp
        
        return 1.0 - energy_loss


class AdvancedMechanicsManager:
    """
    Tüm gelişmiş mekanikleri yönetir
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Alt sistemler
        self.elite_resistance = EliteResistance()
        self.anti_inbreeding = AntiInbreeding()
        self.hibernation = Hibernation()
        self.feedback_brake = PositiveFeedbackBrake(cooldown_matches=20)
        
        print("🎯 Gelişmiş Mekanikler başlatıldı:")
        print("   ✅ Elit Direnci (Max %60 zırh)")
        print("   ✅ Sağ Kalan Sendromu")
        print("   ✅ Kan Uyuşmazlığı (Anti-Inbreeding)")
        print("   ✅ Kış Uykusu (Hibernation)")
        print("   ✅ Pozitif Geri Besleme Freni (20 maç cooldown)")
    
    def apply_disaster_with_armor(self, population: List, kill_ratio: float, 
                                  event_type: str, match_num: int,
                                  all_loras_ever: dict = None,
                                  miracle_system = None) -> Tuple[List, List]:
        """
        Felaketi zırh ile uygula (ELİTE KONTROLÜ!)
        
        Elite = Top 50 veya Mucize
        
        Returns:
            (survivors, survivor_guilt_list)
        """
        survivors = []
        guilt_list = []
        
        for lora in population:
            fitness = lora.get_recent_fitness()
            
            # 🏆 ELİTE KONTROLÜ
            is_elite = EliteResistance.is_elite(lora, all_loras_ever, miracle_system)
            
            # Zırh hesapla (elite bonus dahil!)
            armor = EliteResistance.calculate_armor(fitness, is_elite=is_elite)
            lora.elite_armor = armor  # Kaydet (log için)
            lora.is_elite = is_elite  # Kaydet
            
            # Temel ölüm şansı
            death_chance = kill_ratio
            
            # Zırh ile ölüm şansı azalır (armor zaten hesaplandı!)
            death_chance_with_armor = death_chance * (1 - armor)
            
            # Ölüm testi
            if random.random() < death_chance_with_armor:
                # Öldü (zırh yetmedi veya yoktu)
                pass
            else:
                # Hayatta kaldı!
                survivors.append(lora)
                
                # Zırh kullanıldı mı?
                if armor > 0.0 and random.random() < kill_ratio:
                    # Normalde ölecekti ama zırh korudu!
                    # SAĞ KALAN SENDROMU
                    
                    guilt = SurvivorGuilt(
                        survived_event=event_type,
                        match=match_num,
                        guilt_severity=random.uniform(0.4, 0.8),
                        fitness_penalty=armor * 0.3,
                        trauma_gain=armor * 0.5
                    )
                    
                    guilt_list.append((lora, guilt))
        
        return survivors, guilt_list
    
    def apply_survivor_guilt(self, lora, guilt: SurvivorGuilt):
        """
        Sağ kalan sendromunun etkilerini uygula
        
        1. Fitness düşer (zayıfladı)
        2. Travma artar (arkadaşları öldü)
        3. Mizaç değişir (paranoyak, korkak)
        """
        # 1) FITNESS CEZASI
        # Son fitness değerlerini düşür
        if len(lora.fitness_history) > 0:
            for i in range(min(10, len(lora.fitness_history))):
                lora.fitness_history[-(i+1)] *= (1 - guilt.fitness_penalty)
        
        # 2) TRAVMA EKLE
        if not hasattr(lora, 'trauma_history'):
            lora.trauma_history = []
        
        from .nature_entropy_system import TraumaEvent
        trauma = TraumaEvent(
            type=f'survivor_guilt_{guilt.survived_event}',
            severity=guilt.guilt_severity,
            timestamp=guilt.match
        )
        lora.trauma_history.append(trauma)
        
        # 3) MİZAÇ DEĞİŞİMİ
        if hasattr(lora, 'temperament'):
            # Cesaret azalır
            if 'risk_appetite' in lora.temperament:
                lora.temperament['risk_appetite'] *= 0.7  # %30 azalır
            
            # Stres toleransı azalır
            if 'stres_toleransı' in lora.temperament:
                lora.temperament['stres_toleransı'] *= 0.8  # %20 azalır
            
            # Dürtüsellik azalır (daha temkinli)
            if 'dürtüsellik' in lora.temperament:
                lora.temperament['dürtüsellik'] *= 0.75  # %25 azalır
    
    def check_and_mate(self, lora1, lora2) -> Tuple[bool, Optional[object], str]:
        """
        Çiftleşme kontrolü + mutant doğum
        
        Returns:
            (can_mate, child_or_none, reason)
        """
        can_mate, reason = self.anti_inbreeding.can_mate(lora1, lora2, threshold=0.95)
        
        if can_mate:
            return True, None, reason
        
        # Çok benzerler! İki seçenek:
        if random.random() < 0.5:
            # İptal
            return False, None, reason
        else:
            # MUTANT DOĞUR!
            mutant = self.anti_inbreeding.force_mutant_birth(lora1, lora2)
            return True, mutant, "Mutant doğdu (genetik çeşitlilik koruması)"
    
    def manage_hibernation(self, population: List, meta_attention_weights: np.ndarray) -> Tuple[List, int]:
        """
        Hibernation yönetimi
        
        Returns:
            (active_population, hibernated_count)
        """
        population_size = len(population)
        
        if population_size <= 100:
            return population, 0  # Uyutma gerek yok
        
        active_population = []
        hibernated_count = 0
        
        for i, lora in enumerate(population):
            attention_weight = meta_attention_weights[i] if i < len(meta_attention_weights) else 0.0
            
            if self.hibernation.should_hibernate(lora, attention_weight, population_size):
                # UYUT!
                file_path = self.hibernation.hibernate_lora(lora)
                hibernated_count += 1
            else:
                active_population.append(lora)
        
        return active_population, hibernated_count
    
    def check_nature_event_allowed(self, current_match: int, event_severity: float) -> Tuple[bool, str]:
        """
        Doğa olayı tetiklenebilir mi? (Fren kontrolü)
        """
        return self.feedback_brake.can_trigger_event(current_match, event_severity)
    
    def register_nature_event(self, match: int, event_type: str, severity: float):
        """Doğa olayını kaydet (fren için)"""
        self.feedback_brake.register_event(match, event_type, severity)
    
    def get_nature_energy(self, current_match: int) -> float:
        """Doğanın mevcut enerjisi"""
        return self.feedback_brake.get_nature_energy(current_match)

