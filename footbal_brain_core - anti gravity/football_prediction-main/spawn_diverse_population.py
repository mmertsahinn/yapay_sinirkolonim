"""
🌪️ ÇOK ÇEŞİTLİ POPÜLASYON SPAWN SİSTEMİ
========================================

Her arketip kombinasyonu, uç değerler, maksimum çeşitlilik!
250 LoRA ile tamamen farklı bir başlangıç!

UÇ DEĞERLER:
- Çok düşük (0.0-0.2)
- Çok yüksek (0.8-1.0)
- Her kombinasyon!
"""

import os
import sys
import torch
import random
import numpy as np
from datetime import datetime

# Proje kök dizinine ekle
sys.path.insert(0, os.path.dirname(__file__))

from lora_system.lora_adapter import LoRAAdapter


class DiversePopulationSpawner:
    """
    Maksimum çeşitlilik ile LoRA spawner
    """
    
    def __init__(self, target_population: int = 250, device='cuda'):
        self.target_population = target_population
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 15 mizaç özelliği
        self.temperament_traits = [
            'independence', 'social_intelligence', 'herd_tendency', 'contrarian_score',
            'emotional_depth', 'empathy', 'anger_tendency',
            'ambition', 'competitiveness', 'resilience', 'will_to_live',
            'patience', 'impulsiveness', 'stress_tolerance', 'risk_appetite'
        ]
        
        print(f"🌪️ Diverse Population Spawner başlatıldı!")
        print(f"   Hedef: {target_population} LoRA")
        print(f"   Device: {self.device}")
    
    def generate_extreme_temperament(self, extreme_ratio: float = None):
        """
        🌊 AKIŞKAN UÇ DEĞER ÜRETME!
        
        Extreme_ratio sabit değil, her LoRA için rastgele!
        
        Args:
            extreme_ratio: None ise tamamen akışkan
        """
        temperament = {}
        
        # 🌊 AKIŞKAN EXTREME RATIO! (Her LoRA farklı!)
        if extreme_ratio is None:
            extreme_ratio = random.uniform(0.4, 0.9)  # %40-90 arası rastgele
        
        for trait in self.temperament_traits:
            # 🌊 AKIŞKAN UÇ DEĞER SEÇİMİ!
            # Her trait için farklı olasılık
            trait_extreme_prob = random.uniform(0.3, 1.0)
            
            if random.random() < trait_extreme_prob * extreme_ratio:
                # Uç değer: Sigmoid dağılım (0 veya 1'e yakın)
                if random.random() < 0.5:
                    # Çok düşük - Sigmoid (0'a yakın)
                    temperament[trait] = random.betavariate(0.5, 2)  # 0'a yakın eğilimli
                else:
                    # Çok yüksek - Sigmoid (1'e yakın)
                    temperament[trait] = random.betavariate(2, 0.5)  # 1'e yakın eğilimli
            else:
                # Normal değer - Gaussian dağılım
                temperament[trait] = np.clip(random.gauss(0.5, 0.15), 0.0, 1.0)
        
        return temperament
    
    def generate_archetype_focused(self, trait_groups):
        """
        🌊 AKIŞKAN ARKETİP ODAKLI!
        
        Sabit 0.85-1.0 gibi aralıklar YOK!
        Beta dağılımı ile doğal eğilimler
        """
        temperament = {}
        
        # Önce tüm özelliklere Gaussian dağılım
        for trait in self.temperament_traits:
            temperament[trait] = np.clip(random.gauss(0.5, 0.2), 0.0, 1.0)
        
        # Sonra odak özelliklerini Beta dağılımı ile ayarla
        for trait, level in trait_groups:
            if level == 'high':
                # Beta(2, 0.5): 1'e yakın eğilimli
                temperament[trait] = random.betavariate(3, 0.7)
            elif level == 'low':
                # Beta(0.5, 2): 0'a yakın eğilimli
                temperament[trait] = random.betavariate(0.7, 3)
            elif level == 'extreme_high':
                # Beta(5, 0.5): Çok 1'e yakın
                temperament[trait] = random.betavariate(5, 0.5)
            elif level == 'extreme_low':
                # Beta(0.5, 5): Çok 0'a yakın
                temperament[trait] = random.betavariate(0.5, 5)
        
        return temperament
    
    def spawn_diverse_population(self):
        """
        250 LoRA'lık çeşitli popülasyon spawn et!
        """
        print(f"\n{'='*80}")
        print(f"🌪️ ÇOK ÇEŞİTLİ POPÜLASYON SPAWN EDİLİYOR!")
        print(f"{'='*80}\n")
        
        population = []
        
        # 1) 🌊 TAMAMEN AKIŞKAN UÇ DEĞERLİ (50 LoRA)
        print("1️⃣ 🌊 Tamamen akışkan uç değerli LoRA'lar (50 adet)...")
        for i in range(50):
            lora = LoRAAdapter(
                input_dim=78,  # 🌊 Tarihsel veri dahil!
                hidden_dim=128,
                rank=16,
                alpha=16.0,
                device=self.device
            )
            lora.name = f"Fluid_Extreme_{i+1}"
            # 🌊 Her LoRA için farklı extreme_ratio! (Tam akışkan!)
            lora.temperament = self.generate_extreme_temperament(extreme_ratio=None)
            population.append(lora)
            
            if (i+1) % 10 == 0:
                print(f"   ✅ {i+1}/50 LoRA spawn edildi")
        
        # 2) Arketip odaklı kombinasyonlar (150 LoRA)
        print("\n2️⃣ Arketip odaklı kombinasyonlar (150 adet)...")
        
        archetype_combinations = [
            # Bağımsız + Contrarian
            [('independence', 'extreme_high'), ('contrarian_score', 'extreme_high'), ('herd_tendency', 'extreme_low')],
            # Sosyal + Empati
            [('social_intelligence', 'extreme_high'), ('empathy', 'extreme_high'), ('independence', 'low')],
            # Hırslı + Rekabetçi
            [('ambition', 'extreme_high'), ('competitiveness', 'extreme_high'), ('resilience', 'high')],
            # Sakin + Sabırlı
            [('patience', 'extreme_high'), ('stress_tolerance', 'extreme_high'), ('impulsiveness', 'extreme_low')],
            # Kaotik + Dürtüsel
            [('impulsiveness', 'extreme_high'), ('risk_appetite', 'extreme_high'), ('patience', 'extreme_low')],
            # Duygusal + Derin
            [('emotional_depth', 'extreme_high'), ('empathy', 'extreme_high'), ('anger_tendency', 'high')],
            # Soğuk + Mantıklı
            [('emotional_depth', 'extreme_low'), ('independence', 'high'), ('patience', 'high')],
            # Sürü + Sosyal
            [('herd_tendency', 'extreme_high'), ('social_intelligence', 'high'), ('independence', 'extreme_low')],
            # Karşıt + Bağımsız
            [('contrarian_score', 'extreme_high'), ('independence', 'extreme_high'), ('herd_tendency', 'extreme_low')],
            # Dayanıklı + Kararlı
            [('resilience', 'extreme_high'), ('will_to_live', 'extreme_high'), ('stress_tolerance', 'extreme_high')],
            # Kırılgan + Hassas
            [('resilience', 'extreme_low'), ('emotional_depth', 'extreme_high'), ('stress_tolerance', 'extreme_low')],
            # Agresif + Hırslı
            [('anger_tendency', 'extreme_high'), ('ambition', 'extreme_high'), ('competitiveness', 'extreme_high')],
            # Pasif + Uyumlu
            [('herd_tendency', 'high'), ('patience', 'high'), ('competitiveness', 'extreme_low')],
            # Risk avcısı
            [('risk_appetite', 'extreme_high'), ('impulsiveness', 'high'), ('patience', 'extreme_low')],
            # Güvenli oyuncu
            [('risk_appetite', 'extreme_low'), ('patience', 'extreme_high'), ('stress_tolerance', 'high')],
        ]
        
        archetype_idx = 0
        for i in range(150):
            lora = LoRAAdapter(
                input_dim=78,  # 🌊 Tarihsel veri dahil!
                hidden_dim=128,
                rank=16,
                alpha=16.0,
                device=self.device
            )
            
            # Sırayla arketipleri kullan
            combo = archetype_combinations[archetype_idx % len(archetype_combinations)]
            lora.name = f"Archetype_{archetype_idx % len(archetype_combinations)}_{i+1}"
            lora.temperament = self.generate_archetype_focused(combo)
            population.append(lora)
            
            archetype_idx += 1
            
            if (i+1) % 30 == 0:
                print(f"   ✅ {i+1}/150 LoRA spawn edildi")
        
        # 3) 🌊 AKIŞKAN YÜKSEK DEĞERLER (25 LoRA)
        print("\n3️⃣ 🌊 Akışkan yüksek değerli LoRA'lar (25 adet)...")
        for i in range(25):
            lora = LoRAAdapter(
                input_dim=78,  # 🌊 Tarihsel veri dahil!
                hidden_dim=128,
                rank=16,
                alpha=16.0,
                device=self.device
            )
            lora.name = f"Fluid_High_{i+1}"
            # Beta dağılımı (yüksek değerlere eğilimli)
            lora.temperament = {trait: random.betavariate(3, 0.7) for trait in self.temperament_traits}
            population.append(lora)
        
        # 4) 🌊 AKIŞKAN DÜŞÜK DEĞERLER (25 LoRA)
        print("4️⃣ 🌊 Akışkan düşük değerli LoRA'lar (25 adet)...")
        for i in range(25):
            lora = LoRAAdapter(
                input_dim=78,  # 🌊 Tarihsel veri dahil!
                hidden_dim=128,
                rank=16,
                alpha=16.0,
                device=self.device
            )
            lora.name = f"Fluid_Low_{i+1}"
            # Beta dağılımı (düşük değerlere eğilimli)
            lora.temperament = {trait: random.betavariate(0.7, 3) for trait in self.temperament_traits}
            population.append(lora)
        
        print(f"\n{'='*80}")
        print(f"✅ TOPLAM {len(population)} LoRA SPAWN EDİLDİ!")
        print(f"{'='*80}\n")
        
        return population
    
    def save_population(self, population, reset_memory: bool = False):
        """
        Popülasyonu kaydet
        
        Args:
            reset_memory: Ortak hafızayı sıfırla (DEFAULT: False - Hafıza korunur!)
        """
        state_file = "lora_population_state.pt"
        
        print(f"💾 Popülasyon kaydediliyor: {state_file}")
        
        # 🛡️ ORTAK HAFIZAYI KORU! (500+ maçlık deneyim çok değerli!)
        if reset_memory:
            print("   ⚠️ UYARI: Ortak hafıza sıfırlanıyor!")
            collective_memory = {}
            all_loras_ever = {}
        else:
            # 🛡️ ESKİ HAFIZAYI KORU!
            print("   🛡️ Ortak hafıza korunuyor (500+ maçlık deneyim!)")
            if os.path.exists(state_file):
                old_state = torch.load(state_file, map_location='cpu')
                collective_memory = old_state.get('collective_memory', {})
                all_loras_ever = old_state.get('all_loras_ever', {})
            else:
                collective_memory = {}
                all_loras_ever = {}
        
        # 🔥 CPU'YA TAŞI KAYDETMEDEN ÖNCE! (Yükleme sırasında device uyumsuzluğu olmasın!)
        print("   🔄 LoRA'lar CPU'ya taşınıyor (kaydetmek için)...")
        cpu_population = []
        for lora in population:
            # CPU'ya taşı
            lora_cpu = lora.cpu()
            lora_cpu.device = 'cpu'
            cpu_population.append(lora_cpu)
        
        torch.save({
            'population': cpu_population,  # CPU'da kaydedildi!
            'collective_memory': collective_memory,
            'all_loras_ever': all_loras_ever,
            'spawn_info': {
                'type': 'DIVERSE_SPAWN',
                'date': datetime.now().isoformat(),
                'count': len(cpu_population),
                'diversity_level': 'EXTREME',
                'memory_reset': reset_memory,
                'saved_device': 'cpu'  # Bilgi için
            }
        }, state_file)
        
        print(f"✅ {len(population)} LoRA kaydedildi!")
        print(f"   Dosya: {state_file}")
        if reset_memory:
            print(f"   🔥 Ortak hafıza temizlendi (yeni başlangıç!)")


def main():
    """
    Ana fonksiyon (🌊 TAM AKIŞKAN!)
    """
    import argparse
    parser = argparse.ArgumentParser(description='Çeşitli Popülasyon Spawn')
    parser.add_argument('--target', type=int, default=250, help='Hedef LoRA sayısı (Default: 250)')
    parser.add_argument('--reset-memory', action='store_true', help='⚠️ Ortak hafızayı sıfırla (tehlikeli!)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"🌪️ ÇOK ÇEŞİTLİ POPÜLASYON SPAWN SİSTEMİ ({args.target} LoRA!)")
    print("="*80 + "\n")
    
    if args.reset_memory:
        print("⚠️⚠️⚠️ UYARI: Ortak hafıza sıfırlanacak! (500+ maçlık deneyim silinecek!)")
        print("   Emin misiniz? (Ctrl+C ile iptal edin)\n")
        import time
        time.sleep(3)
    
    # Device seç
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}\n")
    
    # Spawner oluştur
    spawner = DiversePopulationSpawner(target_population=args.target, device=device)
    
    # Popülasyon spawn et
    population = spawner.spawn_diverse_population()
    
    # Kaydet (🛡️ HAFIZAYI KORU - Default!)
    spawner.save_population(population, reset_memory=args.reset_memory)
    
    print("\n" + "="*80)
    print("🎉 SPAWN TAMAMLANDI!")
    print("="*80)
    print("\n💡 ŞİMDİ ÇALIŞTIR:")
    print("   python run_evolutionary_learning.py --csv prediction_matches.csv --results results_matches.csv --max 500\n")


if __name__ == "__main__":
    main()

