"""
🧟 ACİL DURUM DİRİLTME SİSTEMİ (TOP 50'DEN!)
==============================================

KULLANIM:
    Popülasyon kritik seviyeye düştüğünde veya yeni başlangıç için
    TOP 50 LoRA'yı dirilt!
    
    python emergency_resurrect_all.py --target 250

MANTIK:
    1. en_iyi_loralar/⭐_AKTIF_EN_IYILER/ klasöründeki Top 50 LoRA'yı oku
    2. Lazarus Lambda'ya göre sırala
    3. En iyi N tanesini seç (default: 250, ama sadece 50 var → hepsi alınır)
    4. 200 tane yeni spawn et (250'ye tamamla)
    5. lora_population_state.pt oluştur
    
⚠️ YENİ SİSTEM:
    - Einstein/Newton/Darwin Hall artık yok (dinamik sistem!)
    - Mucizeler artık yok (çoklu uzmanlık = mucize!)
    - Tek kaynak: Top 50 LoRA
"""

import os
import sys
import torch
import yaml
import re
from datetime import datetime

class EmergencyResurrection:
    """
    Tüm LoRA'ları toplu diriltme sistemi
    """
    
    def __init__(self):
        self.config_path = "evolutionary_config.yaml"
        self.state_file = "lora_population_state.pt"
        self.wallets_dir = "lora_wallets"
        
        # Hall of Fame klasörleri
        self.einstein_dir = "en_iyi_loralar/🌟_EINSTEIN_HALL"
        self.darwin_dir = "en_iyi_loralar/🧬_DARWIN_HALL"
        self.newton_dir = "en_iyi_loralar/🏛️_NEWTON_HALL"
        self.active_dir = "en_iyi_loralar/⭐_AKTIF_EN_IYILER"
        self.miracle_dir = "mucizeler"  # 🏆 MUCİZELER!
        
        # Config yükle
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("🧟 Acil Durum Diriltme Sistemi başlatıldı!")
    
    def collect_all_pt_files(self) -> dict:
        """
        TÜM Hall klasörlerinden .pt dosyalarını topla!
        
        Returns:
            {lora_id: filepath} dictionary
        """
        all_pt_files = {}
        
        # SADECE AKTIF TOP 50! (Tek kaynak!)
        all_dirs = [
            ("⭐ Top 50 LoRA", self.active_dir)
        ]
        
        print(f"\n📂 .PT DOSYALARI TOPLANIYOR:")
        print(f"{'='*80}")
        
        for hall_name, hall_dir in all_dirs:
            if os.path.exists(hall_dir):
                pt_files = [f for f in os.listdir(hall_dir) if f.endswith('.pt')]
                
                for pt_file in pt_files:
                    lora_id = pt_file.replace('.pt', '')
                    filepath = os.path.join(hall_dir, pt_file)
                    
                    # Eğer daha önce eklenmemişse ekle
                    if lora_id not in all_pt_files:
                        all_pt_files[lora_id] = filepath
                
                print(f"   {hall_name}: {len(pt_files)} LoRA")
            else:
                print(f"   {hall_name}: Klasör yok ⚠️")
        
        print(f"{'='*80}")
        print(f"   📊 TOPLAM: {len(all_pt_files)} benzersiz LoRA!\n")
        
        return all_pt_files
    
    def load_lora_from_pt_file(self, filepath: str):
        """
        .pt dosyasından LoRA'yı yükle!
        
        Args:
            filepath: .pt dosyasının tam yolu
        
        Returns:
            LoRA objesi veya None
        """
        if not os.path.exists(filepath):
            return None
        
        try:
            data = torch.load(filepath, map_location='cpu')
            
            # LoRA objesini yeniden oluştur
            from lora_system import LoRAAdapter
            
            # Config'den boyutları al
            input_dim = self.config.get('lora', {}).get('input_dim', 63)
            hidden_dim = self.config.get('lora', {}).get('hidden_dim', 128)
            rank = self.config.get('lora', {}).get('rank', 16)
            alpha = self.config.get('lora', {}).get('alpha', 16.0)
            
            lora = LoRAAdapter(
                input_dim=78,  # 🌊 YENİ: Tarihsel veri dahil!
                hidden_dim=hidden_dim,
                rank=rank,
                alpha=alpha,
                device='cpu'
            )
            
            # Metadata'yı yükle
            metadata = data.get('metadata', {})
            
            # ID'yi metadata'dan al, yoksa dosya adından çıkar
            extracted_id = os.path.basename(filepath).replace('.pt', '')
            
            lora.id = metadata.get('id', extracted_id)
            lora.name = metadata.get('name', f"Resurrected_{extracted_id}")
            lora.generation = metadata.get('generation', 0)
            lora.birth_match = metadata.get('birth_match', 0)
            lora.parents = metadata.get('parents', [])
            lora.fitness_history = metadata.get('fitness_history', [0.5])
            lora.match_history = metadata.get('match_history', [])
            lora.temperament = metadata.get('temperament', {})
            lora.specialization = metadata.get('specialization', None)
            lora.emotional_archetype = metadata.get('emotional_archetype', 'Dengeli')
            lora.life_energy = metadata.get('life_energy', 1.0)
            
            # Parçacık fiziği verileri
            lora._langevin_temp = metadata.get('langevin_temp', 0.01)
            lora._nose_hoover_xi = metadata.get('nose_hoover_xi', 0.0)
            lora._kinetic_energy = metadata.get('kinetic_energy', 0.0)
            lora._lazarus_lambda = metadata.get('lazarus_lambda', 0.5)
            lora._om_action = metadata.get('om_action', 0.0)
            lora._ghost_potential = metadata.get('ghost_potential', 0.0)
            lora._particle_archetype = metadata.get('particle_archetype', 'Unknown')
            
            # LoRA parametrelerini yükle
            lora_params = data.get('lora_params', {})
            if lora_params:
                lora.set_all_lora_params(lora_params)
            
            return lora
            
        except Exception as e:
            print(f"⚠️ {filepath} yüklenemedi: {e}")
            return None
    
    def resurrect_all(self, target_population: int = None):
        """
        TÜM LoRA'ları dirilt! (Lazarus Lambda sıralı!)
        
        Args:
            target_population: Hedef LoRA sayısı (None = Hepsini dirilt!)
        """
        print("\n" + "🧟"*40)
        if target_population:
            print(f"ACİL DURUM DİRİLTME (Hedef: {target_population} LoRA!)")
        else:
            print(f"ACİL DURUM DİRİLTME (TÜM LORA'LAR!)")
        print("🧟"*40 + "\n")
        
        # 1) Mevcut durumu yükle
        if os.path.exists(self.state_file):
            print(f"📂 Mevcut durum yükleniyor: {self.state_file}")
            state = torch.load(self.state_file, map_location='cpu')
            current_population = state.get('population', [])
            all_loras_ever = state.get('all_loras_ever', {})
            collective_memory = state.get('collective_memory', {})
            
            print(f"   📊 Mevcut popülasyon: {len(current_population)} LoRA")
        else:
            print("⚠️ Mevcut durum bulunamadı, yeni başlangıç yapılacak!")
            current_population = []
            all_loras_ever = {}
            collective_memory = {}
        
        # 2) EINSTEIN + DARWIN + NEWTON HALL'larından TÜM .pt dosyalarını topla!
        all_pt_files = self.collect_all_pt_files()
        
        print(f"\n🔍 {len(all_pt_files)} LoRA bulundu!")
        
        # 3) 🌊 LAZARUS LAMBDA İLE SIRALA! (En iyi 250'yi alacağız!)
        print(f"🧟 Lazarus Lambda hesaplanıyor...\n")
        
        lora_scores = []
        for lora_id, filepath in all_pt_files.items():
            lora = self.load_lora_from_pt_file(filepath)
            if lora:
                lazarus_lambda = getattr(lora, '_lazarus_lambda', 0.5)
                lora_scores.append((lora, filepath, lazarus_lambda))
        
        # Lazarus Lambda'ya göre sırala (yüksekten düşüğe!)
        lora_scores.sort(key=lambda x: x[2], reverse=True)
        
        # İlk target_population'ı al (None ise HEPSİNİ!)
        if target_population:
            top_loras = lora_scores[:target_population]
        else:
            top_loras = lora_scores  # TÜM LORA'LAR!
        
        print(f"✅ Lazarus Lambda sıralaması tamamlandı!")
        print(f"   {'HEPSİ' if not target_population else f'En iyi {len(top_loras)}'} LoRA seçildi!\n")
        
        # 4) Seçilen LoRA'ları yükle
        resurrected_count = 0
        failed_count = 0
        new_population = []
        
        for idx, (lora, filepath, lazarus_lambda) in enumerate(top_loras, start=1):
            print(f"   [{idx}/{len(top_loras)}] {lora.id} (🧟 Λ:{lazarus_lambda:.3f})...", end=" ")
            
            new_population.append(lora)
            resurrected_count += 1
            
            # Top 50'den
            hall = "⭐ Top50"
            
            print(f"✅ Diriltildi! ({hall} | Fit:{lora.get_recent_fitness():.3f})")
        
        print(f"\n{'='*80}")
        print(f"📊 DİRİLTME SONUÇLARI:")
        print(f"{'='*80}")
        print(f"   ✅ Başarılı: {resurrected_count} LoRA")
        print(f"   ❌ Başarısız: {failed_count} LoRA")
        print(f"   📊 Yeni Popülasyon: {len(new_population)} LoRA")
        print(f"{'='*80}\n")
        
        # 4) Durumu kaydet
        if len(new_population) > 0:
            # all_loras_ever'ı güncelle
            for lora in new_population:
                all_loras_ever[lora.id] = {
                    'lora': lora,
                    'alive': True,
                    'birth_match': lora.birth_match,
                    'generation': lora.generation
                }
            
            # State dosyasını kaydet
            torch.save({
                'population': new_population,
                'all_loras_ever': all_loras_ever,
                'collective_memory': collective_memory,
                'resurrection_info': {
                    'type': 'EMERGENCY_FULL_RESURRECTION',
                    'date': datetime.now().isoformat(),
                    'resurrected_count': resurrected_count,
                    'total_population': len(new_population)
                }
            }, self.state_file)
            
            print(f"💾 Yeni durum kaydedildi: {self.state_file}")
            print(f"   📊 Popülasyon: {len(new_population)} LoRA")
            
            # Özet rapor
            print(f"\n📋 POPÜLASYON ÖZETİ:")
            print(f"{'='*80}")
            
            # Nesil dağılımı
            generations = [lora.generation for lora in new_population]
            print(f"   Ortalama Nesil: {sum(generations) / len(generations):.1f}")
            print(f"   En Genç: Gen {min(generations)}")
            print(f"   En Yaşlı: Gen {max(generations)}")
            
            # Fitness dağılımı
            fitnesses = [lora.get_recent_fitness() for lora in new_population]
            print(f"   Ortalama Fitness: {sum(fitnesses) / len(fitnesses):.3f}")
            print(f"   En Düşük: {min(fitnesses):.3f}")
            print(f"   En Yüksek: {max(fitnesses):.3f}")
            
            # Enerji dağılımı
            energies = [getattr(lora, 'life_energy', 1.0) for lora in new_population]
            print(f"   Ortalama Enerji: {sum(energies) / len(energies):.3f}")
            
            print(f"{'='*80}\n")
            
            print("✅ ACİL DİRİLTME TAMAMLANDI!")
            print("   🚀 Artık run_evolutionary_learning.py --resume ile devam edebilirsiniz!")
        else:
            print("❌ Hiçbir LoRA diriltileemedi!")


def main():
    """
    Ana fonksiyon
    """
    print("\n" + "="*80)
    print("🧟 ACİL DURUM DİRİLTME SİSTEMİ")
    print("="*80)
    import argparse
    parser = argparse.ArgumentParser(description='Acil Diriltme Sistemi')
    parser.add_argument('--target', type=int, default=250, help='Hedef LoRA sayısı (Default: 250)')
    parser.add_argument('--no-confirm', action='store_true', help='Onay sorma (otomasyon için)')
    args = parser.parse_args()
    
    print(f"\n⚠️  Bu script SADECE acil durumlarda kullanılır!")
    print(f"   En iyi {args.target} LoRA'yı (Lazarus Lambda sıralı) geri getirir.\n")
    print("📂 Diriltilecek kaynaklar:")
    print("   🏆 Mucizeler")
    print("   🌟 Einstein Hall")
    print("   🧬 Darwin Hall")
    print("   🏛️ Newton Hall")
    print("   ⭐ Aktif En İyiler")
    print(f"\n🧟 LAZARUS LAMBDA SIRALAMA! {'HEPSİNİ' if not args.target or args.target <= 0 else f'En iyi {args.target}'} LoRA!\n")
    
    # Direkt başla (onay isteme!)
    print("✅ Diriltme başlıyor...\n")
    
    # Diriltme sistemini başlat
    resurrector = EmergencyResurrection()
    target = None if (not args.target or args.target <= 0) else args.target
    resurrector.resurrect_all(target_population=target)


if __name__ == "__main__":
    main()

