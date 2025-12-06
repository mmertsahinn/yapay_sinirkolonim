"""
🧟 TOP 50 DİRİLTME (Basit!)
============================

Top 50 LoRA'yı diriltir, onay sormaz!
"""

import os
import glob
import torch
from lora_system import LoRAAdapter

def resurrect_top50():
    """Top 50'yi dirilt"""
    print("\n" + "🧟"*40)
    print("TOP 50 DİRİLTME!")
    print("🧟"*40 + "\n")
    
    # Top 50 klasörü
    top50_dir = "en_iyi_loralar/⭐_AKTIF_EN_IYILER"
    
    pt_files = glob.glob(os.path.join(top50_dir, "*.pt"))
    print(f"📂 {len(pt_files)} LoRA bulundu!\n")
    
    population = []
    
    for idx, filepath in enumerate(pt_files, 1):
        try:
            data = torch.load(filepath, map_location='cpu')
            
            # LoRA oluştur
            lora = LoRAAdapter(input_dim=78, hidden_dim=128, rank=16, alpha=16.0, device='cpu')
            
            # Metadata yükle
            meta = data.get('metadata', {})
            lora.id = meta.get('id', os.path.basename(filepath).replace('.pt', ''))
            lora.name = meta.get('name', f"Top50_{lora.id}")
            lora.generation = meta.get('generation', 0)
            lora.birth_match = meta.get('birth_match', 0)
            lora.fitness_history = meta.get('fitness_history', [0.5])
            lora.match_history = meta.get('match_history', [])
            lora.temperament = meta.get('temperament', {})
            lora.specialization = meta.get('specialization', None)
            lora.life_energy = meta.get('life_energy', 1.0)
            lora.parents = meta.get('parents', [])
            
            # Parçacık fiziği
            lora._langevin_temp = meta.get('langevin_temp', 0.01)
            lora._lazarus_lambda = meta.get('lazarus_lambda', 0.5)
            lora._om_action = meta.get('om_action', 0.0)
            
            # Parametreleri yükle
            lora_params = data.get('lora_params', {})
            if lora_params:
                lora.set_all_lora_params(lora_params)
            
            population.append(lora)
            print(f"   [{idx}/{len(pt_files)}] {lora.name} ✅")
            
        except Exception as e:
            print(f"   [{idx}/{len(pt_files)}] HATA: {e}")
    
    # Kaydet
    if population:
        torch.save({
            'population': population,
            'all_loras_ever': {},
            'collective_memory': {},
            'resurrection_info': {
                'type': 'TOP50_RESURRECTION',
                'count': len(population)
            }
        }, 'lora_population_state.pt')
        
        print(f"\n✅ {len(population)} LoRA diriltildi!")
        print(f"💾 lora_population_state.pt oluşturuldu!")
        print(f"\n🚀 HAZIR! python run_evolutionary_learning.py --resume ile başlat!")
    else:
        print("\n❌ Hiçbir LoRA yüklenemedi!")

if __name__ == "__main__":
    resurrect_top50()


