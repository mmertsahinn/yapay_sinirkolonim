import torch
import os

state_file = "lora_population_state.pt"

if os.path.exists(state_file):
    state = torch.load(state_file, map_location='cpu')
    population = state.get('population', [])
    all_loras_ever = state.get('all_loras_ever', {})
    
    print(f"Popülasyon: {len(population)} LoRA")
    print(f"Tüm zamanlar: {len(all_loras_ever)} LoRA (ölüler dahil)")
    
    if len(population) == 0:
        print("\n❌ HERKES ÖLÜ!")
        print("   Acil diriltme gerekiyor!")
else:
    print("State dosyası yok!")


