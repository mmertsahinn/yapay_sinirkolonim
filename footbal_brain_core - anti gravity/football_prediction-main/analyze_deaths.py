"""
ÖLÜM ANALİZİ - NEDEN HERKES ÖLÜYOR?
"""
import torch
import os
import json

print(f"{'='*80}")
print(f"💀 ÖLÜM ANALİZ RAPORU")
print(f"{'='*80}\n")

# 1. State kontrol
if os.path.exists('lora_population_state.pt'):
    state = torch.load('lora_population_state.pt')
    
    current_pop = len(state['population'])
    all_loras = state.get('all_loras_summary', {})
    
    alive_count = sum(1 for info in all_loras.values() if info.get('alive', False))
    dead_count = len(all_loras) - alive_count
    
    print(f"📊 GENEL DURUM:")
    print(f"{'─'*80}")
    print(f"Mevcut popülasyon: {current_pop} LoRA")
    print(f"Toplam kayıt: {len(all_loras)} LoRA")
    print(f"  ⭐ Yaşayan: {alive_count}")
    print(f"  💀 Ölü: {dead_count}")
    print(f"  💀 Ölüm oranı: {dead_count / len(all_loras) * 100:.1f}%")
    
    # Ölüm sebeplerini topla
    death_reasons = {}
    dead_loras = []
    
    for lora_id, info in all_loras.items():
        if not info.get('alive', True):
            reason = info.get('death_reason', 'Bilinmiyor')
            
            if reason not in death_reasons:
                death_reasons[reason] = 0
            death_reasons[reason] += 1
            
            dead_loras.append({
                'name': info['name'],
                'reason': reason,
                'fitness': info.get('final_fitness', 0),
                'age': info.get('age', 0),
                'death_match': info.get('death_match', '?')
            })
    
    print(f"\n💀 ÖLÜM SEBEPLERİ ANALİZİ:")
    print(f"{'─'*80}")
    for reason, count in sorted(death_reasons.items(), key=lambda x: x[1], reverse=True):
        percentage = count / dead_count * 100
        print(f"  {count:2d}x ({percentage:5.1f}%) - {reason}")
    
    print(f"\n💀 ÖLÜLER LİSTESİ:")
    print(f"{'─'*80}")
    for i, lora in enumerate(dead_loras[:10], 1):
        print(f"  {i}. {lora['name']}")
        print(f"     • Sebep: {lora['reason']}")
        print(f"     • Fitness: {lora['fitness']:.3f}")
        print(f"     • Yaş: {lora['age']} maç")
        print(f"     • Ölüm: Maç #{lora['death_match']}")
    
    if len(dead_loras) > 10:
        print(f"  ... ve {len(dead_loras) - 10} LoRA daha")

# 2. Evolution events analizi
if os.path.exists('evolution_logs/evolution_data.json'):
    with open('evolution_logs/evolution_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    births = [e for e in data['events'] if e['type'] == 'birth']
    deaths = [e for e in data['events'] if e['type'] == 'death']
    matings = [e for e in data['events'] if e['type'] in ['mating_success', 'mating']]
    
    print(f"\n🧬 EVRİM İSTATİSTİKLERİ:")
    print(f"{'─'*80}")
    print(f"  🐣 Doğumlar: {len(births)}")
    print(f"  💀 Ölümler: {len(deaths)}")
    print(f"  💕 Çiftleşmeler: {len(matings)}")
    print(f"  ⚖️ Denge: {len(births)} doğum vs {len(deaths)} ölüm")
    
    if len(births) < len(deaths):
        diff = len(deaths) - len(births)
        print(f"\n  ⚠️ SORUN: {diff} fazla ölüm var!")
        print(f"  → Üreme yetersiz veya ölüm çok fazla!")

# 3. Doğa durumu
if os.path.exists('lora_population_state.pt'):
    nature = state.get('nature_state', {})
    
    print(f"\n🌍 DOĞA DURUMU:")
    print(f"{'─'*80}")
    print(f"  ❤️ Sağlık: {nature.get('health', 'N/A')}")
    print(f"  😡 Öfke: {nature.get('anger', 'N/A')}")
    print(f"  🌪️ Kaos: {nature.get('chaos_index', 'N/A')}")
    
    if isinstance(nature.get('health'), (int, float)) and nature.get('health') < 0.3:
        print(f"\n  ⚠️ SORUN: Doğa sağlığı çok düşük!")
        print(f"  → Felaket riski yüksek!")
    
    if isinstance(nature.get('anger'), (int, float)) and nature.get('anger') > 0.7:
        print(f"\n  ⚠️ SORUN: Doğa öfkesi çok yüksek!")
        print(f"  → Deprem/Veba riski!")

print(f"\n{'='*80}")
print(f"📝 Analiz tamamlandı!")
print(f"{'='*80}\n")



