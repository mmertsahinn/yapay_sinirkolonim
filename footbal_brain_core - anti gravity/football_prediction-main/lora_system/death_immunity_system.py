"""
☠️ ÖLÜMSÜZLÜK SİSTEMİ (Çoklu Uzmanlık Koruması!)
=================================================

Çoklu uzmanlığı olan LoRA'lar neredeyse ölümsüz!

FORMÜL:
  10+ uzmanlık → %98 ölümsüz (Tanrı seviyesi!)
  7+ uzmanlık  → %95 ölümsüz (Efsane!)
  5+ uzmanlık  → %90 ölümsüz (Süper uzman!)
  3+ uzmanlık  → %70 ölümsüz (Çok uzman)
  2 uzmanlık   → %50 ölümsüz (İkili uzman)
  1 uzmanlık   → %25 ölümsüz (Tekli uzman)
  0 uzmanlık   → %0 ölümsüz (Normal LoRA)

Uzmanlık kaybettikçe ölüm riski yavaş yavaş artar!
"""

from typing import Dict, List, Tuple


def calculate_death_immunity(lora, top_5_cache: Dict = None) -> Tuple[float, int]:
    """
    LoRA'nın ölümsüzlük seviyesini hesapla!
    
    Args:
        lora: LoRA instance
        top_5_cache: Tüm takımların Top 5 listeleri
                     {
                         'Manchester_United': {
                             'win_experts': [(lora, score), ...],
                             'goal_experts': [(lora, score), ...],
                             'hype_experts': [(lora, score), ...],
                             'vs_experts': {
                                 'Liverpool': [(lora, score), ...],
                                 ...
                             }
                         },
                         ...
                     }
    
    Returns:
        (immunity_level, specialization_count)
        immunity_level: 0.0-1.0 (ölümsüzlük oranı)
        specialization_count: Toplam uzmanlık sayısı
    """
    
    if top_5_cache is None:
        # Cache yoksa 0 dön
        return 0.0, 0
    
    specialization_count = 0
    
    # Tüm takımları ve uzmanlıkları tara
    for team_name, team_data in top_5_cache.items():
        # Win experts
        if any(l.id == lora.id for l, _ in team_data.get('win_experts', [])):
            specialization_count += 1
        
        # Goal experts
        if any(l.id == lora.id for l, _ in team_data.get('goal_experts', [])):
            specialization_count += 1
        
        # Hype experts
        if any(l.id == lora.id for l, _ in team_data.get('hype_experts', [])):
            specialization_count += 1
        
        # VS experts
        for opponent, vs_experts in team_data.get('vs_experts', {}).items():
            if any(l.id == lora.id for l, _ in vs_experts):
                specialization_count += 1
    
    # ÖLÜMSÜZLÜK SEVİYESİ HESAPLA
    if specialization_count >= 10:
        immunity = 0.98  # Tanrı seviyesi!
    elif specialization_count >= 7:
        immunity = 0.95  # Efsane!
    elif specialization_count >= 5:
        immunity = 0.90  # Süper uzman!
    elif specialization_count >= 3:
        immunity = 0.70  # Çok uzman
    elif specialization_count == 2:
        immunity = 0.50  # İkili uzman
    elif specialization_count == 1:
        immunity = 0.25  # Tekli uzman
    else:
        immunity = 0.0   # Normal LoRA
    
    return immunity, specialization_count


def apply_death_immunity_to_energy_loss(lora, base_energy_loss: float, 
                                        top_5_cache: Dict = None) -> float:
    """
    Ölümsüzlük korumasını life energy kaybına uygula!
    
    Args:
        lora: LoRA instance
        base_energy_loss: Orijinal enerji kaybı (negatif değer)
        top_5_cache: Top 5 listeleri
    
    Returns:
        Modifiye edilmiş enerji kaybı (daha az kayıp!)
    """
    immunity, spec_count = calculate_death_immunity(lora, top_5_cache)
    
    # Ölümsüzlük kaybı azaltır!
    actual_energy_loss = base_energy_loss * (1 - immunity)
    
    if immunity > 0:
        print(f"   🛡️ {lora.name}: {spec_count} uzmanlık → %{immunity*100:.0f} koruma!")
        print(f"      Base kayıp: {base_energy_loss:.3f} → Gerçek kayıp: {actual_energy_loss:.3f}")
    
    return actual_energy_loss


def check_specialization_loss_warning(lora, old_spec_count: int, new_spec_count: int):
    """
    Uzmanlık kaybı uyarısı!
    
    LoRA Top 5'ten düşerse uyarı ver.
    """
    if new_spec_count < old_spec_count:
        lost_count = old_spec_count - new_spec_count
        
        old_immunity = _calculate_immunity_from_count(old_spec_count)
        new_immunity = _calculate_immunity_from_count(new_spec_count)
        
        print(f"\n⚠️ UZMANLIK KAYBI!")
        print(f"   LoRA: {lora.name}")
        print(f"   Eski uzmanlık: {old_spec_count} → Yeni: {new_spec_count}")
        print(f"   Kaybedilen: {lost_count} uzmanlık")
        print(f"   Ölümsüzlük: %{old_immunity*100:.0f} → %{new_immunity*100:.0f}")
        print(f"   ☠️ ÖLÜM RİSKİ ARTTI!")
        
        if new_spec_count == 0:
            print(f"   ⚠️ DİKKAT: Tüm uzmanlıklar kayboldu! Normal LoRA seviyesine düştü!")


def _calculate_immunity_from_count(count: int) -> float:
    """Uzmanlık sayısından ölümsüzlük hesapla"""
    if count >= 10:
        return 0.98
    elif count >= 7:
        return 0.95
    elif count >= 5:
        return 0.90
    elif count >= 3:
        return 0.70
    elif count == 2:
        return 0.50
    elif count == 1:
        return 0.25
    else:
        return 0.0


