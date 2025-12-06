"""
⚡ DİNAMİK YERLEŞTİRME MOTORU
==============================

LoRA'ları dinamik olarak doğru klasörlere yerleştirir:

AKILLI SİSTEM:
- Performansa göre otomatik kategori seçimi
- Çoklu yerleştirme (bir LoRA birden fazla kategoride olabilir)
- Otomatik sıralama (her kategoride Top 5/10)
- Zaman içinde değişebilir (performans değiştikçe yeniden yerleşir)

ÖRNEKler:
1. LoRA_X:
   - Genel: %75 başarı
   - Manchester: %85 başarı
   - Liverpool: %65 başarı
   → Yerleştirme: GENEL (Top 10) + MANCHESTER (Top 5)

2. LoRA_Y:
   - Genel: %50 başarı
   - Man vs Liv: %90 başarı
   → Yerleştirme: SADECE VS_MAN_LIV (Top 5)

3. LoRA_Z:
   - Genel: %80 başarı
   - Tüm takımlar: %75-85 arası (dengeli)
   - High Hype: %85 başarı
   → Yerleştirme: GENEL (Top 5) + HYPE_HIGH (Top 5)
"""

import os
import torch
from typing import Dict, List, Tuple
from datetime import datetime


class DynamicPlacementEngine:
    """
    LoRA'ları dinamik olarak yerleştirir
    """
    
    def __init__(self):
        self.placement_history = {}  # {lora_id: [placements...]}
        print("⚡ Dynamic Placement Engine başlatıldı")
    
    def place_lora_intelligently(self,
                                 lora,
                                 categorization: Dict,
                                 match_count: int) -> Dict:
        """
        LoRA'yı akıllıca yerleştir
        
        Returns:
            {
                'placements': [
                    {
                        'path': str,
                        'category': str,
                        'rank': int,  # Bu kategorideki sırası
                        'score': float,  # Bu kategorideki skoru
                        'reason': str
                    },
                    ...
                ],
                'primary_placement': {...},  # Ana yerleştirme
                'total_placements': int
            }
        """
        
        placements = []
        
        # 1) GENEL UZMAN YERLEŞTİRMESİ
        if categorization['global_accuracy'] >= 0.65:
            # Genel başarı yeterince yüksek
            general_score = categorization['global_accuracy']
            
            # ✅ DÜZELTME: os.path.join kullan!
            general_dir = os.path.join('en_iyi_loralar', '🌍_GENEL_UZMANLAR', '🎯_WIN_EXPERTS')
            
            placements.append({
                'path': general_dir,
                'category': 'GENERAL_WIN',
                'score': general_score,
                'reason': f"Genel başarı yüksek (%{general_score*100:.0f})"
            })
        
        # 2) TAKIM SPESIFIK YERLEŞTİRMELER
        for team_name, team_score in categorization['team_specializations']:
            # Takımda yeterince iyi VE global'den daha iyi mi?
            if team_score >= 0.70:
                # ✅ DÜZELTME: Güvenli dosya adı + os.path.join!
                safe_name = self._safe_team_name(team_name)
                team_dir = os.path.join('en_iyi_loralar', 'takım_uzmanlıkları', safe_name, '🎯_WIN_EXPERTS')
                
                placements.append({
                    'path': team_dir,
                    'category': f'TEAM_{safe_name}',
                    'score': team_score,
                    'reason': f"{team_name} uzmanı (%{team_score*100:.0f})"
                })
        
        # 3) HYPE YERLEŞTİRMESİ
        if categorization['hype_specialization']:
            hype_level, hype_score = categorization['hype_specialization']
            
            if hype_score >= 0.70:
                # ✅ DÜZELTME: os.path.join kullan!
                hype_dir = os.path.join('en_iyi_loralar', '🌍_GENEL_UZMANLAR', '🔥_HYPE_EXPERTS')
                
                placements.append({
                    'path': hype_dir,
                    'category': f'HYPE_{hype_level.upper()}',
                    'score': hype_score,
                    'reason': f"Hype uzmanı ({hype_level}: %{hype_score*100:.0f})"
                })
        
        # 4) HYBRID PLACEMENT (Hem genel hem spesifik!)
        general_good = categorization['global_accuracy'] >= 0.65
        specific_good = len([p for p in placements if 'TEAM_' in p['category']]) >= 2
        
        if general_good and specific_good:
            # ✅ DÜZELTME: Doğru dosya yolu!
            hybrid_dir = os.path.join('en_iyi_loralar', '🌈_HYBRID_HALL')
            placements.append({
                'path': hybrid_dir,
                'category': 'HYBRID',
                'score': (categorization['global_accuracy'] + categorization['specificity_score']) / 2,
                'reason': "Hem genel hem özel başarılı!"
            })
        
        # Ana yerleştirme (en yüksek skorlu)
        if placements:
            primary = max(placements, key=lambda x: x['score'])
        else:
            primary = None
        
        # History'e kaydet
        self.placement_history[lora.id] = {
            'match_count': match_count,
            'placements': placements,
            'primary': primary
        }
        
        return {
            'placements': placements,
            'primary_placement': primary,
            'total_placements': len(placements)
        }
    
    def _safe_team_name(self, team_name: str) -> str:
        """Dosya sistemi için güvenli takım ismi"""
        return team_name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_').replace(':', '').replace('*', '').replace('?', '').replace('"', '').replace('<', '').replace('>', '').replace('|', '')
    
    def export_placements_to_files(self, 
                                   lora,
                                   placement_result: Dict,
                                   match_count: int):
        """
        LoRA'yı belirlenen klasörlere yerleştir (.pt dosyası kopyala)
        """
        
        for placement in placement_result['placements']:
            path = placement['path']
            
            # Klasörü oluştur
            os.makedirs(path, exist_ok=True)
            
            # PT dosyası kaydet
            filename = f"{lora.name}_{lora.id}.pt"
            filepath = os.path.join(path, filename)
            
            torch.save({
                'lora_params': lora.get_all_lora_params(),
                'metadata': {
                    'id': lora.id,
                    'name': lora.name,
                    'category': placement['category'],
                    'score': placement['score'],
                    'reason': placement['reason'],
                    'match_count': match_count,
                    'placement_timestamp': datetime.now().isoformat(),
                    'is_primary': (placement == placement_result['primary_placement'])
                }
            }, filepath)


# Global instance
dynamic_placement_engine = DynamicPlacementEngine()

