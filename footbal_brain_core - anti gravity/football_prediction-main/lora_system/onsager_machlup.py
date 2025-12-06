"""
🌀 ONSAGER-MACHLUP YÖRÜNGE İNTEGRALİ
=====================================

Fitness sadece "şu an" değil, "TÜM TARİHÇE"!

Onsager-Machlup Fonksiyoneli:

S_OM(φ) = ∫[0,T] [(dφ/dt + ∇U)^2 / (4T) + ∇·V] dt

Anlam:
  • Birinci terim: "Ne kadar zorlandı?" (Newton cost!)
  • İkinci terim: "Parametre uzayı ne kadar değişti?" (Entropi!)

En düşük eylem (S_OM) = En iyi yörünge!
"""

import torch
import math
from typing import List, Dict


class OnsagerMachlup:
    """
    Yörünge integrali hesaplama
    """
    
    def __init__(self, temperature: float = 0.01):
        """
        Args:
            temperature: Sistem sıcaklığı (T)
        """
        self.T = temperature
        print(f"🌀 Onsager-Machlup başlatıldı (T={temperature})")
    
    def calculate_action(
        self,
        lora,
        trajectory: List[Dict] = None
    ) -> Dict:
        """
        LoRA'nın yörünge eylemini hesapla!
        
        S_OM = ∫ [(dθ/dt + ∇U)^2 / (4T) + div] dt
        
        Args:
            lora: LoRA instance
            trajectory: Yörünge geçmişi (opsiyonel)
        
        Returns:
            {
                'action': S_OM değeri,
                'newton_cost': İlk terim (zorluk!),
                'entropy_term': İkinci terim (çeşitlilik!)
            }
        """
        # Yörünge bilgisi yoksa basit hesapla
        if trajectory is None:
            trajectory = self._reconstruct_trajectory(lora)
        
        if len(trajectory) < 2:
            return {'action': 0.0, 'newton_cost': 0.0, 'entropy_term': 0.0}
        
        total_action = 0.0
        total_newton = 0.0
        total_entropy = 0.0
        
        # Her adım için
        for i in range(len(trajectory) - 1):
            theta_t = trajectory[i]['params']
            theta_t1 = trajectory[i+1]['params']
            grad_t = trajectory[i]['gradient']
            
            dt = 1.0  # Zaman adımı
            
            # 1) PARAMETRE DEĞİŞİMİ: dθ/dt
            dtheta_dt = (theta_t1 - theta_t) / dt
            
            # 2) NEWTON TERİMİ: (dθ/dt + ∇U)^2 / (4T)
            # ∇U = gradyan
            deviation = dtheta_dt + grad_t
            newton_cost = torch.sum(deviation ** 2).item() / (4 * self.T)
            
            # 3) ENTROPİ TERİMİ: ∇·V (Diverjans!)
            # Basit yaklaşım: Parametre değişiminin varyansı
            entropy_term = torch.var(theta_t1 - theta_t).item()
            
            # 4) TOPLAM EYLEM
            action_t = (newton_cost + entropy_term) * dt
            
            total_action += action_t
            total_newton += newton_cost * dt
            total_entropy += entropy_term * dt
        
        return {
            'action': total_action,
            'newton_cost': total_newton,
            'entropy_term': total_entropy,
            'trajectory_length': len(trajectory),
            'efficiency': 1.0 / (total_action + 1e-8)  # Düşük eylem = Yüksek verimlilik!
        }
    
    def _reconstruct_trajectory(self, lora) -> List[Dict]:
        """
        LoRA'nın geçmişinden yörünge rekonstrüksiyonu
        
        (Gerçek uygulamada LoRA her adımda parametrelerini kaydetmeli!)
        """
        trajectory = []
        
        # Eğer LoRA'nın param geçmişi varsa
        if hasattr(lora, 'param_history') and len(lora.param_history) > 0:
            for entry in lora.param_history:
                trajectory.append({
                    'params': entry.get('params'),
                    'gradient': entry.get('gradient', torch.zeros_like(entry['params']))
                })
        else:
            # Yoksa şu anki parametrelerle dummy yörünge
            current_params = lora.get_all_lora_params()
            trajectory.append({
                'params': current_params,
                'gradient': torch.zeros_like(current_params)
            })
        
        return trajectory
    
    def compare_loras_by_action(
        self,
        lora_list: List
    ) -> List[tuple]:
        """
        LoRA'ları eylemlerine göre karşılaştır!
        
        Returns:
            [(lora, action_data), ...] (Küçükten büyüğe sıralı!)
        """
        results = []
        
        for lora in lora_list:
            try:
                action_data = self.calculate_action(lora)
                results.append((lora, action_data))
            except:
                continue
        
        # Eyleme göre sırala (Küçükten büyüğe!)
        # Düşük eylem = Verimli yörünge!
        results.sort(key=lambda x: x[1]['action'])
        
        return results


# Global instance
onsager_machlup = OnsagerMachlup(temperature=0.01)



