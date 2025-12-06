"""
🌊 LANGEVIN DİNAMİKLERİ (Stokastik Gradyan İnişi → Fiziksel SDE!)
==================================================================

DETERMİNİZM YOK! KAOS VAR! MATEMATİK TAM!

Langevin Denklemi:
------------------
dθ = -∇U(θ) dt + √(2T) dW

Nerede:
  • θ: Parametre (LoRA ağırlıkları!)
  • U(θ): Potansiyel enerji (Loss fonksiyonu!)
  • T: Sıcaklık (Gürültü seviyesi!)
  • dW: Wiener süreci (Brownian hareket!)

Bu, LoRA'yı deterministik bir "robot"tan, termal banyoda yüzen bir
"parçacık"a dönüştürür!
"""

import torch
import math


class LangevinDynamics:
    """
    Langevin SDE ile LoRA parametre güncellemesi
    """
    
    def __init__(
        self,
        base_temperature: float = 0.01,
        dt: float = 0.01,
        adaptive: bool = True
    ):
        """
        Args:
            base_temperature: Temel sıcaklık (T_base)
            dt: Zaman adımı
            adaptive: Nosé-Hoover termostat kullan mı?
        """
        self.T_base = base_temperature
        self.dt = dt
        self.adaptive = adaptive
        
        # Nosé-Hoover için
        self.xi = {}  # Her LoRA için sürtünme katsayısı
        self.momentum = {}  # Her LoRA için momentum
        
        print(f"🌊 Langevin Dynamics başlatıldı (T={base_temperature}, adaptive={adaptive})")
    
    def update_parameters(
        self,
        lora,
        gradients: dict,
        temperature: float = None
    ) -> dict:
        """
        LoRA parametrelerini Langevin SDE ile güncelle!
        
        Args:
            lora: LoRA instance
            gradients: {layer_name: gradient_tensor}
            temperature: Opsiyonel sıcaklık (None ise otomatik!)
        
        Returns:
            {
                'T_eff': Efektif sıcaklık,
                'noise_magnitude': Gürültü büyüklüğü,
                'drift_magnitude': Sürüklenme büyüklüğü
            }
        """
        # Sıcaklık hesapla
        if temperature is None:
            if self.adaptive:
                T_eff = self._compute_adaptive_temperature(lora, gradients)
            else:
                T_eff = self.T_base
        else:
            T_eff = temperature
        
        # Langevin güncellemesi!
        total_drift = 0.0
        total_noise = 0.0
        
        for layer_name, grad in gradients.items():
            # 1) DRİFT: -∇U(θ) dt
            drift = -grad * self.dt
            
            # 2) DİFÜZYON: √(2T) dW
            noise_std = math.sqrt(2 * T_eff * self.dt)
            noise = torch.randn_like(grad) * noise_std
            
            # 3) TOPLAM GÜNCELLEME
            delta = drift + noise
            
            # Parametreyi güncelle (lora.lora_A veya lora_B'ye uygula!)
            # (Bu kısım lora.update_params() ile yapılacak!)
            
            # İstatistikler
            total_drift += drift.abs().mean().item()
            total_noise += noise.abs().mean().item()
        
        return {
            'T_eff': T_eff,
            'noise_magnitude': total_noise,
            'drift_magnitude': total_drift,
            'noise_to_drift_ratio': total_noise / (total_drift + 1e-8)
        }
    
    def _compute_adaptive_temperature(self, lora, gradients: dict) -> float:
        """
        Nosé-Hoover Termostat ile adaptif sıcaklık!
        
        dξ = (KE - KE_target) dt
        T_eff = KE / (d/2)
        """
        lora_id = lora.id
        
        # İlk kez mi?
        if lora_id not in self.xi:
            self.xi[lora_id] = 0.0
            self.momentum[lora_id] = {}
        
        # 1) KİNETİK ENERJİ HESAPLA
        KE = 0.0
        d = 0  # Toplam parametre sayısı
        
        for layer_name, grad in gradients.items():
            # Momentum = Gradyan
            if layer_name not in self.momentum[lora_id]:
                self.momentum[lora_id][layer_name] = torch.zeros_like(grad)
            
            p = self.momentum[lora_id][layer_name]
            
            # Momentum güncelle
            xi = self.xi[lora_id]
            p_new = p + (-grad - xi * p) * self.dt
            
            # KE += (1/2) ||p||^2
            KE += 0.5 * (p_new ** 2).sum().item()
            d += p_new.numel()
            
            # Kaydet
            self.momentum[lora_id][layer_name] = p_new
        
        # 2) HEDEF KİNETİK ENERJİ
        KE_target = (d / 2.0) * self.T_base
        
        # 3) TERMOSTAT GÜNCELLEMESİ
        # dξ = (KE - KE_target) dt
        dxi = (KE - KE_target) * self.dt * 0.01  # 0.01: damp factor
        self.xi[lora_id] += dxi
        
        # 4) EFEKTİF SICAKLIK
        T_eff = (2.0 * KE) / (d + 1e-8)
        
        # Sınırla (çok aşırı olmasın!)
        T_eff = max(0.001, min(T_eff, 0.5))
        
        return T_eff
    
    def compute_gradient_variance_temperature(self, lora, window: int = 10) -> float:
        """
        Gradyan varyansına göre sıcaklık (alternatif yöntem!)
        
        T(t) = T_base × (1 + α × Var(∇loss))
        """
        # Eğer LoRA'nın son gradyan geçmişi varsa
        if not hasattr(lora, 'gradient_history') or len(lora.gradient_history) < 2:
            return self.T_base
        
        recent_grads = lora.gradient_history[-window:]
        
        # Varyans hesapla
        grad_variance = torch.var(torch.stack(recent_grads)).item()
        
        # Adaptif sıcaklık
        alpha = 0.5
        T = self.T_base * (1.0 + alpha * grad_variance)
        
        return min(T, 0.5)  # Max 0.5


# Global instance
langevin_dynamics = LangevinDynamics(
    base_temperature=0.01,
    dt=0.01,
    adaptive=True  # Nosé-Hoover aktif!
)



