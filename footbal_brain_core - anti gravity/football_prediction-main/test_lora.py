"""
LoRA test
"""
import torch
import numpy as np
from lora_system import LoRAAdapter

print("🧬 LoRA Testi...")

# LoRA oluştur
lora = LoRAAdapter(input_dim=63, hidden_dim=128, rank=16, alpha=16.0)
lora = lora.to('cuda')

print(f"✅ LoRA oluşturuldu: {lora.name}")

# Test input
features = np.random.randn(60).astype(np.float32)
base_proba = np.array([0.4, 0.3, 0.3], dtype=np.float32)

print(f"Features: {features[:5]}...")
print(f"Base proba: {base_proba}")

# Tahmin
try:
    pred = lora.predict(features, base_proba, device='cuda')
    print(f"\n✅ TAHMİN BAŞARILI!")
    print(f"Prediction: {pred}")
    
    if np.isnan(pred).any():
        print(f"\n❌ NaN BULUNDU!")
    else:
        print(f"\n✅ NaN YOK!")
        
except Exception as e:
    print(f"\n❌ HATA: {e}")
    import traceback
    traceback.print_exc()




