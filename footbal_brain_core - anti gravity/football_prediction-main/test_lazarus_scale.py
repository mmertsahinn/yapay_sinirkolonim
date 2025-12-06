
import torch
import sys
import os
import numpy as np

# Add path
sys.path.append(os.getcwd())

from lora_system.lora_adapter import LoRAAdapter
from lora_system.lazarus_potential import lazarus_potential
from lora_system.kfac_fisher import kfac_fisher

def test_scaling():
    print("🧪 LAZARUS SCALING TEST")
    print("=======================")
    
    # 1. Fresh LoRA (Untrained)
    lora = LoRAAdapter().to('cpu')
    print("\n1) FRESH LoRA (Random Weights):")
    
    # Fake some fitness history for entropy
    lora.fitness_history = [0.5] * 10
    
    # Calculate
    fisher_data = kfac_fisher.compute_fisher_kfac(lora)
    lazarus_data = lazarus_potential.calculate_lazarus_lambda(lora, fisher_info_matrix=None)
    
    print(f"   Log-Det: {fisher_data['fisher_logdet']:.2f}")
    print(f"   Fisher Term: {lazarus_data['fisher_term']:.4f}")
    print(f"   Comment: {get_comment(lazarus_data['fisher_term'])}")
    
    # 2. Trained LoRA (Simulated)
    print("\n2) TRAINED LoRA (Larger Weights):")
    
    # Manually increase weights to simulate learning (larger covariance)
    with torch.no_grad():
        lora.fc1.lora_A.data *= 10.0
        lora.fc1.lora_B.data *= 10.0
        lora.fc2.lora_A.data *= 10.0
        lora.fc2.lora_B.data *= 10.0
        lora.fc3.lora_A.data *= 10.0
        lora.fc3.lora_B.data *= 10.0
        
    fisher_data = kfac_fisher.compute_fisher_kfac(lora)
    lazarus_data = lazarus_potential.calculate_lazarus_lambda(lora, fisher_info_matrix=None)
    
    print(f"   Log-Det: {fisher_data['fisher_logdet']:.2f}")
    print(f"   Fisher Term: {lazarus_data['fisher_term']:.4f}")
    print(f"   Comment: {get_comment(lazarus_data['fisher_term'])}")

def get_comment(fisher_term):
    if fisher_term < 0.50: return "Düşük Fisher - Az deneyim"
    elif fisher_term < 0.60: return "Orta Fisher - Standart"
    else: return "Yüksek Fisher - Çok öğrenmiş!"

if __name__ == "__main__":
    test_scaling()
