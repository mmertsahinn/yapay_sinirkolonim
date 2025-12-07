"""
🧪 DEEP LEARNING TEST SUITE
============================

10, 20, 30, 40, 50, 60, 70 maçlık testler
Her hatayı yakala ve deep learning ile çöz!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from lora_system.deep_collective_intelligence import get_deep_collective_intelligence
from lora_system.temperament_encoder import get_temperament_encoder
from lora_system.social_attention_layer import get_social_attention

def test_phase_1_modules():
    """Phase 1: Temperament ve Social Attention test"""
    print("\n" + "="*80)
    print("🧪 PHASE 1 TEST: Temperament Encoder + Social Attention")
    print("="*80)
    
    try:
        # Test temperament encoder
        temp_encoder = get_temperament_encoder(embed_dim=128)
        
        test_temperament = {
            'openness': 0.7,
            'contrarian_score': 0.3,
            'independence': 0.5,
            'risk_tolerance': 0.6,
            'hype_sensitivity': 0.4
        }
        
        emb = temp_encoder(test_temperament)
        print(f"✅ TemperamentEncoder: {emb.shape}")
        assert emb.shape == (128,), f"Wrong shape: {emb.shape}"
        
        # Test social attention
        social_attn = get_social_attention(embed_dim=128, num_heads=4)
        
        # Dummy LoRA embeddings
        N = 10
        lora_embs = torch.randn(N, 128)
        adjacency = torch.rand(N, N)
        adjacency = (adjacency + adjacency.T) / 2  # Symmetric
        
        updated_embs, attn_weights = social_attn(lora_embs, adjacency)
        print(f"✅ SocialAttention: {updated_embs.shape}, attention: {attn_weights.shape}")
        assert updated_embs.shape == (N, 128)
        assert attn_weights.shape[0] == N
        
        print("="*80)
        print("✅ PHASE 1 PASSED!")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"❌ PHASE 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_Phase_2_modules():
    """Phase 2: Knowledge Distillation + Meta-Learning"""
    print("\n" + "="*80)
    print("🧪 PHASE 2 TEST: Knowledge Distillation + Meta-Learning")
    print("="*80)
    
    try:
        from lora_system.knowledge_distillation import get_distillation
        from lora_system.meta_learning import get_meta_learner
        
        # Test distillation
        distill = get_distillation(embed_dim=128, temperature=2.0)
        
        teacher_emb = torch.randn(128)
        student_emb = torch.randn(128)
        
        distilled, soft_targets = distill.distill_knowledge(teacher_emb, student_emb)
        print(f"✅ DiscoveryDistillation: distilled={distilled.shape}, soft={soft_targets.shape}")
        assert distilled.shape == (128,)
        
        # Test meta-learner
        meta = get_meta_learner(embed_dim=128)
        
        parent_emb = torch.randn(128)
        child_emb = torch.randn(128)
        
        enhanced, strength = meta.inherit_from_parent(parent_emb, child_emb)
        print(f"✅ MetaLearner: enhanced={enhanced.shape}, strength={strength:.3f}")
        assert enhanced.shape == (128,)
        assert 0.0 <= strength <= 1.0
        
        print("="*80)
        print("✅ PHASE 2 PASSED!")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"❌ PHASE 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_master_system():
    """Phase 3: Master system integration"""
    print("\n" + "="*80)
    print("🧪 PHASE 3 TEST: Master Deep Collective Intelligence")
    print("="*80)
    
    try:
        # Initialize master system
        deep_collective = get_deep_collective_intelligence(
            embed_dim=128,
            num_heads=4,
            distillation_temp=2.0,
            device='cpu'
        )
        
        print(f"✅ Master system initialized")
        print(f"   Parameters: {sum(p.numel() for p in deep_collective.parameters()):,}")
        
        print("="*80)
        print("✅ PHASE 3 PASSED!")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"❌ PHASE 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "🌌"*40)
    print("DEEP LEARNING COLLECTIVE INTELLIGENCE - TEST SUITE")
    print("🌌"*40 + "\n")
    
    # Run all tests
    results = []
    
    results.append(("Phase 1", test_phase_1_modules()))
    results.append(("Phase 2", test_Phase_2_modules()))
    results.append(("Phase 3", test_master_system()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20} {status}")
    
    total_passed = sum(1 for _, p in results if p)
    total = len(results)
    
    print("="*80)
    print(f"TOTAL: {total_passed}/{total} tests passed")
    print("="*80)
    
    if total_passed == total:
        print("\n🎉 ALL TESTS PASSED! System ready for integration!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Fix errors before proceeding.")
        sys.exit(1)
