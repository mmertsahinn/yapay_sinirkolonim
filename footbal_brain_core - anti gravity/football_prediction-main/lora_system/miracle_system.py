"""
🏆 MUCIZE LoRA SİSTEMİ (HALL OF FAME)
=====================================

Olağanüstü performans gösteren LoRA'ları saklar.

MUCİZE KRİTERLERİ:
1. Fitness > 0.85 (mükemmel!)
2. Yaş > 100 maç (deneyimli!)
3. En az 1 evrim geçirmiş (adaptif!)
4. Özel başarılar:
   - Tam skor %30+ oranında
   - 10+ maç doğru streak
   - Kara Veba'dan 2+ kez kurtulmuş

KULLANIM:
- Ölen veya hibernation'a giren LoRA'lar kontrol edilir
- Kriterleri sağlarsa "mucizeler/" klasörüne kaydedilir
- Sistem sıfırlanırsa mucizeler geri yüklenebilir
- Mucizeler "ilk nesil" olarak geri gelir (deneyimleriyle!)
"""

import os
import json
import torch
from typing import List, Dict, Optional
from datetime import datetime


class MiracleSystem:
    """
    Mucize LoRA yönetimi
    """
    
    def __init__(self, miracle_dir: str = "mucizeler"):
        self.miracle_dir = miracle_dir
        os.makedirs(miracle_dir, exist_ok=True)
        
        # Mucize kayıt dosyası
        self.miracle_file = os.path.join(miracle_dir, "mucize_kayitlari.json")
        
        # Mevcut mucizeleri yükle
        self.miracles = {}
        if os.path.exists(self.miracle_file):
            with open(self.miracle_file, 'r', encoding='utf-8') as f:
                self.miracles = json.load(f)
        
        print(f"🏆 Mucize Sistemi başlatıldı: {miracle_dir}")
        if self.miracles:
            print(f"   📚 {len(self.miracles)} mucize LoRA kayıtlı")
    
    def check_miracle_criteria(self, lora, match_count: int, 
                              specialization_count: int = 0) -> Dict:
        """
        LoRA mucize kriterlerini sağlıyor mu?
        
        🏆 3 KATMANLI MUCİZE SİSTEMİ:
        1. POTANSIYEL MUCİZE (Yolda Olanlar - Genç Yetenekler)
        2. MUCİZE (Deneyimli Başarılılar)
        3. YÜCE MUCİZE (Efsaneler - En Üst Seviye!)
        
        🌟 YENİ: ÇOKLU UZMANLIK = OTOMATİK MUCİZE!
        - 5+ uzmanlık → Doğrudan MUCİZE!
        - 7+ uzmanlık → Doğrudan YÜCE MUCİZE!
        
        Args:
            specialization_count: Kaç uzmanlığı var? (takım + genel)
        
        Returns:
            {
                'is_miracle': True/False,
                'miracle_tier': 'POTANSIYEL' | 'MUCIZE' | 'YUCE_MUCIZE' | None,
                'score': 0.0-1.0,
                'reasons': ['...', '...']
            }
        """
        age = match_count - lora.birth_match
        fitness = lora.get_recent_fitness()
        
        score = 0.0
        reasons = []
        
        # 1) FİTNESS (0-40 puan)
        if fitness > 0.85:
            fitness_points = 40
            reasons.append(f"🌟 Mükemmel fitness ({fitness:.3f})")
        elif fitness > 0.75:
            fitness_points = 30
            reasons.append(f"⭐ Çok iyi fitness ({fitness:.3f})")
        elif fitness > 0.65:
            fitness_points = 20
        else:
            fitness_points = 0
        
        score += fitness_points
        
        # 2) YAŞ (0-20 puan)
        if age > 200:
            age_points = 20
            reasons.append(f"👴 Çok deneyimli ({age} maç)")
        elif age > 100:
            age_points = 15
            reasons.append(f"🧓 Deneyimli ({age} maç)")
        elif age > 50:
            age_points = 10
        else:
            age_points = 0
        
        score += age_points
        
        # 3) EVRİM (0-15 puan)
        if hasattr(lora, 'specialization_history') and len(lora.specialization_history) > 1:
            evolutions = len(lora.specialization_history) - 1
            evolution_points = min(15, evolutions * 5)
            reasons.append(f"🦋 {evolutions} kez evrimleşti")
            score += evolution_points
        
        # 4) STREAK BAŞARILARI (0-15 puan)
        max_streak = 0  # ✅ DÜZELTME: Her zaman tanımlı olmalı!
        # Her zaman hesapla (sadece > 10 değil!)
        current_streak = 0
        for fit in lora.fitness_history:
            if fit > 0.5:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        # Puan ver (sadece uzun streak'lere)
        if len(lora.fitness_history) > 10:
            if max_streak >= 20:
                streak_points = 15
                reasons.append(f"🔥 {max_streak} maç üst üste doğru!")
            elif max_streak >= 10:
                streak_points = 10
                reasons.append(f"🔥 {max_streak} maç streak")
            else:
                streak_points = 0
            
            score += streak_points
        
        # 5) TRAVMA HAYATTA KALMA (0-10 puan)
        if hasattr(lora, 'trauma_history'):
            kara_veba_survivals = len([t for t in lora.trauma_history 
                                      if hasattr(t, 'type') and 'veba' in str(t.type).lower()])
            if kara_veba_survivals >= 3:
                trauma_points = 10
                reasons.append(f"☠️ {kara_veba_survivals} Kara Veba'dan kurtuldu!")
            elif kara_veba_survivals >= 1:
                trauma_points = 5
                reasons.append(f"☠️ Kara Veba'dan kurtuldu")
            else:
                trauma_points = 0
            
            score += trauma_points
        
        # Normalize: 100 puan üzerinden
        normalized_score = score / 100.0
        
        # ============================================
        # 🏆 3 KATMANLI MUCİZE SİSTEMİ!
        # ============================================
        
        is_miracle = False
        miracle_tier = None
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🌟 YENİ: ÇOKLU UZMANLIK = OTOMATİK MUCİZE!
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if specialization_count >= 7:
            # 7+ uzmanlık → YÜCE MUCİZE!
            is_miracle = True
            miracle_tier = 'YUCE_MUCIZE'
            reasons.append(f"👑 ÇOKLU SÜPER UZMAN! ({specialization_count} uzmanlık!)")
            reasons.append("🌟 7+ uzmanlık → Otomatik YÜCE MUCİZE!")
            
            # Erken dön (artık diğer kriterlere bakmaya gerek yok)
            return {
                'is_miracle': True,
                'miracle_tier': 'YUCE_MUCIZE',
                'score': 1.0,  # Maksimum!
                'total_points': 100,
                'reasons': reasons,
                'fitness': fitness,
                'age': age
            }
        
        elif specialization_count >= 5:
            # 5-6 uzmanlık → MUCİZE!
            is_miracle = True
            miracle_tier = 'MUCIZE'
            reasons.append(f"🏆 ÇOKLU UZMAN! ({specialization_count} uzmanlık!)")
            reasons.append("🌟 5+ uzmanlık → Otomatik MUCİZE!")
            
            return {
                'is_miracle': True,
                'miracle_tier': 'MUCIZE',
                'score': 0.9,
                'total_points': 90,
                'reasons': reasons,
                'fitness': fitness,
                'age': age
            }
        
        elif specialization_count >= 3:
            # 3-4 uzmanlık → Bonus puan!
            score += 20  # +20 puan bonus
            reasons.append(f"🎯 ÇOK UZMAN! ({specialization_count} uzmanlık, +20 bonus!)")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1️⃣ POTANSIYEL MUCİZE (Genç Yetenekler + Erken Ölenler!)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if age <= 15:  # Çok genç!
            # Art arda 5+ maç başarılı + yüksek oran
            if max_streak >= 5 and fitness >= 0.90:
                is_miracle = True
                miracle_tier = 'POTANSIYEL'
                reasons.append("🌱 POTANSIYEL MUCİZE! (Genç + Art arda 5+ + %90+)")
            
            # Veya süper başarı (daha kısa streak ama mükemmel)
            elif fitness >= 0.95 and max_streak >= 3:
                is_miracle = True
                miracle_tier = 'POTANSIYEL'
                reasons.append("🌱 POTANSIYEL MUCİZE! (Genç + Mükemmel %95+)")
        
        # 🆕 ERKEN ÖLENLER (20-80 yaş arası, yüksek potansiyel!)
        elif age > 15 and age <= 80:  # Erken ölüm ama deneyimli!
            # Başarı oranı hesapla (fitness_history'den)
            if len(lora.fitness_history) >= 10:
                success_rate = sum(1 for f in lora.fitness_history if f > 0.5) / len(lora.fitness_history)
                
                # Yüksek streak + yüksek başarı = POTANSIYEL!
                if max_streak >= 7 and success_rate >= 0.70:
                    is_miracle = True
                    miracle_tier = 'POTANSIYEL'
                    reasons.append(f"🌱 POTANSIYEL! (Erken öldü ama {max_streak} streak + %{success_rate*100:.0f} başarı!)")
                
                # Veya çok yüksek başarı + orta streak
                elif max_streak >= 5 and success_rate >= 0.75:
                    is_miracle = True
                    miracle_tier = 'POTANSIYEL'
                    reasons.append(f"🌱 POTANSIYEL! (Erken öldü ama yüksek başarı: %{success_rate*100:.0f}, streak {max_streak})")
                
                # Veya süper yüksek başarı (streak az da olsa)
                elif success_rate >= 0.80 and max_streak >= 4:
                    is_miracle = True
                    miracle_tier = 'POTANSIYEL'
                    reasons.append(f"🌱 POTANSIYEL! (Erken öldü ama mükemmel: %{success_rate*100:.0f}!)")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2️⃣ MUCİZE (Deneyimli Başarılılar!)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        elif age >= 50 and age < 150:  # Deneyimli ama çok yaşlı değil
            # Çok iyi fitness + uzun streak
            if fitness >= 0.80 and max_streak >= 15:
                is_miracle = True
                miracle_tier = 'MUCIZE'
                reasons.append("🏆 MUCİZE! (Deneyimli + İstikrarlı + Streak 15+)")
            
            # Veya mükemmel fitness + yaş kombinasyonu
            elif fitness >= 0.85 and score >= 70:
                is_miracle = True
                miracle_tier = 'MUCIZE'
                reasons.append("🏆 MUCİZE! (Deneyim + Mükemmellik)")
            
            # Travma survivor
            elif score >= 65 and 'Kara Veba' in ' '.join(reasons):
                is_miracle = True
                miracle_tier = 'MUCIZE'
                reasons.append("🏆 MUCİZE! (Travma Survivor + Deneyim)")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3️⃣ YÜCE MUCİZE (Efsaneler!)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        elif age >= 150:  # Çok deneyimli!
            # Efsane kriterleri: Yaş + Fitness + Streak
            if fitness >= 0.85 and max_streak >= 20:
                is_miracle = True
                miracle_tier = 'YUCE_MUCIZE'
                reasons.append("👑 YÜCE MUCİZE! (Efsane Yaş + Mükemmel + Streak 20+)")
            
            # Veya çok yüksek toplam puan
            elif score >= 80:
                is_miracle = True
                miracle_tier = 'YUCE_MUCIZE'
                reasons.append("👑 YÜCE MUCİZE! (Efsane Toplam Puan 80+)")
            
            # Veya uzun yaşam + iyi fitness
            elif age >= 250 and fitness >= 0.75:
                is_miracle = True
                miracle_tier = 'YUCE_MUCIZE'
                reasons.append("👑 YÜCE MUCİZE! (250+ Maç Yaşadı + İyi Performans)")
        
        return {
            'is_miracle': is_miracle,
            'miracle_tier': miracle_tier,  # 🆕 KATMAN!
            'score': normalized_score,
            'total_points': score,
            'reasons': reasons,
            'fitness': fitness,
            'age': age
        }
    
    def save_miracle(self, lora, match_count: int, criteria_result: Dict):
        """
        Mucize LoRA'yı kaydet (3 KATMANLI!)
        
        Klasörler:
        - mucizeler/🌱_POTANSIYEL/
        - mucizeler/🏆_MUCIZE/
        - mucizeler/👑_YUCE_MUCIZE/
        """
        miracle_tier = criteria_result.get('miracle_tier', 'MUCIZE')
        
        # Katmana göre klasör seç
        tier_folders = {
            'POTANSIYEL': '🌱_POTANSIYEL',
            'MUCIZE': '🏆_MUCIZE',
            'YUCE_MUCIZE': '👑_YUCE_MUCIZE'
        }
        tier_folder = tier_folders.get(miracle_tier, '🏆_MUCIZE')
        tier_dir = os.path.join(self.miracle_dir, tier_folder)
        os.makedirs(tier_dir, exist_ok=True)
        
        miracle_id = f"{lora.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # LoRA parametrelerini kaydet (katman klasörüne!)
        miracle_path = os.path.join(tier_dir, f"{miracle_id}.pt")
        
        torch.save({
            'lora_params': lora.get_all_lora_params(),
            'metadata': {
                'id': lora.id,
                'name': lora.name,
                'generation': lora.generation,
                'birth_match': lora.birth_match,
                'death_match': match_count,
                'age': match_count - lora.birth_match,
                'final_fitness': lora.get_recent_fitness(),
                'fitness_history': lora.fitness_history,
                'match_history': lora.match_history,
                'specialization': getattr(lora, 'specialization', None),
                'specialization_history': getattr(lora, 'specialization_history', []),
                'temperament': getattr(lora, 'temperament', {}),
                'parents': getattr(lora, 'parents', []),
                'miracle_score': criteria_result['total_points'],
                'miracle_tier': miracle_tier,  # 🆕 KATMAN!
                'miracle_reasons': criteria_result['reasons'],
                'saved_at': datetime.now().isoformat()
            }
        }, miracle_path)
        
        # Kayıt dosyasına ekle
        self.miracles[miracle_id] = {
            'name': lora.name,
            'miracle_tier': miracle_tier,  # 🆕 KATMAN!
            'specialization': getattr(lora, 'specialization', None),
            'fitness': lora.get_recent_fitness(),
            'age': match_count - lora.birth_match,
            'miracle_score': criteria_result['total_points'],
            'reasons': criteria_result['reasons'],
            'saved_at': datetime.now().isoformat(),
            'file': f"{tier_folder}/{miracle_id}.pt"  # Katman klasörü dahil!
        }
        
        # JSON güncelle
        with open(self.miracle_file, 'w', encoding='utf-8') as f:
            json.dump(self.miracles, f, indent=2, ensure_ascii=False)
        
        # Katmana göre emoji
        tier_emoji = {'POTANSIYEL': '🌱', 'MUCIZE': '🏆', 'YUCE_MUCIZE': '👑'}.get(miracle_tier, '🏆')
        
        print(f"\n{tier_emoji*40}")
        print(f"{tier_emoji} {miracle_tier} LoRA KAYDEDİLDİ!")
        print(f"{tier_emoji*40}")
        print(f"  • İsim: {lora.name}")
        print(f"  • Katman: {miracle_tier}")
        print(f"  • Fitness: {lora.get_recent_fitness():.3f}")
        print(f"  • Yaş: {match_count - lora.birth_match} maç")
        print(f"  • Uzmanlık: {getattr(lora, 'specialization', 'Genel')}")
        print(f"  • Mucize Puanı: {criteria_result['total_points']}/100")
        print(f"  • Sebepler:")
        for reason in criteria_result['reasons']:
            print(f"      - {reason}")
        print(f"  • Klasör: {tier_folder}/")
        print(f"  • Dosya: {miracle_id}.pt")
        print(f"{tier_emoji*40}\n")
        
        return miracle_id
    
    def load_all_miracles(self, device='cpu') -> List:
        """
        Tüm mucize LoRA'ları yükle (yeniden başlangıç için!)
        
        Returns:
            List of LoRAAdapter instances
        """
        from .lora_adapter import LoRAAdapter
        
        miracles = []
        
        for miracle_id, info in self.miracles.items():
            miracle_path = os.path.join(self.miracle_dir, info['file'])
            
            if not os.path.exists(miracle_path):
                print(f"   ⚠️ Mucize dosyası bulunamadı: {info['file']}")
                continue
            
            try:
                checkpoint = torch.load(miracle_path)
                meta = checkpoint['metadata']
                
                # LoRA oluştur (artık __init__ içinde .to(device) çağrılıyor)
                lora = LoRAAdapter(input_dim=78, hidden_dim=128, rank=16, alpha=16.0, device=device)
                lora.set_all_lora_params(checkpoint['lora_params'])
                
                # Temperament eksik anahtarları düzelt
                default_temperament = {
                    'independence': 0.6, 'social_intelligence': 0.6, 'herd_tendency': 0.4, 'contrarian_score': 0.3,
                    'emotional_depth': 0.5, 'empathy': 0.5, 'anger_tendency': 0.5,
                    'ambition': 0.6, 'competitiveness': 0.5, 'resilience': 0.6, 'will_to_live': 0.7,
                    'patience': 0.6, 'impulsiveness': 0.4, 'stress_tolerance': 0.6, 'risk_appetite': 0.5
                }
                for key, default_value in default_temperament.items():
                    if key not in lora.temperament:
                        lora.temperament[key] = default_value
                
                # Metadata'yı geri yükle
                lora.id = meta['id']
                lora.name = f"Legend_{meta['name']}"  # 🏆 LEGEND prefix!
                lora.generation = 0  # Yeni nesil olarak başlar
                lora.birth_match = 0
                lora.fitness_history = []  # Sıfırdan başlar ama deneyimli!
                lora.match_history = []
                lora.specialization = meta.get('specialization')
                temp = meta.get('temperament', {})
                if not isinstance(temp, dict):
                    print(f"⚠️ UYARI: {lora.name} mizaç verisi bozuk (Tip: {type(temp)}) -> Sıfırlanıyor.")
                    temp = {}
                lora.temperament = temp
                lora.parents = []  # İlk nesil gibi (ama legend!)
                
                miracles.append(lora)
                
                print(f"   🏆 {lora.name} yüklendi! (Eski fitness: {meta['final_fitness']:.3f}, {meta['age']} maç)")
            
            except Exception as e:
                print(f"   ❌ {info['file']} yüklenemedi: {e}")
        
        return miracles
    
    def get_all_miracle_ids(self) -> List[str]:
        """
        Tüm mucize LoRA ID'lerini döndür (Elite kontrolü için)
        
        Returns:
            List of miracle LoRA IDs
        """
        return [m['id'] for m in self.miracles]
    
    def get_miracle_summary(self) -> str:
        """Mucizeler özeti"""
        if not self.miracles:
            return "Henüz mucize LoRA yok."
        
        summary = f"\n{'🏆'*40}\n"
        summary += f"HALL OF FAME - MUCİZE LoRA'LAR\n"
        summary += f"{'🏆'*40}\n\n"
        
        sorted_miracles = sorted(
            self.miracles.items(),
            key=lambda x: x[1]['miracle_score'],
            reverse=True
        )
        
        for i, (mid, info) in enumerate(sorted_miracles, 1):
            summary += f"{i}. {info['name']}\n"
            summary += f"   • Fitness: {info['fitness']:.3f}\n"
            summary += f"   • Yaş: {info['age']} maç\n"
            summary += f"   • Uzmanlık: {info['specialization']}\n"
            summary += f"   • Mucize Puanı: {info['miracle_score']}/100\n"
            summary += f"   • Sebep: {', '.join(info['reasons'][:3])}\n"
            summary += f"   • Kayıt: {info['saved_at'][:10]}\n\n"
        
        return summary


# Global instance
miracle_system = MiracleSystem()

