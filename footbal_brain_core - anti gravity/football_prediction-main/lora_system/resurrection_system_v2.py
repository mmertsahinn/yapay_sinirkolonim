"""
⚡ DİRİLTME SİSTEMİ V2 (3 Aşamalı)
===================================

50'ye tamamlama stratejisi:
1. Top 50 listesinden dirilt (ölüler önce)
2. Mucizelerden yükle
3. Rastgele spawn et
"""

import os
import torch
from typing import List
from datetime import datetime


class ResurrectionSystemV2:
    """
    3 aşamalı diriltme sistemi
    """
    
    def __init__(self):
        self.resurrection_count = {}  # {lora_id: dirilme_sayısı}
        # Sessiz başlatma (sadece --resurrect olunca mesaj ver)
    
    def resurrect_to_50(self, 
                        current_population: int,
                        target: int = 250,  # 🌊 BÜYÜK BAŞLANGIÇ!
                        export_dir: str = "en_iyi_loralar",
                        miracle_dir: str = "mucizeler",
                        device='cpu') -> tuple:
        """
        Hedef popülasyona tamamla (5 aşamalı)
        
        🌊 YENİ DEFAULT: 250 LoRA (Maksimum çeşitlilik!)
        
        Returns:
            (resurrected_loras, stats)
        """
        from .lora_adapter import LoRAAdapter
        from .miracle_system import MiracleSystem
        
        needed = target - current_population
        
        if needed <= 0:
            print(f"⚠️ Zaten yeterli LoRA var ({current_population})!")
            return [], {}
        
        print(f"\n{'⚡'*80}")
        print(f"⚡ DİRİLTME SİSTEMİ V2 (5 AŞAMALI)")
        print(f"{'⚡'*80}")
        print(f"Mevcut yaşayan: {current_population} LoRA")
        print(f"Hedef: {target} LoRA")
        print(f"Gereken: {needed} LoRA")
        print(f"{'─'*80}\n")
        
        all_resurrected = []
        stats = {
            'from_top_list': 0,
            'from_miracles': 0,
            'balanced_spawned': 0,
            'extreme_spawned': 0,
            'alien_spawned': 0
        }
        
        remaining = needed
        
        # ═══════════════════════════════════════════════════════════
        # AŞAMA 1: MUCİZELERDEN YÜKLE (ÖNCELİK!)
        # 🧟 LAZARUS Λ + PERFECT HYBRID BONUSU!
        # ═══════════════════════════════════════════════════════════
        print(f"🏆 AŞAMA 1: MUCİZE LoRA'LAR (🧟 LAZARUS Λ + 💎 HYBRID BONUSU!)")
        print(f"{'─'*80}")
        print(f"🔍 DEBUG: Lazarus skorları hesaplanıyor...")
        
        miracle_system = MiracleSystem(miracle_dir=miracle_dir)
        miracles = miracle_system.load_all_miracles(device=device)
        
        if miracles:
            # 🧟 LAZARUS Λ + PERFECT HYBRID ÖNCELIĞI!
            from .lazarus_potential import lazarus_potential
            from .tes_scoreboard import tes_scoreboard
            
            miracle_scores = []
            for lora in miracles:
                try:
                    # Lazarus Lambda hesapla
                    lazarus_data = lazarus_potential.calculate_lazarus_lambda(lora)
                    lazarus_lambda = lazarus_data['lambda']
                    
                    # 🆕 TES tipini kontrol et!
                    tes_data = tes_scoreboard.calculate_tes_score(lora, [], None)
                    lora_type = tes_data['lora_type']
                    
                    # 💎 PERFECT HYBRID BONUSU! (+0.3 Lazarus)
                    if 'PERFECT HYBRID💎💎💎' in lora_type:
                        bonus = 0.3
                    elif 'STRONG HYBRID🌟🌟' in lora_type:
                        bonus = 0.15
                    else:
                        bonus = 0.0
                    
                    final_score = lazarus_lambda + bonus
                    miracle_scores.append((lora, final_score, lazarus_lambda, lora_type))
                except:
                    miracle_scores.append((lora, 0.5, 0.5, 'UNKNOWN'))  # Default
            
            # Final score'a göre sırala (Lazarus + Hybrid bonusu!)
            miracle_scores.sort(key=lambda x: x[1], reverse=True)
            sorted_miracles = [item[0] for item in miracle_scores]
            
            to_load = min(remaining, len(sorted_miracles))
            print(f"   📊 Mucize sayısı: {len(sorted_miracles)}")
            print(f"   ⚡ Yüklenecek: {to_load} LoRA (🧟 Lazarus Λ + 💎 Hybrid bonusu sıralı!)\n")
            
            # 🔍 DEBUG: İlk 5'in skorlarını göster
            print(f"   🔍 DEBUG - İLK 5 MUCIZENIN SKORLARI:")
            for i, (lora, final, lam, typ) in enumerate(miracle_scores[:5], 1):
                bonus = final - lam
                print(f"      {i}. {lora.name[:25]:25s} | Λ:{lam:.3f} + Bonus:{bonus:.3f} = {final:.3f} | {typ[:30]}")
            
            for i, lora in enumerate(sorted_miracles[:to_load], 1):
                final_score, lazarus_lambda, lora_type = miracle_scores[i-1][1], miracle_scores[i-1][2], miracle_scores[i-1][3]
                
                # Hybrid tier göster
                hybrid_tag = ""
                if 'PERFECT HYBRID💎💎💎' in lora_type:
                    hybrid_tag = " 💎PERFECT!"
                elif 'STRONG HYBRID🌟🌟' in lora_type:
                    hybrid_tag = " 🌟STRONG"
                
                all_resurrected.append(lora)
                stats['from_miracles'] += 1
                remaining -= 1
                print(f"      {i}. 🏆 {lora.name} | Fit:{lora.get_recent_fitness():.3f} | 🧟 Λ:{lazarus_lambda:.3f}{hybrid_tag} | Skor:{final_score:.3f}")
            
            print(f"\n   ✅ {stats['from_miracles']} Mucize yüklendi!")
            print(f"   🔄 Kalan: {remaining}\n")
            
            # 🧟 DEBUG: Mucize diriltmeleri logla
            from .resurrection_debugger import resurrection_debugger
            miracle_scores_dict = {lora.id: (lam, final, typ) for lora, final, lam, typ in miracle_scores}
            resurrection_debugger.log_resurrection_batch(
                match_idx=0,  # Başlangıç diriltmesi
                resurrected_loras=[item[0] for item in miracle_scores[:stats['from_miracles']]],
                source='MIRACLES',
                lazarus_scores=miracle_scores_dict
            )
        else:
            print(f"   ⚠️ Henüz mucize LoRA yok!\n")
        
        # ═══════════════════════════════════════════════════════════
        # AŞAMA 2: SCOREBOARD'DAN DİRİLT
        # 🧟 LAZARUS Λ ÖNCEL İKLİ!
        # ═══════════════════════════════════════════════════════════
        if remaining > 0:
            print(f"📋 AŞAMA 2: SCOREBOARD'DAN DİRİLTME (🧟 LAZARUS Λ ÖNCELİKLİ!)")
            print(f"{'─'*80}")
            
            active_dir = os.path.join(export_dir, "⭐_AKTIF_EN_IYILER")
            
            if os.path.exists(active_dir):
                files = [f for f in os.listdir(active_dir) if f.endswith('.pt')]
                
                # 🧟 TÜM LoRA'LARI YÜKLE VE LAZARUS Λ HESAPLA!
                from .lazarus_potential import lazarus_potential
                
                lora_lambda_pairs = []
                for file in files:
                    try:
                        lora = self._load_lora_from_file(
                            os.path.join(active_dir, file),
                            device
                        )
                        if lora:
                            lazarus_data = lazarus_potential.calculate_lazarus_lambda(lora)
                            lora_lambda_pairs.append((lora, lazarus_data['lambda'], file))
                    except:
                        pass
                
                # 🆕 Λ + PERFECT HYBRID BONUSU ile sırala!
                lora_lambda_with_bonus = []
                for lora, lam, file in lora_lambda_pairs:
                    try:
                        tes_data = tes_scoreboard.calculate_tes_score(lora, [], None)
                        lora_type = tes_data['lora_type']
                        
                        # Perfect Hybrid bonusu
                        if 'PERFECT HYBRID💎💎💎' in lora_type:
                            bonus = 0.3
                        elif 'STRONG HYBRID🌟🌟' in lora_type:
                            bonus = 0.15
                        else:
                            bonus = 0.0
                        
                        final_score = lam + bonus
                        lora_lambda_with_bonus.append((lora, final_score, lam, lora_type, file))
                    except:
                        lora_lambda_with_bonus.append((lora, lam, lam, 'UNKNOWN', file))
                
                # Final score'a göre sırala!
                lora_lambda_with_bonus.sort(key=lambda x: x[1], reverse=True)
                
                dead_count = sum(1 for f in files if "💀" in f)
                alive_count = len(files) - dead_count
                
                print(f"   📊 Scoreboard'da: {len(files)} dosya")
                print(f"      💀 Ölü: {dead_count}")
                print(f"      ⭐ Yaşayan: {alive_count}")
                print(f"   🎯 Diriltme sırası: 🧟 LAZARUS Λ (Öğrenme kapasitesi!)")
                
                to_load = min(remaining, len(lora_lambda_pairs))
                print(f"   ⚡ Diriltilecek: {to_load} LoRA\n")
                
                for i, (lora, final_score, lazarus_lambda, lora_type, file) in enumerate(lora_lambda_with_bonus[:to_load], 1):
                    try:
                        # Lora zaten yüklü!
                        all_resurrected.append(lora)
                        stats['from_top_list'] += 1
                        remaining -= 1
                        
                        status = "💀" if "💀" in file else "⭐"
                        fitness = lora.original_fitness if hasattr(lora, 'original_fitness') else lora.get_recent_fitness()
                        # Hybrid tag göster
                        hybrid_tag = ""
                        if 'PERFECT HYBRID💎💎💎' in lora_type:
                            hybrid_tag = " 💎PERFECT!"
                        elif 'STRONG HYBRID🌟🌟' in lora_type:
                            hybrid_tag = " 🌟STRONG"
                        
                        print(f"      {i}. {status} {lora.name} | Fit:{fitness:.3f} | 🧟 Λ:{lazarus_lambda:.3f}{hybrid_tag} | Skor:{final_score:.3f}")
                    
                    except Exception as e:
                        print(f"      ❌ {file} yüklenemedi: {e}")
                
                print(f"\n   ✅ {stats['from_top_list']} LoRA dirildi!")
                print(f"   🔄 Kalan: {remaining}\n")
                
                # 🔍 DEBUG: Perfect Hybrid sayısını göster
                perfect_count = sum(1 for _, _, _, typ, _ in lora_lambda_with_bonus[:to_load] if 'PERFECT HYBRID💎💎💎' in typ)
                strong_count = sum(1 for _, _, _, typ, _ in lora_lambda_with_bonus[:to_load] if 'STRONG HYBRID🌟🌟' in typ)
                if perfect_count > 0 or strong_count > 0:
                    print(f"   🔍 DEBUG - Hybrid Dağılımı:")
                    print(f"      💎 Perfect: {perfect_count}")
                    print(f"      🌟 Strong: {strong_count}\n")
                
                # 🧟 DEBUG: Top list diriltmeleri logla
                from .resurrection_debugger import resurrection_debugger
                top_list_scores_dict = {lora.id: (lam, final, typ) for lora, final, lam, typ, _ in lora_lambda_with_bonus}
                resurrected_from_list = [item[0] for item in lora_lambda_with_bonus[:stats['from_top_list']]]
                resurrection_debugger.log_resurrection_batch(
                    match_idx=0,
                    resurrected_loras=resurrected_from_list,
                    source='TOP_LIST',
                    lazarus_scores=top_list_scores_dict
                )
            else:
                print(f"   ⚠️ Scoreboard klasörü bulunamadı!\n")
        
        # ═══════════════════════════════════════════════════════════
        # AŞAMA 3: DENGELİ KARAKTERLER SPAWN ET
        # ═══════════════════════════════════════════════════════════
        if remaining > 0:
            print(f"⚖️ AŞAMA 3: DENGELİ KARAKTERLER (Normal insanlar, orta seviye)")
            print(f"{'─'*80}")
            
            # Dengeli arketipleri al
            from .lora_archetypes import LoRAArchetypes
            balanced_archetypes = LoRAArchetypes.get_all_balanced_versions()
            
            # Kaç dengeli karakter spawn edilecek?
            balanced_count = min(remaining, len(balanced_archetypes))
            print(f"   ⚡ Spawn edilecek: {balanced_count} Dengeli LoRA\n")
            
            for i, (arch_key, arch_data) in enumerate(balanced_archetypes[:balanced_count], 1):
                # Dengeli karakter spawn et (SPAWN_TYPE = 'balanced')
                lora = self._spawn_random_lora(device, arch_key, arch_data, spawn_type='balanced')
                all_resurrected.append(lora)
                stats['balanced_spawned'] += 1
                remaining -= 1
                
                archetype_name = arch_data['name']
                archetype_desc = arch_data['description']
                
                print(f"      {i}. ⚖️ {lora.name}")
                print(f"         {archetype_desc}")
            
            print(f"\n   ✅ {stats['balanced_spawned']} Dengeli karakter spawn edildi!")
            print(f"   🔄 Kalan: {remaining}\n")
        
        # ═══════════════════════════════════════════════════════════
        # AŞAMA 4: UÇ ÖRNEKLER (ARKETİP) SPAWN ET
        # ═══════════════════════════════════════════════════════════
        if remaining > 0:
            print(f"🎭 AŞAMA 4: UÇ KARAKTERLER (Ekstrem arketipler)")
            print(f"{'─'*80}")
            
            # Uç karakterler için gerekli sayı hesapla
            # Eğer çok fazla kalırsa, bir kısmı uç, bir kısmı alien olacak
            extreme_count = min(remaining, 20)  # Max 20 uç karakter
            print(f"   ⚡ Spawn edilecek: {extreme_count} Uç LoRA\n")
            
            # ARKETİPLERİ SEÇ (ÇEŞİTLİLİK GARANTİLİ!)
            from .lora_archetypes import LoRAArchetypes
            archetypes = LoRAArchetypes.get_diverse_archetypes(extreme_count)
            
            for i, (arch_key, arch_data) in enumerate(archetypes, 1):
                # Arketip bazlı uç karakter spawn et (SPAWN_TYPE = 'extreme')
                lora = self._spawn_random_lora(device, arch_key, arch_data, spawn_type='extreme')
                all_resurrected.append(lora)
                stats['extreme_spawned'] += 1
                remaining -= 1
                
                archetype_emoji = arch_data['emoji']
                archetype_name = arch_data['name']
                archetype_desc = arch_data['description']
                
                print(f"      {i}. {archetype_emoji} {lora.name}")
                print(f"         Arketip: {archetype_name} - {archetype_desc}")
            
            print(f"\n   ✅ {stats['extreme_spawned']} Uç karakter spawn edildi!")
            print(f"   🔄 Kalan: {remaining}\n")
        
        # ═══════════════════════════════════════════════════════════
        # AŞAMA 5: ALIEN (NÖROTİPİK FARKLILIK) SPAWN ET
        # ═══════════════════════════════════════════════════════════
        if remaining > 0:
            print(f"👽 AŞAMA 5: ALIEN (Nörotipik farklılık, tahmin edilemez)")
            print(f"{'─'*80}")
            print(f"   ⚡ Spawn edilecek: {remaining} GERÇEK ALIEN LoRA\n")
            print(f"   💬 'Hiçbir arketipe uymuyorlar, tamamen rastgele!'")
            
            for i in range(remaining):
                # GERÇEK ALIEN: Hiçbir arketip yok!
                lora = self._spawn_random_lora(device, spawn_type='alien')
                all_resurrected.append(lora)
                stats['alien_spawned'] += 1  # Alien ayrı sayılır!
                
                print(f"      {i+1}. 👽 {lora.name} (Nörotipik farklılık)")
            
            print(f"\n   ✅ {remaining} Gerçek ALIEN spawn edildi!")
            remaining = 0
        
        # ═══════════════════════════════════════════════════════════
        # ÖZET
        # ═══════════════════════════════════════════════════════════
        print(f"\n{'═'*80}")
        print(f"✅ DİRİLTME TAMAMLANDI!")
        print(f"{'═'*80}")
        print(f"📊 ÖZET (ÖNCELİK SIRASINA GÖRE):")
        print(f"   🏆 1. Mucizelerden: {stats['from_miracles']} LoRA (hall of fame - en öncelikli!)")
        print(f"   📋 2. Scoreboard'dan: {stats['from_top_list']} LoRA (top liste)")
        print(f"   ⚖️ 3. Dengeli spawn: {stats['balanced_spawned']} LoRA (normal insanlar)")
        print(f"   🎭 4. Uç spawn: {stats['extreme_spawned']} LoRA (ekstrem arketip)")
        print(f"   👽 5. Alien spawn: {stats['alien_spawned']} LoRA (nörotipik farklılık)")
        print(f"   {'─'*76}")
        print(f"   ✅ TOPLAM YENİ: {len(all_resurrected)} LoRA")
        print(f"   👥 FİNAL POPÜLASYON: {current_population} (mevcut) + {len(all_resurrected)} (yeni) = {current_population + len(all_resurrected)}")
        print(f"{'═'*80}\n")
        
        return all_resurrected, stats
    
    def _load_lora_from_file(self, file_path: str, device='cpu'):
        """Dosyadan LoRA yükle"""
        from .lora_adapter import LoRAAdapter
        
        checkpoint = torch.load(file_path)
        meta = checkpoint['metadata']
        
        # LoRA oluştur
        lora = LoRAAdapter(input_dim=78, hidden_dim=128, rank=16, alpha=16.0, device=device)  # __init__ içinde .to(device) çağrılıyor
        lora.set_all_lora_params(checkpoint['lora_params'])
        
        # Metadata
        lora.id = meta['id']
        original_name = meta['name']
        
        # Dirilme sayısı
        if lora.id not in self.resurrection_count:
            self.resurrection_count[lora.id] = 0
        self.resurrection_count[lora.id] += 1
        
        resurrection_num = self.resurrection_count[lora.id]
        
        # Yeni isim
        if resurrection_num > 1:
            lora.name = f"Resurrected_{original_name}_x{resurrection_num}"
        else:
            lora.name = f"Resurrected_{original_name}"
        
        lora.generation = meta.get('generation', 0)
        lora.birth_match = 0  # YENİ BAŞLANGIÇ!
        lora.fitness_history = []
        lora.match_history = []
        lora.specialization = meta.get('specialization')
        temp = meta.get('temperament', {})
        if not isinstance(temp, dict):
            print(f"⚠️ UYARI: {lora.name} mizaç verisi bozuk (Tip: {type(temp)}) -> Sıfırlanıyor.")
            temp = {}
        lora.temperament = temp
        lora.parents = meta.get('parents', [])
        
        # Dirilme metadata
        lora.resurrection_count = resurrection_num
        lora.original_fitness = meta.get('fitness', 0.5)
        lora.was_dead = not meta.get('alive', True)
        
        return lora
    
    def _spawn_random_lora(self, device='cpu', archetype_key=None, archetype_data=None, spawn_type='alien'):
        """
        LoRA spawn et (ARKETİP BAZLI veya ALIEN!)
        
        Args:
            archetype_key: Arketip anahtarı (örn: "zen_master")
            archetype_data: Arketip verisi (emoji, temperament, vs)
            spawn_type: 'balanced', 'extreme', veya 'alien'
        """
        from .lora_adapter import LoRAAdapter
        import random
        
        lora = LoRAAdapter(input_dim=78, hidden_dim=128, rank=16, alpha=16.0, device=device)  # __init__ içinde .to(device) çağrılıyor
        
        # ID ve İSİM YAPISI
        random_num = random.randint(1000, 9999)
        
        # ARKETİP BAZLI İSİM VE MİZAÇ
        if archetype_data and spawn_type != 'alien':
            archetype_name = archetype_data['name'].replace(' ', '')
            
            if spawn_type == 'balanced':
                # DENGELİ: "Balanced_ZenMaster_347"
                lora.id = f"balanced_{archetype_key}_{random_num}"
                lora.name = f"Balanced_{archetype_name}_{random.randint(100, 999)}"
            else:
                # UÇ: "ZenMaster_234" (Alien değil!)
                lora.id = f"{archetype_key}_{random_num}"
                lora.name = f"{archetype_name}_{random.randint(100, 999)}"
            
            # Mizaç: Arketipten al (küçük varyasyon ekle!)
            lora.temperament = {}
            for key, base_value in archetype_data['temperament'].items():
                if spawn_type == 'balanced':
                    # Dengeli: ±5% varyasyon (çok yakın orta seviye)
                    variation = random.uniform(-0.05, 0.05)
                else:
                    # Uç: ±10% varyasyon (ekstrem kalsın!)
                    variation = random.uniform(-0.10, 0.10)
                
                final_value = max(0.0, min(1.0, base_value + variation))
                lora.temperament[key] = final_value
        else:
            # 👽 GERÇEK ALIEN: Hiçbir arketipe uymuyor!
            # Nörotipik farklılık, otizm spektrum, tahmin edilemez
            lora.id = f"alien_{random_num}"
            lora.name = f"Alien_{random_num}"
            
            # TAMAMEN RASTGELE MİZAÇ (hiçbir kural yok!)
            lora.temperament = {
                'patience': random.uniform(0.0, 1.0),
                'risk_tolerance': random.uniform(0.0, 1.0),
                'stress_tolerance': random.uniform(0.0, 1.0),
                'impulsiveness': random.uniform(0.0, 1.0),
                'hype_sensitivity': random.uniform(0.0, 1.0),
                'independence': random.uniform(0.0, 1.0),  # Tamamen rastgele!
                'social_intelligence': random.uniform(0.0, 1.0),  # Tamamen rastgele!
                'herd_tendency': random.uniform(0.0, 1.0),
                'contrarian_score': random.uniform(0.0, 1.0),
                'ambition': random.uniform(0.0, 1.0)
            }
        
        lora.generation = 0
        lora.birth_match = 0
        lora.parents = []
        
        return lora


# Global instance
resurrection_system_v2 = ResurrectionSystemV2()

