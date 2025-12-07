"""
🎒 LoRA KİŞİSEL CÜZDAN SİSTEMİ
===============================

Her LoRA'nın kendi kişisel dosyası.
Tüm geçmişi, genetik bilgisi, travmaları, sosyal bağları kaydedilir.

Arka planda çalışır, ana log'u kirletmez.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
import json


class LoRAWallet:
    """
    Bir LoRA'nın kişisel cüzdanı
    """
    
    def __init__(self, lora, wallet_dir: str = "lora_wallets"):
        self.lora = lora
        self.wallet_dir = wallet_dir
        os.makedirs(wallet_dir, exist_ok=True)
        
        # Dosya yolu - İSİM_ID FORMATI! (Excel'de de görünsün!)
        # WALLET İSMİ = LoRA İsmi + ID (diriltmede değişmez ama isimle bulabilirsin!)
        safe_name = lora.name.replace(' ', '_').replace('/', '_').replace('\\', '_')[:30]  # Max 30 karakter
        self.wallet_file = os.path.join(wallet_dir, f"{safe_name}_{lora.id}.txt")
        
        # Hareket geçmişi
        self.action_history = []
        
        # İlk oluşturma (sadece yoksa!)
        if not os.path.exists(self.wallet_file):
            self._create_initial_wallet()
    
    def _create_initial_wallet(self):
        """İlk cüzdan dosyasını oluştur (KİMLİK DETAYLI!)"""
        with open(self.wallet_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"🎒 LoRA KİŞİSEL CÜZDANI (KİMLİK BELGESİ)\n")
            f.write("="*80 + "\n")
            f.write(f"İsim: {self.lora.name}\n")
            f.write(f"ID: {self.lora.id}\n")
            f.write(f"Doğum Maçı: #{self.lora.birth_match}\n")
            f.write(f"Generasyon: {self.lora.generation}\n")
            f.write(f"🎭 Duygu Arketipi: {getattr(self.lora, 'emotional_archetype', 'Dengeli')}\n")
            f.write(f"Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # 🌳 SOYAĞACI
            f.write("🌳 SOYAĞACI:\n")
            f.write("─"*80 + "\n")
            if hasattr(self.lora, 'parents') and self.lora.parents:
                if len(self.lora.parents) >= 2:
                    f.write(f"  👨 Anne: {self.lora.parents[0]}\n")
                    f.write(f"  👩 Baba: {self.lora.parents[1]}\n")
                else:
                    f.write(f"  Ebeveyn: {', '.join(self.lora.parents)}\n")
            else:
                f.write(f"  İlk Nesil (Ebeveyn yok)\n")
            f.write("─"*80 + "\n\n")
            
            # 🎭 MİZAÇ (BAR GRAFİĞİ!)
            f.write("🎭 MİZAÇ PROFİLİ:\n")
            f.write("─"*80 + "\n")
            temp = self.lora.temperament
            
            # Her özellik için bar
            for key, value in temp.items():
                # Bar grafiği oluştur (10 karakter)
                bar_length = int(value * 10)
                bar = "█" * bar_length + "░" * (10 - bar_length)
                
                # İsim formatla
                key_formatted = key.replace('_', ' ').title()
                
                f.write(f"  {key_formatted:25s}: [{bar}] {value:.2f}\n")
            
            f.write("─"*80 + "\n\n")
            
            # 🔗 SOSYAL BAĞLAR (boş başlangıç)
            f.write("🔗 SOSYAL BAĞLAR:\n")
            f.write("─"*80 + "\n")
            f.write("  Henüz sosyal bağ yok (yeni doğdu)\n")
            f.write("─"*80 + "\n\n")
    
    def update_full_wallet(self, match_num: int, population: List = None):
        """
        Cüzdana kısa durum güncellemesi ekle (APPEND - TEMİZ!)
        
        NOT: Üzerine yazmaz, sadece ekler! Ölene kadar büyür!
        Her 20 maçta bir özet yaz.
        """
        
        # Her 20 maçta bir durum güncellemesi (çok sık olmasın!)
        if match_num % 20 != 0:
            return
        
        with open(self.wallet_file, 'a', encoding='utf-8') as f:  # ✅ APPEND MODE!
            f.write("\n" + "─"*80 + "\n")
            f.write(f"📊 DURUM RAPORU - Maç #{match_num} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write("─"*80 + "\n")
            f.write(f"⏳ Yaş: {match_num - self.lora.birth_match} maç | ")
            f.write(f"🎯 Fitness: {self.lora.get_recent_fitness():.3f} | ")
            f.write(f"🧬 Gen: {self.lora.generation}\n")
            
            # 🌊 PARÇACIK FİZİĞİ VERİLERİ (GÜNCEL!)
            f.write(f"\n🌊 PARÇACIK FİZİĞİ RAPORU:\n")
            f.write(f"   ⚡ Life Energy: {getattr(self.lora, 'life_energy', 1.0):.3f}\n")
            f.write(f"   🌡️  Sıcaklık (T): {getattr(self.lora, '_langevin_temp', 0.01):.4f}\n")
            f.write(f"   🛑 Sürtünme (ξ): {getattr(self.lora, '_nose_hoover_xi', 0.0):.3f}\n")
            f.write(f"   💨 Kinetik Enerji: {getattr(self.lora, '_kinetic_energy', 0.0):.3f}\n")
            f.write(f"   🌀 Opsiyon (Action): {getattr(self.lora, '_om_action', 0.0):.3f}\n")
            f.write(f"   🧟 Lazarus Λ: {getattr(self.lora, '_lazarus_lambda', 0.5):.3f}\n")
            f.write(f"   👻 Ghost Potansiyel: {getattr(self.lora, '_ghost_potential', 0.0):.3f}\n")
            
            # Performans özeti
            if len(self.lora.fitness_history) > 0:
                correct = sum(1 for f in self.lora.fitness_history if f > 0.5)
                f.write(f"📊 Son 20 maç: {correct}/{min(20, len(self.lora.fitness_history))} doğru\n")
            
            # Uzmanlık
            if hasattr(self.lora, 'specialization') and self.lora.specialization:
                f.write(f"🎯 Uzmanlık: {self.lora.specialization}\n")
            
            f.write("\n")
            
            # İlk 40 maçta detaylı bilgi göster (sonra gereksiz!)
            if match_num <= 40:
                f.write("\n🧬 GENETİK BİLGİ:\n")
                f.write("-"*30 + "\n")
            
            if hasattr(self.lora, 'parents') and len(self.lora.parents) > 0:
                if population:
                    parent1 = next((l for l in population if l.id == self.lora.parents[0]), None)
                    if parent1:
                        f.write(f"  Anne: {parent1.name}")
                        if hasattr(parent1, 'specialization') and parent1.specialization:
                            f.write(f" ({parent1.specialization})")
                        f.write(f" | Fitness: {parent1.get_recent_fitness():.3f}\n")
                    
                    if len(self.lora.parents) > 1:
                        parent2 = next((l for l in population if l.id == self.lora.parents[1]), None)
                        if parent2:
                            f.write(f"  Baba: {parent2.name}")
                            if hasattr(parent2, 'specialization') and parent2.specialization:
                                f.write(f" ({parent2.specialization})")
                            f.write(f" | Fitness: {parent2.get_recent_fitness():.3f}\n")
                
                # Kardeşler (aynı ebeveynler)
                if population:
                    siblings = [l for l in population 
                               if l.id != self.lora.id and 
                               hasattr(l, 'parents') and 
                               set(l.parents) == set(self.lora.parents)]
                    if siblings:
                        f.write(f"  Kardeşler: {', '.join([s.name for s in siblings[:5]])}\n")
                
                # Çocuklar
                if population:
                    children = [l for l in population 
                               if hasattr(l, 'parents') and 
                               self.lora.id in l.parents]
                    if children:
                        f.write(f"  Çocuklar: {', '.join([c.name for c in children[:10]])}\n")
                        if len(children) > 10:
                            f.write(f"           ... ve {len(children)-10} daha\n")
            else:
                f.write(f"  Doğum Tipi: Spontane/İlk Nesil (ebeveyn yok)\n")
            
            f.write("\n")
            
            # PERFORMANS
            f.write("📊 PERFORMANS:\n")
            f.write("-"*80 + "\n")
            f.write(f"  Toplam Maç: {len(self.lora.match_history)}\n")
            
            if len(self.lora.fitness_history) > 0:
                correct = sum(1 for f in self.lora.fitness_history if f > 0.5)
                wrong = len(self.lora.fitness_history) - correct
                f.write(f"  Doğru: {correct} (%{correct/len(self.lora.fitness_history)*100:.1f})\n")
                f.write(f"  Yanlış: {wrong} (%{wrong/len(self.lora.fitness_history)*100:.1f})\n")
            
            f.write(f"  Güncel Fitness: {self.lora.get_recent_fitness():.3f}\n")
            
            if len(self.lora.fitness_history) > 0:
                f.write(f"  En Yüksek Fitness: {max(self.lora.fitness_history):.3f}\n")
                f.write(f"  En Düşük Fitness: {min(self.lora.fitness_history):.3f}\n")
            
            f.write("\n")
            
            # UZMANLIK GEÇMİŞİ
            if hasattr(self.lora, 'specialization_history') and len(self.lora.specialization_history) > 0:
                f.write("🎯 UZMANLIK GEÇMİŞİ:\n")
                f.write("-"*80 + "\n")
                
                for i, spec in enumerate(self.lora.specialization_history, 1):
                    duration = "şimdi" if spec.end_match is None else f"{spec.end_match - spec.start_match} maç"
                    evolution_mark = " 🦋 (EVRİM)" if i > 1 else ""
                    
                    f.write(f"  {i}. {spec.specialization} (Maç #{spec.start_match}, süre: {duration}){evolution_mark}\n")
                
                if len(self.lora.specialization_history) > 1:
                    f.write(f"\n  → Bu LoRA {len(self.lora.specialization_history)-1} kez EVRİM GEÇİRDİ! 🦋\n")
                
                f.write(f"  → Şu anki Uzmanlık: {self.lora.specialization}\n")
                f.write("\n")
            
            # PATTERN PERFORMANSI
            if hasattr(self.lora, 'pattern_performance'):
                f.write("📈 PATTERN PERFORMANSI:\n")
                f.write("-"*80 + "\n")
                
                sorted_patterns = sorted(
                    self.lora.pattern_performance.items(),
                    key=lambda x: x[1]['correct'] / max(1, x[1]['total']),
                    reverse=True
                )
                
                for pattern, stats in sorted_patterns:
                    if stats['total'] > 0:
                        rate = stats['correct'] / stats['total']
                        star = "⭐" if rate > 0.75 else ""
                        f.write(f"  {pattern}: {stats['correct']}/{stats['total']} (%{rate*100:.0f}) {star}\n")
                
                f.write("\n")
            
            # SOSYAL BAĞLAR
            if hasattr(self.lora, 'social_bonds') and len(self.lora.social_bonds) > 0:
                f.write("🔗 SOSYAL BAĞLAR:\n")
                f.write("-"*80 + "\n")
                
                sorted_bonds = sorted(self.lora.social_bonds.items(), key=lambda x: abs(x[1]), reverse=True)
                
                for other_id, strength in sorted_bonds[:10]:
                    if population:
                        other = next((l for l in population if l.id == other_id), None)
                        if other:
                            bond_type = self._get_bond_emoji(strength)
                            f.write(f"  → {other.name} (çekim: {strength:+.2f}) {bond_type}\n")
                
                if len(sorted_bonds) > 10:
                    f.write(f"  ... ve {len(sorted_bonds)-10} bağ daha\n")
                
                f.write("\n")
            
            # TRAVMA GEÇMİŞİ
            if hasattr(self.lora, 'trauma_history'):
                # Trauma hem dict hem TraumaEvent olabilir
                severe_traumas = [t for t in self.lora.trauma_history 
                                 if (t.get('severity', 0) if isinstance(t, dict) else t.severity) > 0.3]
                
                if len(severe_traumas) > 0:
                    f.write("🩹 TRAVMA GEÇMİŞİ (Ciddi olanlar):\n")
                    f.write("-"*80 + "\n")
                    
                    for trauma in severe_traumas[-10:]:  # Son 10
                        # Trauma hem dict hem TraumaEvent olabilir
                        if isinstance(trauma, dict):
                            f.write(f"  • Maç #{trauma.get('timestamp', trauma.get('match', 0))}: {trauma.get('type', 'unknown')} (şiddet: {trauma.get('severity', 0):.2f})\n")
                        else:
                            f.write(f"  • Maç #{trauma.timestamp}: {trauma.type} (şiddet: {trauma.severity:.2f})\n")
                    
                    if len(severe_traumas) > 10:
                        f.write(f"  ... ve {len(severe_traumas)-10} travma daha\n")
                    
                    f.write(f"\n  Toplam Ciddi Travma: {len(severe_traumas)}\n")
                    f.write("\n")
            
            # MİZAÇ
            if hasattr(self.lora, 'temperament'):
                f.write("🧠 MİZAÇ:\n")
                f.write("-"*80 + "\n")
                
                for trait, value in self.lora.temperament.items():
                    bar = self._create_bar(value, 10)
                    f.write(f"  {trait.capitalize()}: [{bar}] {value:.2f}\n")
                
                f.write("\n")
            
            # HEDEFLER
            if hasattr(self.lora, 'main_goal') and self.lora.main_goal:
                f.write("🎯 HEDEFLER:\n")
                f.write("-"*80 + "\n")
                f.write(f"  Ana Hedef: {self.lora.main_goal.type}\n")
                f.write(f"  Heves: {self.lora.main_goal.heves:.2f}\n")
                f.write(f"  Sabır: {self.lora.main_goal.patience} maç\n")
                f.write(f"  İlerleme Durumu: {self.lora.main_goal.match_count_stuck} maç durgun\n")
                f.write("\n")
            
            # SON HAREKETLER
            f.write("📜 SON 20 HAREKET:\n")
            f.write("-"*80 + "\n")
            
            for action in self.action_history[-20:]:
                f.write(f"  {action}\n")
            
            if len(self.action_history) > 20:
                f.write(f"  ... toplam {len(self.action_history)} hareket kaydedildi\n")
            
            f.write("\n")
            
            # BAŞARILAR
            f.write("🏆 BAŞARILAR VE REKORLAR:\n")
            f.write("-"*80 + "\n")
            
            if len(self.lora.fitness_history) > 0:
                max_fitness_idx = self.lora.fitness_history.index(max(self.lora.fitness_history))
                f.write(f"  • En yüksek fitness: {max(self.lora.fitness_history):.3f} (Maç #{self.lora.birth_match + max_fitness_idx})\n")
            
            # Doğru streak hesapla
            if len(self.lora.fitness_history) > 0:
                current_streak = 0
                max_streak = 0
                
                for fit in self.lora.fitness_history:
                    if fit > 0.5:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 0
                
                if max_streak > 0:
                    f.write(f"  • En uzun doğru streak: {max_streak} maç\n")
            
            # Çocuk sayısı
            if population:
                children_count = len([l for l in population if hasattr(l, 'parents') and self.lora.id in l.parents])
                if children_count > 0:
                    f.write(f"  • Toplam çocuk: {children_count}\n")
            
            # Kara Veba'dan kurtulma
            if hasattr(self.lora, 'trauma_history'):
                # Kara veba hayatta kalmaları (hem dict hem TraumaEvent)
                kara_veba_survivals = [t for t in self.lora.trauma_history 
                                      if (t.get('type') if isinstance(t, dict) else t.type) == 'kara_veba']
                if kara_veba_survivals:
                    f.write(f"  • Kara Veba'dan {len(kara_veba_survivals)} kez hayatta kaldı! ☠️\n")
            
            f.write("\n")
            
            # RİSK DURUMU
            f.write("💀 RİSK DURUMU:\n")
            f.write("-"*80 + "\n")
            
            fitness = self.lora.get_recent_fitness()
            if fitness < 0.35:
                f.write(f"  ⚠️ KRİTİK! Ölüm riski yüksek (fitness: {fitness:.3f})\n")
            elif fitness < 0.50:
                f.write(f"  ⚠️ Orta risk (fitness: {fitness:.3f})\n")
            else:
                f.write(f"  ✅ Düşük risk (fitness: {fitness:.3f})\n")
            
            if hasattr(self.lora, 'goalless_death_risk'):
                if self.lora.goalless_death_risk > 0:
                    f.write(f"  🌀 Hedefsizlik riski: {self.lora.goalless_death_risk*100:.1f}%\n")
            
            f.write("\n")
            f.write("="*80 + "\n")
            f.write(f"Son Güncelleme: Maç #{match_num} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write("="*80 + "\n")
    
    def log_action(self, match_num: int, action_type: str, details: str):
        """Hareket kaydet"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        action = f"Maç #{match_num} [{timestamp}] {action_type}: {details}"
        
        self.action_history.append(action)
        
        # Dosyaya ekle (append)
        with open(self.wallet_file, 'a', encoding='utf-8') as f:
            f.write(f"{action}\n")
    
    def log_prediction(self, match_num: int, home_team: str, away_team: str, 
                      prediction: str, confidence: float, 
                      predicted_score: tuple = None,
                      actual: str = None, 
                      actual_score: tuple = None):
        """Tahmin kaydet (SKOR DETAYLI!)"""
        
        # Kazanan sonucu
        winner_result = ""
        if actual:
            winner_result = "✅ DOĞRU" if prediction == actual else "❌ YANLIŞ"
        
        # Skor sonucu
        score_result = ""
        if predicted_score and actual_score:
            if predicted_score == actual_score:
                score_result = "🎯 SKOR TAM!"
            elif abs((predicted_score[0] - predicted_score[1]) - (actual_score[0] - actual_score[1])) == 0:
                score_result = "✅ Gol farkı doğru"
            elif abs(predicted_score[0] - actual_score[0]) <= 1 and abs(predicted_score[1] - actual_score[1]) <= 1:
                score_result = "➖ Skor yakın"
            else:
                score_result = "❌ Skor uzak"
        
        # Detaylı log
        details = f"{home_team} vs {away_team}\n"
        details += f"      → Kazanan: {prediction} ({confidence*100:.0f}%)"
        if predicted_score:
            details += f" | Skor: {predicted_score[0]}-{predicted_score[1]}\n"
        else:
            details += "\n"
        
        if actual:
            details += f"      → Gerçek: {actual} {winner_result}"
            if actual_score:
                details += f" | {actual_score[0]}-{actual_score[1]} {score_result}"
        
        self.log_action(match_num, "TAHMİN", details)
    
    def log_learning(self, match_num: int, old_fitness: float, new_fitness: float):
        """Öğrenme kaydet"""
        
        change = new_fitness - old_fitness
        arrow = "↗️" if change > 0 else "↘️"
        
        details = f"Fitness: {old_fitness:.3f} → {new_fitness:.3f} ({change:+.3f}) {arrow}"
        self.log_action(match_num, "ÖĞRENME", details)
    
    def log_evolution_event(self, match_num: int, event_type: str, details: str):
        """Evrim olayı kaydet"""
        
        self.log_action(match_num, f"EVRİM-{event_type.upper()}", details)
    
    def _create_bar(self, value: float, max_blocks: int = 10):
        """ASCII bar"""
        filled = int(value * max_blocks)
        empty = max_blocks - filled
        return "█" * filled + "░" * empty
    
    def _get_bond_emoji(self, strength: float):
        """Bağ gücü emoji"""
        if strength > 0.8:
            return "💚"
        elif strength > 0.6:
            return "💙"
        elif strength > 0.4:
            return "💛"
        elif strength > 0.2:
            return "🧡"
        elif strength < 0:
            return "💔"
        else:
            return "🤍"


class WalletManager:
    """
    Tüm LoRA cüzdanlarını yönetir
    """
    
    def __init__(self, wallet_dir: str = "lora_wallets"):
        self.wallet_dir = wallet_dir
        os.makedirs(wallet_dir, exist_ok=True)
        
        self.wallets: Dict[str, LoRAWallet] = {}  # lora_id -> wallet
    
    def get_or_create_wallet(self, lora, population: List = None):
        """LoRA için cüzdan al veya oluştur"""
        
        if lora.id not in self.wallets:
            self.wallets[lora.id] = LoRAWallet(lora, self.wallet_dir)
        
        return self.wallets[lora.id]
    
    def update_all_wallets(self, population: List, match_num: int):
        """Tüm cüzdanları güncelle (snapshot)"""
        
        for lora in population:
            wallet = self.get_or_create_wallet(lora, population)
            wallet.update_full_wallet(match_num, population)
    
    def log_match_for_all(self, population: List, match_num: int, 
                         home_team: str, away_team: str, 
                         predictions: Dict, actual: str):
        """
        Her LoRA için tahmin ve sonuç kaydet
        
        predictions: {lora_id: (prediction, confidence)}
        """
        
        for lora in population:
            wallet = self.get_or_create_wallet(lora, population)
            
            if lora.id in predictions:
                pred, conf = predictions[lora.id]
                wallet.log_prediction(match_num, home_team, away_team, pred, conf, actual)
    
    def cleanup_dead_loras(self):
        """Ölen LoRA'ların cüzdanlarını arşivle"""
        
        # Arşiv klasörü
        archive_dir = os.path.join(self.wallet_dir, "archived_dead_loras")
        os.makedirs(archive_dir, exist_ok=True)
        
        # Cüzdanları arşivle
        # (Gelecekte: ölen LoRA'ları tespit edip taşı)
    
    def get_wallet_summary(self) -> str:
        """Tüm cüzdanlar özeti"""
        
        return f"💼 Toplam {len(self.wallets)} LoRA cüzdanı aktif"


