"""
🌍 GENEL UZMANLIK YÖNETİCİSİ
============================

Takıma özel değil, GENEL uzmanlar!

Uzmanlar:
- General_Win_Expert: Tüm maçlarda kazanan doğru bilir
- General_Goal_Expert: Tüm maçlarda golleri doğru bilir
- General_Hype_Expert: Tüm maçlarda hype doğru bilir

Bu LoRA'lar her maçta iyi! Takıma bağlı değil!
"""

import os
import torch
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict


class GlobalSpecializationManager:
    """
    Genel (takım-bağımsız) uzmanlık yöneticisi
    """
    
    def __init__(self, base_dir: str = "en_iyi_loralar"):
        self.base_dir = base_dir
        
        # Genel uzmanlar klasörü
        self.global_dir = os.path.join(base_dir, "🌍_GENEL_UZMANLAR")
        
        # Alt klasörler
        self.global_win_dir = os.path.join(self.global_dir, "🎯_WIN_EXPERTS")
        self.global_goal_dir = os.path.join(self.global_dir, "⚽_GOAL_EXPERTS")
        self.global_hype_dir = os.path.join(self.global_dir, "🔥_HYPE_EXPERTS")
        
        # Klasörleri oluştur
        for directory in [self.global_dir, self.global_win_dir, 
                         self.global_goal_dir, self.global_hype_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # İstatistikler (tüm maçlar için)
        self.all_match_stats = {
            'win_predictions': [],    # [(lora_id, correct, match_idx), ...]
            'goal_predictions': [],   # [(lora_id, predicted, actual, match_idx), ...]
            'hype_predictions': []    # [(lora_id, correct, match_idx), ...]
        }
        
        print(f"🌍 Genel Uzmanlık Yöneticisi başlatıldı: {self.global_dir}")
    
    def record_global_prediction(self, lora, predicted_winner: str, actual_winner: str,
                                 predicted_home_goals: int, predicted_away_goals: int,
                                 actual_home_goals: int, actual_away_goals: int,
                                 home_support: float, match_idx: int):
        """
        GENEL tahmin kaydet (takıma bakmaksızın!)
        """
        # Win
        win_correct = (predicted_winner == actual_winner)
        self.all_match_stats['win_predictions'].append((lora.id, win_correct, match_idx))
        
        # Goal (Home + Away MAE ortalaması)
        home_error = abs(predicted_home_goals - actual_home_goals)
        away_error = abs(predicted_away_goals - actual_away_goals)
        total_predicted = predicted_home_goals + predicted_away_goals
        total_actual = actual_home_goals + actual_away_goals
        
        self.all_match_stats['goal_predictions'].append(
            (lora.id, total_predicted, total_actual, match_idx)
        )
        
        # Hype
        hype_prediction = 'HOME' if home_support > 0.7 else ('AWAY' if home_support < 0.3 else 'NEUTRAL')
        hype_correct = (hype_prediction == actual_winner) if hype_prediction != 'NEUTRAL' else None
        
        if hype_correct is not None:
            self.all_match_stats['hype_predictions'].append((lora.id, hype_correct, match_idx))
    
    def calculate_global_specialization_scores(self, population: List, match_count: int) -> Dict:
        """
        GENEL uzmanları hesapla (Top 10!)
        
        Returns:
            {
                'win_experts': [(lora, score), ...],  # Top 10
                'goal_experts': [(lora, score), ...],
                'hype_experts': [(lora, score), ...]
            }
        """
        from lora_system.advanced_score_calculator import AdvancedScoreCalculator
        
        results = {
            'win_experts': [],
            'goal_experts': [],
            'hype_experts': []
        }
        
        for lora in population:
            lora_id = lora.id
            
            # 1) GENEL WIN EXPERT
            win_preds = [(correct, idx) for (lid, correct, idx) in self.all_match_stats['win_predictions'] if lid == lora_id]
            
            if len(win_preds) >= 50:  # Genel uzman için daha yüksek minimum (50 maç!)
                # GENEL ADVANCED SCORE HESAPLA
                score = AdvancedScoreCalculator.calculate_advanced_score(lora, match_count)
                results['win_experts'].append((lora, score))
            
            # 2) GENEL GOAL EXPERT
            goal_preds = [(pred, actual, idx) for (lid, pred, actual, idx) in self.all_match_stats['goal_predictions'] if lid == lora_id]
            
            if len(goal_preds) >= 50:
                # Goal MAE hesapla
                mae = np.mean([abs(pred - actual) for (pred, actual, _) in goal_preds])
                # MAE'yi skor'a çevir (düşük MAE = yüksek skor)
                goal_accuracy = max(0, 1 - (mae / 4.0))  # 0 MAE = 1.0, 4 MAE = 0.0
                
                # Advanced score ile birleştir
                base_score = AdvancedScoreCalculator.calculate_advanced_score(lora, match_count)
                # %50 goal accuracy, %50 advanced score
                score = (goal_accuracy * 0.5) + (base_score * 0.5)
                
                results['goal_experts'].append((lora, score))
            
            # 3) GENEL HYPE EXPERT
            hype_preds = [(correct, idx) for (lid, correct, idx) in self.all_match_stats['hype_predictions'] if lid == lora_id]
            
            if len(hype_preds) >= 50:
                hype_accuracy = sum(1 for (correct, _) in hype_preds if correct) / len(hype_preds)
                
                base_score = AdvancedScoreCalculator.calculate_advanced_score(lora, match_count)
                score = (hype_accuracy * 0.5) + (base_score * 0.5)
                
                results['hype_experts'].append((lora, score))
        
        # Top 10'a sırala (genel uzmanlar daha az, daha elit!)
        results['win_experts'].sort(key=lambda x: x[1], reverse=True)
        results['win_experts'] = results['win_experts'][:10]
        
        results['goal_experts'].sort(key=lambda x: x[1], reverse=True)
        results['goal_experts'] = results['goal_experts'][:10]
        
        results['hype_experts'].sort(key=lambda x: x[1], reverse=True)
        results['hype_experts'] = results['hype_experts'][:10]
        
        return results
    
    def export_global_specializations(self, specialization_results: Dict, match_count: int):
        """
        GENEL uzmanları export et (.pt + .txt)
        """
        # Win Experts
        self._export_global_type(
            self.global_win_dir,
            '🎯_WIN_EXPERTS',
            specialization_results['win_experts'],
            'GENEL_WIN',
            match_count
        )
        
        # Goal Experts
        self._export_global_type(
            self.global_goal_dir,
            '⚽_GOAL_EXPERTS',
            specialization_results['goal_experts'],
            'GENEL_GOAL',
            match_count
        )
        
        # Hype Experts
        self._export_global_type(
            self.global_hype_dir,
            '🔥_HYPE_EXPERTS',
            specialization_results['hype_experts'],
            'GENEL_HYPE',
            match_count
        )
        
        print(f"✅ Genel uzmanlıklar export edildi!")
    
    def _export_global_type(self, export_dir: str, dir_name: str,
                           experts: List[Tuple], spec_type: str, match_count: int):
        """
        Bir genel uzmanlık tipi için export
        """
        # .pt dosyaları
        for idx, (lora, score) in enumerate(experts, start=1):
            # Dosya adı: İsim_ID.pt (wallet ile aynı format)
            pt_filename = f"{lora.name}_{lora.id}.pt"
            pt_file = os.path.join(export_dir, pt_filename)
            
            torch.save({
                'lora_params': lora.get_all_lora_params(),
                'metadata': {
                    'id': lora.id,
                    'name': lora.name,
                    'pt_filename': pt_filename,  # 🆕 Dosya adı kaydet!
                    'specialization_type': spec_type,
                    'score': score,
                    'rank': idx,
                    'match_count': match_count,
                    'exported_at': datetime.now().isoformat(),
                    # ✅ TÜM FİZİK PARAMETRELERİ!
                    'life_energy': getattr(lora, 'life_energy', 1.0),
                    'lazarus_lambda': getattr(lora, '_lazarus_lambda', 0.5),
                    'tes_scores': getattr(lora, '_tes_scores', {}),
                    'temperament': getattr(lora, 'temperament', {}),
                    'particle_archetype': getattr(lora, '_particle_archetype', 'Unknown')
                }
            }, pt_file)
        
        # .txt scoreboard
        txt_file = os.path.join(export_dir, f"{dir_name.lower()}_top10.txt")
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"{spec_type} TOP 10 (GENEL UZMANLAR!)\n")
            f.write("="*80 + "\n")
            f.write(f"Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Maç: {match_count}\n")
            f.write("="*80 + "\n\n")
            
            f.write("💡 Bu LoRA'lar TÜAÇLARDA başarılı!\n")
            f.write("   Takıma bağlı değil, genel pattern'leri bilir!\n\n")
            f.write("="*80 + "\n\n")
            
            for idx, (lora, score) in enumerate(experts, start=1):
                # Ölümsüzlük
                from lora_system.death_immunity_system import calculate_death_immunity
                immunity, spec_count = calculate_death_immunity(lora, {})  # Cache boş (henüz hesaplanmadı)
                
                # Dosya adı
                pt_filename = f"{lora.name}_{lora.id}.pt"
                
                f.write("━"*80 + "\n")
                f.write(f"#{idx:02d} | {lora.name} | SKOR: {score:.3f}\n")
                f.write(f"📁 Dosya: {pt_filename}\n")
                f.write(f"🧟 Lazarus Λ: {getattr(lora, '_lazarus_lambda', 0.5):.3f}\n")
                f.write("━"*80 + "\n\n")
                
                f.write(f"📊 GENEL BAŞARI: {score:.3f}\n")
                f.write(f"   (TÜM maçlarda tutarlı başarı!)\n\n")
                
                f.write(f"⏳ DENEYİM:\n")
                f.write(f"   Yaş: {match_count - lora.birth_match} maç\n")
                f.write(f"   Nesil: {lora.generation}\n\n")
                
                f.write(f"🛡️ ÖLÜMSÜZLÜK:\n")
                f.write(f"   Diğer uzmanlıklar: {spec_count}\n")
                f.write(f"   Death immunity: {immunity*100:.1f}%\n\n")
                
                f.write("="*80 + "\n\n")


# Global instance
global_specialization_manager = GlobalSpecializationManager()

