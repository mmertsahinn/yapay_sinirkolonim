"""
🏆 TAKIM UZMANLIK YÖNETİCİSİ
=============================

Her takım için Top 5 uzman LoRA'ları yönetir:
- Win Experts (Kazanan tahmin)
- Goal Experts (Gol tahmin)
- Hype Experts (Hype doğruluk)
- VS Experts (Rakip bazlı)

Scoreboard tarzı, Advanced Score ile sıralama!
"""

import os
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict

# 🔄 JSON Serialization Helper for defaultdict
def default_team_stats():
    return {
        'win_predictions': [],
        'goal_predictions': [],
        'hype_predictions': [], 
        'vs_predictions': {}  # vs_predictions is a dict, handled dynamically
    }



class TeamSpecializationManager:
    """
    Takım uzmanlık sistemi yöneticisi
    """
    
    def __init__(self, base_dir: str = None):
        # 🆕 en_iyi_loralar altında olsun!
        if base_dir is None:
            base_dir = os.path.join("en_iyi_loralar", "takım_uzmanlıkları")
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Her takım için istatistikler
        self.team_stats = defaultdict(default_team_stats)
        
        # Persistence File
        self.state_file = os.path.join(self.base_dir, "team_specialization_memory.json")
        
        # Yükle
        self._load_state()
        
        # Top 5 listeler (cache)
        self.top_5_cache = {}
        
        print(f"🏆 Takım Uzmanlık Yöneticisi başlatıldı: {base_dir}")
        print(f"   📂 Hafıza dosyası: {self.state_file}")

    def _load_state(self):
        """Hafıza dosyasından yükle"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Dict -> DefaultDict conversion
                for team, stats in data.items():
                    self.team_stats[team] = stats
                    # VS predictions dict conversion (if needed)
                    if 'vs_predictions' in stats and isinstance(stats['vs_predictions'], dict):
                        # Convert list to dict if needed or keep as dict
                        pass
                        
                print(f"   ✅ {len(self.team_stats)} takımın hafızası yüklendi.")
            except Exception as e:
                print(f"   ⚠️ Hafıza yüklenemedi: {e}")

    def _save_state(self):
        """Hafızayı diske kaydet"""
        try:
            # DefaultDict -> Dict
            data = dict(self.team_stats)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # print(f"   💾 Takım hafızası kaydedildi.")
        except Exception as e:
            print(f"   ❌ Hafıza kaydedilemedi: {e}")

    
    def record_match_prediction(self, 
                                lora,
                                home_team: str,
                                away_team: str,
                                predicted_winner: str,
                                actual_winner: str,
                                predicted_home_goals: int,
                                predicted_away_goals: int,
                                actual_home_goals: int,
                                actual_away_goals: int,
                                home_support: float,
                                match_idx: int):
        """
        Maç tahminini kaydet (her LoRA için)
        """
        
        # 🛡️ DEDUPLICATION CHECK (Aynı maçı tekrar kaydetme!)
        # Home ve Away istatistiklerini kontrol et
        home_recorded = any(p[2] == match_idx for p in self.team_stats[home_team]['win_predictions'] if p[0] == lora.id)
        away_recorded = any(p[2] == match_idx for p in self.team_stats[away_team]['win_predictions'] if p[0] == lora.id)
        
        if home_recorded and away_recorded:
            return # Zaten kayıtlı!

        # Win prediction
        win_correct = (predicted_winner == actual_winner)
        
        # Goal prediction MAE
        home_goal_error = abs(predicted_home_goals - actual_home_goals)
        away_goal_error = abs(predicted_away_goals - actual_away_goals)
        
        # Hype prediction (home_support > 0.7 → ev sahibi kazanmalı)
        hype_prediction = 'HOME' if home_support > 0.7 else ('AWAY' if home_support < 0.3 else 'NEUTRAL')
        hype_correct = (hype_prediction == actual_winner) if hype_prediction != 'NEUTRAL' else None
        
        # Home team kayıt
        # (Sadece kayıtlı değilse ekle - yukarıda check yaptık ama çift dikiş gitmek iyidir)
        if not home_recorded:
            self.team_stats[home_team]['win_predictions'].append((lora.id, win_correct, match_idx))
            self.team_stats[home_team]['goal_predictions'].append((lora.id, predicted_home_goals, actual_home_goals, match_idx))
            if hype_correct is not None:
                self.team_stats[home_team]['hype_predictions'].append((lora.id, hype_correct, match_idx))
            # VS check (biraz pahalı ama gerekli)
            if away_team not in self.team_stats[home_team]['vs_predictions']:
                self.team_stats[home_team]['vs_predictions'][away_team] = []
                
            if not any(p[2] == match_idx for p in self.team_stats[home_team]['vs_predictions'][away_team] if p[0] == lora.id):
                self.team_stats[home_team]['vs_predictions'][away_team].append((lora.id, win_correct, match_idx))
        
        # Away team kayıt
        if not away_recorded:
            self.team_stats[away_team]['win_predictions'].append((lora.id, win_correct, match_idx))
            self.team_stats[away_team]['goal_predictions'].append((lora.id, predicted_away_goals, actual_away_goals, match_idx))
            
            if home_team not in self.team_stats[away_team]['vs_predictions']:
                self.team_stats[away_team]['vs_predictions'][home_team] = []
                
            if not any(p[2] == match_idx for p in self.team_stats[away_team]['vs_predictions'][home_team] if p[0] == lora.id):
                self.team_stats[away_team]['vs_predictions'][home_team].append((lora.id, win_correct, match_idx))
            if hype_correct is not None:
                self.team_stats[away_team]['hype_predictions'].append((lora.id, hype_correct, match_idx))
            # VS check
            if not any(p[2] == match_idx for p in self.team_stats[away_team]['vs_predictions'][home_team] if p[0] == lora.id):
                self.team_stats[away_team]['vs_predictions'][home_team].append((lora.id, win_correct, match_idx))
    
    def calculate_team_specialization_scores(self, population: List, match_count: int) -> Dict:
        """
        Tüm takımlar için Top 5 listelerini hesapla
        
        Returns:
            {
                'Manchester_United': {
                    'win_experts': [(lora, score), ...],  # Top 5
                    'goal_experts': [(lora, score), ...],
                    'hype_experts': [(lora, score), ...],
                    'vs_experts': {
                        'Liverpool': [(lora, score), ...],
                        ...
                    }
                },
                ...
            }
        """
        from lora_system.team_specialization_scorer import calculate_advanced_team_score
        
        results = {}
        
        for team_name, stats in self.team_stats.items():
            team_results = {
                'win_experts': [],
                'goal_experts': [],
                'hype_experts': [],
                'vs_experts': {}
            }
            
            # Her LoRA için skorları hesapla
            for lora in population:
                lora_id = lora.id
                
                # 1) WIN EXPERT SKORU
                win_preds = [(correct, idx) for (lid, correct, idx) in stats['win_predictions'] if lid == lora_id]
                if len(win_preds) >= 20:  # Minimum 20 maç
                    win_score = calculate_advanced_team_score(
                        lora, team_name, 'WIN', win_preds, match_count
                    )
                    team_results['win_experts'].append((lora, win_score))
                
                # 2) GOAL EXPERT SKORU
                goal_preds = [(pred, actual, idx) for (lid, pred, actual, idx) in stats['goal_predictions'] if lid == lora_id]
                if len(goal_preds) >= 20:
                    goal_score = calculate_advanced_team_score(
                        lora, team_name, 'GOAL', goal_preds, match_count
                    )
                    team_results['goal_experts'].append((lora, goal_score))
                
                # 3) HYPE EXPERT SKORU
                hype_preds = [(correct, idx) for (lid, correct, idx) in stats['hype_predictions'] if lid == lora_id]
                if len(hype_preds) >= 20:
                    hype_score = calculate_advanced_team_score(
                        lora, team_name, 'HYPE', hype_preds, match_count
                    )
                    team_results['hype_experts'].append((lora, hype_score))
                
                # 4) VS EXPERTS (Her rakip için)
                for opponent, vs_preds_all in stats['vs_predictions'].items():
                    vs_preds = [(correct, idx) for (lid, correct, idx) in vs_preds_all if lid == lora_id]
                    if len(vs_preds) >= 5:  # VS için minimum 5 maç yeterli (az eşleşme olur)
                        vs_score = calculate_advanced_team_score(
                            lora, team_name, f'VS_{opponent}', vs_preds, match_count
                        )
                        if opponent not in team_results['vs_experts']:
                            team_results['vs_experts'][opponent] = []
                        team_results['vs_experts'][opponent].append((lora, vs_score))
            
            # Top 5'e sırala
            team_results['win_experts'].sort(key=lambda x: x[1], reverse=True)
            team_results['win_experts'] = team_results['win_experts'][:5]
            
            team_results['goal_experts'].sort(key=lambda x: x[1], reverse=True)
            team_results['goal_experts'] = team_results['goal_experts'][:5]
            
            team_results['hype_experts'].sort(key=lambda x: x[1], reverse=True)
            team_results['hype_experts'] = team_results['hype_experts'][:5]
            
            for opponent in team_results['vs_experts']:
                team_results['vs_experts'][opponent].sort(key=lambda x: x[1], reverse=True)
                team_results['vs_experts'][opponent] = team_results['vs_experts'][opponent][:5]
            
            results[team_name] = team_results
        
        return results
    
    def export_team_specializations(self, specialization_results: Dict, match_count: int):
        """
        Top 5 listelerini dosyalara kaydet (.pt + .txt)
        """
        for team_name, team_data in specialization_results.items():
            # Takım klasörü
            team_dir = os.path.join(self.base_dir, self._safe_team_name(team_name))
            os.makedirs(team_dir, exist_ok=True)
            
            # 1) WIN EXPERTS
            self._export_expert_type(
                team_dir, 
                '🎯_WIN_EXPERTS', 
                team_data['win_experts'],
                team_name,
                'WIN',
                match_count
            )
            
            # 2) GOAL EXPERTS
            self._export_expert_type(
                team_dir,
                '⚽_GOAL_EXPERTS',
                team_data['goal_experts'],
                team_name,
                'GOAL',
                match_count
            )
            
            # 3) HYPE EXPERTS
            self._export_expert_type(
                team_dir,
                '🔥_HYPE_EXPERTS',
                team_data['hype_experts'],
                team_name,
                'HYPE',
                match_count
            )
            
            # 4) VS EXPERTS (Her rakip için)
            for opponent, vs_experts in team_data['vs_experts'].items():
                vs_dir_name = f'🆚_VS_{self._safe_team_name(opponent)}'
                self._export_expert_type(
                    team_dir,
                    vs_dir_name,
                    vs_experts,
                    team_name,
                    f'VS_{opponent}',
                    match_count
                )
            
            # 5) MASTER TXT (Takım özeti)
            self._create_team_master_txt(team_dir, team_name, team_data, match_count)
        
        # 6) STATE KAYDET (Persistence!)
        self._save_state()
        
        print(f"\n✅ Takım uzmanlıkları export edildi ve kaydedildi! ({len(specialization_results)} takım)")

    
    def _export_expert_type(self, team_dir: str, subdir_name: str, 
                           experts: List[Tuple], team_name: str, 
                           spec_type: str, match_count: int):
        """
        Bir uzmanlık tipi için klasör + .pt + .txt oluştur
        """
        expert_dir = os.path.join(team_dir, subdir_name)
        os.makedirs(expert_dir, exist_ok=True)
        
        # .pt dosyalarını kaydet (Top 5)
        for idx, (lora, score) in enumerate(experts, start=1):
            # Dosya adı: İsim_ID.pt (wallet ile aynı format)
            pt_filename = f"{lora.name}_{lora.id}.pt"
            pt_file = os.path.join(expert_dir, pt_filename)
            
            torch.save({
                'lora_params': lora.get_all_lora_params(),
                'metadata': {
                    'id': lora.id,
                    'name': lora.name,
                    'pt_filename': pt_filename,  # 🆕 Dosya adı kaydet!
                    'team': team_name,
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
        
        # .txt dosyası (scoreboard)
        txt_file = os.path.join(expert_dir, f"{subdir_name.lower()}_top5.txt")
        self._create_expert_txt(txt_file, experts, team_name, spec_type, match_count)
    
    def _create_expert_txt(self, txt_file: str, experts: List[Tuple],
                          team_name: str, spec_type: str, match_count: int):
        """
        Uzmanlık tipi için txt scoreboard oluştur
        """
        spec_emoji = {
            'WIN': '🎯',
            'GOAL': '⚽',
            'HYPE': '🔥'
        }
        emoji = spec_emoji.get(spec_type, '🆚')
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"{emoji} {team_name.upper()} - {spec_type} EXPERTS TOP 5\n")
            f.write("="*80 + "\n")
            f.write(f"Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Maç: {match_count}\n")
            f.write("="*80 + "\n\n")
            
            # 🆕 ÖZEL FORMÜL AÇIKLAMASI!
            f.write("📐 UZMANLIK SKORU FORMÜLÜ:\n")
            f.write("="*80 + "\n")
            if spec_type == 'WIN' or spec_type == 'HYPE' or spec_type.startswith('VS_'):
                f.write("SKOR = Accuracy (30%) + Age (20%) + Consistency (15%) +\n")
                f.write("       Peak (15%) + Momentum (10%) + Match Count (10%)\n\n")
                f.write("• Accuracy: Doğru tahmin yüzdesi (SADECE bu takımın maçlarında!)\n")
                f.write("• Age: LoRA'nın deneyimi (yaş normalizasyonu)\n")
                f.write("• Consistency: Son 20 maçtaki istikrar (SADECE bu takımda!)\n")
                f.write("• Peak: En iyi 10 maçlık dönem başarısı (SADECE bu takımda!)\n")
                f.write("• Momentum: İlk yarı vs İkinci yarı trend (SADECE bu takımda!)\n")
                f.write("• Match Count: Bu takım için tahmin sayısı bonusu\n\n")
            elif spec_type == 'GOAL':
                f.write("SKOR = Accuracy (30%) + Age (20%) + Consistency (15%) +\n")
                f.write("       Peak (15%) + Momentum (10%) + Match Count (10%)\n\n")
                f.write("• Accuracy: MAE (Mean Absolute Error) bazlı (SADECE bu takımın gollerinde!)\n")
                f.write("  - MAE 0.0 → 1.0 skor\n")
                f.write("  - MAE 3.0 → 0.0 skor\n")
                f.write("• Age: LoRA'nın deneyimi (yaş normalizasyonu)\n")
                f.write("• Consistency: Son 20 maçtaki gol tahmin istikrarı (SADECE bu takımda!)\n")
                f.write("• Peak: En iyi 10 maçlık dönem gol tahmin başarısı (SADECE bu takımda!)\n")
                f.write("• Momentum: İlk yarı vs İkinci yarı gol tahmin trendi (SADECE bu takımda!)\n")
                f.write("• Match Count: Bu takım için gol tahmin sayısı bonusu\n\n")
            f.write("🎯 ÖNEMLİ: Tüm metrikler SADECE bu takımın maçlarına bakıyor!\n")
            f.write("   Örn: Manchester uzmanı → Sadece Manchester maçları sayılır!\n")
            f.write("="*80 + "\n\n")
            
            if not experts:
                f.write("Henüz uzman yok (minimum 20 maç gerekli).\n")
                return
            
            for idx, (lora, score) in enumerate(experts, start=1):
                # Ölümsüzlük hesapla
                from lora_system.death_immunity_system import calculate_death_immunity
                immunity, spec_count = calculate_death_immunity(lora, self.top_5_cache)
                
                # Dosya adı
                pt_filename = f"{lora.name}_{lora.id}.pt"
                
                f.write("━"*80 + "\n")
                f.write(f"#{idx} | {lora.name} | SKOR: {score:.3f}\n")
                f.write(f"📁 Dosya: {pt_filename}\n")
                f.write(f"🧟 Lazarus Λ: {getattr(lora, '_lazarus_lambda', 0.5):.3f}\n")
                f.write("━"*80 + "\n\n")
                
                f.write(f"📊 UZMANLIK SKORU: {score:.3f}\n")
                f.write(f"   (Başarı + Deneyim + İstikrar + Peak + Momentum + Maç Sayısı)\n\n")
                
                f.write(f"⏳ DENEYİM:\n")
                f.write(f"   Yaş: {match_count - lora.birth_match} maç\n")
                f.write(f"   Nesil: {lora.generation}\n\n")
                
                f.write(f"🏅 DİĞER UZMANLIKLAR:\n")
                if spec_count > 0:
                    f.write(f"   Toplam {spec_count} uzmanlık!\n")
                    f.write(f"   ☠️ ÖLÜMSÜZLÜK: %{immunity*100:.0f}\n\n")
                    
                    if immunity >= 0.90:
                        f.write(f"   → ⭐ SÜPER UZMAN! Neredeyse ölümsüz!\n")
                    elif immunity >= 0.70:
                        f.write(f"   → 🏅 ÇOK UZMAN! Yüksek koruma!\n")
                    elif immunity >= 0.25:
                        f.write(f"   → 🎯 UZMAN! Orta koruma.\n")
                else:
                    f.write(f"   Sadece bu uzmanlık.\n")
                
                f.write(f"\n💾 DOSYA:\n")
                f.write(f"   {lora.id}.pt\n\n")
                
                f.write("="*80 + "\n\n")
    
    def _create_team_master_txt(self, team_dir: str, team_name: str, 
                                team_data: Dict, match_count: int):
        """
        Takım özet dosyası (MASTER)
        """
        master_file = os.path.join(team_dir, f"{self._safe_team_name(team_name)}_MASTER.txt")
        
        with open(master_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"🏆 {team_name.upper()} - UZMANLIK MASTER RAPORU\n")
            f.write("="*80 + "\n")
            f.write(f"Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Maç: {match_count}\n")
            f.write("="*80 + "\n\n")
            
            # 🆕 GENEL FORMÜL AÇIKLAMASI!
            f.write("📐 TAKIM UZMANLIK SİSTEMİ:\n")
            f.write("="*80 + "\n")
            f.write("Bu takım için 4 tip uzmanlık kategorisi var:\n\n")
            f.write("1. 🎯 WIN EXPERTS: Bu takımın kazanacağını en iyi tahmin edenler\n")
            f.write("   - Minimum 20 maç gerekli\n")
            f.write("   - Formül: Accuracy(30%) + Age(20%) + Consistency(15%) + Peak(15%) + Momentum(10%) + Match(10%)\n\n")
            f.write("2. ⚽ GOAL EXPERTS: Bu takımın atacağı golleri en iyi tahmin edenler\n")
            f.write("   - Minimum 20 maç gerekli\n")
            f.write("   - Formül: MAE Accuracy(30%) + Age(20%) + Consistency(15%) + Peak(15%) + Momentum(10%) + Match(10%)\n\n")
            f.write("3. 🔥 HYPE EXPERTS: Bu takımın hype'ını en iyi değerlendirenler\n")
            f.write("   - Minimum 20 maç gerekli\n")
            f.write("   - Formül: Accuracy(30%) + Age(20%) + Consistency(15%) + Peak(15%) + Momentum(10%) + Match(10%)\n\n")
            f.write("4. 🆚 VS EXPERTS: Bu takımın belirli bir rakip ile maçlarını en iyi tahmin edenler\n")
            f.write("   - Minimum 5 maç gerekli (az eşleşme olur)\n")
            f.write("   - Her rakip için ayrı klasör (örn: 🆚_VS_Liverpool)\n")
            f.write("   - Formül: Accuracy(30%) + Age(20%) + Consistency(15%) + Peak(15%) + Momentum(10%) + Match(10%)\n\n")
            f.write("🎯 ÖNEMLİ: Tüm metrikler SADECE bu takımın maçlarına bakıyor!\n")
            f.write("   Örn: Manchester WIN Expert → Sadece Manchester maçlarındaki kazanma tahminleri sayılır!\n")
            f.write("="*80 + "\n\n")
            
            # Win Experts
            f.write(f"🎯 WIN EXPERTS: {len(team_data['win_experts'])} uzman\n")
            for idx, (lora, score) in enumerate(team_data['win_experts'], 1):
                f.write(f"   #{idx}. {lora.name} (Skor: {score:.3f})\n")
            f.write("\n")
            
            # Goal Experts
            f.write(f"⚽ GOAL EXPERTS: {len(team_data['goal_experts'])} uzman\n")
            for idx, (lora, score) in enumerate(team_data['goal_experts'], 1):
                f.write(f"   #{idx}. {lora.name} (Skor: {score:.3f})\n")
            f.write("\n")
            
            # Hype Experts
            f.write(f"🔥 HYPE EXPERTS: {len(team_data['hype_experts'])} uzman\n")
            for idx, (lora, score) in enumerate(team_data['hype_experts'], 1):
                f.write(f"   #{idx}. {lora.name} (Skor: {score:.3f})\n")
            f.write("\n")
            
            # VS Experts
            f.write(f"🆚 VS EXPERTS: {len(team_data['vs_experts'])} rakip\n")
            for opponent, vs_experts in team_data['vs_experts'].items():
                f.write(f"\n   vs {opponent}:\n")
                for idx, (lora, score) in enumerate(vs_experts, 1):
                    f.write(f"      #{idx}. {lora.name} (Skor: {score:.3f})\n")
    
    def _safe_team_name(self, team_name: str) -> str:
        """Dosya sistemi için güvenli takım ismi"""
        return team_name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_')



def get_team_specialization_manager(base_dir: str = None):
    """Global instance getter (lazy init)"""
    global _team_spec_manager_instance
    if '_team_spec_manager_instance' not in globals():
        if base_dir is None:
            base_dir = os.path.join("en_iyi_loralar", "takım_uzmanlıkları")
        _team_spec_manager_instance = TeamSpecializationManager(base_dir=base_dir)
    return _team_spec_manager_instance


# Global instance
team_specialization_manager = get_team_specialization_manager()

