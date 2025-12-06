"""
🌐 ORTAK HAFIZA SİSTEMİ (COLLECTIVE INTELLIGENCE)
=================================================

Tüm LoRA'ların düşüncelerini, tahminlerini ve sonuçlarını kaydeder.
Her LoRA bu hafızayı okuyarak:
- Diğer LoRA'ların güvenilirliğini öğrenir
- Kim hangi pattern'de iyi?
- Kim çok emin ama yanılıyor? (Overconfident)
- Consensus nedir? Consensus'a uymak mantıklı mı?

LoRA'lar kişiliklerine göre bu hafızayı farklı yorumlar:
- Bağımsız: "Ben kendi kafama göre yaparım!" (az kullanır)
- Sosyal Zeki: "Kim güvenilir analiz ederim!" (çok iyi kullanır)
- Sürü: "Çoğunluk ne diyorsa o!" (körü körüne kullanır)
- Karşıt: "Çoğunluğun tersi!" (ters kullanır)
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


class CollectiveMemory:
    """
    Tüm LoRA'ların ortak hafızası
    """
    
    def __init__(self):
        """
        Ortak hafıza - MODEL İÇİNDE SAKLANIR!
        
        NOT: JSON dosyası yok, model (.pt) içinde tutuluyor!
        """
        self.memory = {}
        print(f"[Collective Memory] Ortak Hafiza baslatildi")
    
    def load_from_dict(self, memory_dict: Dict):
        """Model'den yüklenen hafızayı al"""
        self.memory = memory_dict
        print(f"   📚 {len(self.memory)} maç hafızadan yüklendi")
    
    def get_others_learning(self, requesting_lora_id: str, pattern_type: str = None, last_n_matches: int = 50) -> Dict:
        """
        Başkalarının öğrenmelerini al (LoRA bakış açısı!)
        
        Args:
            requesting_lora_id: Kim bakıyor?
            pattern_type: Belirli pattern mı? (örn: 'derbi', 'hype')
            last_n_matches: Son kaç maça bak?
        
        Returns:
            {
                'lora_001_Einstein': {
                    'learning_summary': "Derbi pattern'de HOME riskli öğrendim",
                    'adjustments': [...],
                    'success_rate': 0.75,
                    'confidence_trend': 'increasing'
                },
                ...
            }
        """
        others_learning = {}
        
        # Son N maçı tara
        recent_matches = list(self.memory.keys())[-last_n_matches:]
        
        for match_key in recent_matches:
            match_data = self.memory[match_key]
            
            if 'lora_insights' not in match_data:
                continue
            
            # Her LoRA'nın o maçtaki öğrenmesini kaydet
            for lora_id, insight in match_data['lora_insights'].items():
                if lora_id == requesting_lora_id:
                    continue  # Kendini atla!
                
                if lora_id not in others_learning:
                    others_learning[lora_id] = {
                        'name': insight['name'],
                        'learnings': [],
                        'adjustments': [],
                        'correct_count': 0,
                        'wrong_count': 0,
                        'temperament': insight.get('temperament_type', 'Unknown'),
                        'reputation': insight.get('reputation', {}),  # 🏆 İtibar!
                        'emotional_archetype': insight.get('emotional_archetype', 'Dengeli')  # 🎭 Arketip!
                    }
                
                # Öğrenmeyi ekle
                if insight.get('learning'):
                    others_learning[lora_id]['learnings'].append(insight['learning'])
                
                # Ayarlamaları ekle
                if insight.get('personal_adjustments'):
                    others_learning[lora_id]['adjustments'].extend(insight['personal_adjustments'])
                
                # Başarı say
                if insight['correct']:
                    others_learning[lora_id]['correct_count'] += 1
                else:
                    others_learning[lora_id]['wrong_count'] += 1
        
        # Başarı oranları hesapla
        for lora_id, data in others_learning.items():
            total = data['correct_count'] + data['wrong_count']
            if total > 0:
                data['success_rate'] = data['correct_count'] / total
            else:
                data['success_rate'] = 0.0
        
        return others_learning
    
    def record_match(self, 
                     match_idx: int,
                     home_team: str,
                     away_team: str,
                     match_date: str,
                     lora_predictions: List[Dict],
                     actual_result: str,
                     actual_score: Optional[tuple] = None):
        """
        Bir maçı ortak hafızaya kaydet
        
        Args:
            lora_predictions: [
                {
                    'lora_id': 'abc123',
                    'lora_name': 'LoRA_001',
                    'prediction': 'AWAY',
                    'confidence': 0.90,
                    'temperament_type': 'Independent',
                    'result': 'CORRECT'
                },
                ...
            ]
        """
        match_key = f"match_{match_idx}"
        
        # Consensus hesapla
        predictions = [p['prediction'] for p in lora_predictions]
        consensus = max(set(predictions), key=predictions.count)
        agreement_rate = predictions.count(consensus) / len(predictions)
        
        # Doğru/yanlış LoRA'ları ayır
        correct_loras = [p['lora_name'] for p in lora_predictions if p['result'] == 'CORRECT']
        wrong_loras = [p['lora_name'] for p in lora_predictions if p['result'] == 'WRONG']
        
        # ✨ YENİ: Her LoRA'nın detaylı kaydı (öğrenme + itibar!)
        lora_detailed_records = {}
        for pred in lora_predictions:
            lora_detailed_records[pred['lora_id']] = {
                'name': pred['lora_name'],
                'prediction': pred['prediction'],
                'confidence': pred['confidence'],
                'correct': pred['result'] == 'CORRECT',
                'temperament_type': pred.get('temperament_type', 'Unknown'),
                'emotional_archetype': pred.get('emotional_archetype', 'Dengeli'),  # 🎭 YENİ!
                'reasoning': pred.get('reasoning', ''),
                'learning': pred.get('learning', ''),
                'personal_adjustments': pred.get('adjustments', []),
                'reputation': pred.get('reputation', {}),
                'authority_weight': pred.get('authority_weight', 1.0),
                'tes_scores': pred.get('tes_scores', {}),  # 🔬 TES skorları!
                'life_energy': pred.get('life_energy', 1.0),  # ⚡ Yaşam enerjisi!
                'physics_archetype': pred.get('physics_archetype', 'Dengeli ⚖️')  # 🎭 Fizik arketip!
            }
        
        # En emin doğru/yanlış
        correct_preds = [p for p in lora_predictions if p['result'] == 'CORRECT']
        wrong_preds = [p for p in lora_predictions if p['result'] == 'WRONG']
        
        most_confident_correct = max(correct_preds, key=lambda x: x['confidence']) if correct_preds else None
        most_confident_wrong = max(wrong_preds, key=lambda x: x['confidence']) if wrong_preds else None
        
        # Hafızaya kaydet (TARİHSEL VERİ + HYPE!)
        self.memory[match_key] = {
            'match_info': {
                'home': home_team,
                'away': away_team,
                'date': match_date,
                'actual_result': actual_result,
                'actual_score': actual_score,
                # 🔥 HYPE VERİLERİ (Zamanla öğrenilecek!)
                'total_tweets': None,  # Runtime'da doldurulacak
                'sentiment_score': None,
                'home_support': None,
                'away_support': None
            },
            'lora_thoughts': lora_predictions,
            'lora_insights': lora_detailed_records,  # ✨ Detaylı: öğrenme + yorum!
            'consensus': {
                'majority': consensus,
                'agreement_rate': agreement_rate,
                'correct_loras': correct_loras,
                'wrong_loras': wrong_loras,
                'total_correct': len(correct_loras),
                'total_wrong': len(wrong_loras),
                'accuracy': len(correct_loras) / len(lora_predictions) if lora_predictions else 0
            },
            'insights': {
                'most_confident_correct': most_confident_correct,
                'most_confident_wrong': most_confident_wrong,
                'consensus_correct': (consensus.lower() in actual_result.lower())
            }
        }
        
        # NOT: Artık diske ayrı kaydetmiyoruz!
        # Model kaydedilirken otomatik kaydedilecek (.pt içinde)
    
    def get_lora_stats(self, lora_id: str, last_n: int = 50) -> Dict:
        """
        Bir LoRA'nın son N maçtaki performansını analiz et
        
        Returns:
            {
                'total_matches': 50,
                'accuracy': 0.75,
                'avg_confidence': 0.82,
                'overconfident': True/False,  # Emin ama yanlış mı?
                'reliability': 0.85  # Güvenilirlik skoru
            }
        """
        matches = list(self.memory.values())[-last_n:]
        
        total = 0
        correct = 0
        confidences = []
        overconfident_count = 0
        
        for match in matches:
            for thought in match['lora_thoughts']:
                if thought['lora_id'] == lora_id:
                    total += 1
                    if thought['result'] == 'CORRECT':
                        correct += 1
                    
                    confidences.append(thought['confidence'])
                    
                    # Overconfident: >%80 emin ama yanlış
                    if thought['confidence'] > 0.8 and thought['result'] == 'WRONG':
                        overconfident_count += 1
        
        if total == 0:
            return None
        
        accuracy = correct / total
        avg_confidence = sum(confidences) / len(confidences)
        
        # Güvenilirlik = accuracy - overconfident_penalty
        overconfident_ratio = overconfident_count / total
        reliability = accuracy - (overconfident_ratio * 0.3)
        
        return {
            'total_matches': total,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'overconfident_ratio': overconfident_ratio,
            'overconfident': (overconfident_ratio > 0.2),
            'reliability': max(0.0, reliability)
        }
    
    def get_consensus_for_pattern(self, pattern_type: str, last_n: int = 30) -> Dict:
        """
        Belirli bir pattern'de (hype, odds, xG) consensus ne kadar doğru?
        
        Returns:
            {
                'consensus_accuracy': 0.65,
                'recommendation': 'trust' / 'question' / 'ignore'
            }
        """
        # Şimdilik basit versiyon - ileride genişletilebilir
        matches = list(self.memory.values())[-last_n:]
        
        total = 0
        consensus_correct = 0
        
        for match in matches:
            if match['consensus']['majority']:
                total += 1
                if match['insights']['consensus_correct']:
                    consensus_correct += 1
        
        if total == 0:
            return {'consensus_accuracy': 0.5, 'recommendation': 'question'}
        
        accuracy = consensus_correct / total
        
        # Öneri
        if accuracy > 0.7:
            recommendation = 'trust'
        elif accuracy < 0.4:
            recommendation = 'ignore'
        else:
            recommendation = 'question'
        
        return {
            'consensus_accuracy': accuracy,
            'recommendation': recommendation
        }
    
    def get_team_recent_history(self, team_name: str, last_n: int = 5, 
                                current_match_idx: int = 999999) -> Dict:
        """
        🌊 AKIŞKAN TARİHSEL VERİ ÇEKİMİ!
        
        Bir takımın son N maçını ortak hafızadan çek.
        
        LoRA'lar bunu kullanarak:
        - Son 5 maçta kaç gol attı?
        - Son 5 maçta kaç gol yedi?
        - Formu ne? (3W 1D 1L → +3)
        - Hype trendi? (artan/azalan)
        
        Args:
            team_name: Takım ismi
            last_n: Son kaç maç? (default: 5)
            current_match_idx: Şimdiki maç (o maçı dahil etme!)
        
        Returns:
            {
                'scored': [2, 1, 3, 0, 2],  # Son 5 maçta attığı goller
                'conceded': [1, 0, 1, 2, 1],  # Son 5 maçta yediği goller
                'results': ['WIN', 'WIN', 'LOSS', 'DRAW', 'WIN'],
                'form': '+3',  # Win=+1, Draw=0, Loss=-1
                'avg_scored': 1.6,
                'avg_conceded': 1.0,
                'hype_trend': 'increasing',  # Hype artıyor mu?
                'avg_hype': 0.65,  # Ortalama home_support
                'matches_found': 5
            }
        """
        # Ortak hafızadaki tüm maçları tara
        team_matches = []
        
        for match_key, match_data in self.memory.items():
            # Maç index'i çıkar
            try:
                match_idx = int(match_key.split('_')[1])
            except:
                continue
            
            # Şimdiki maçtan önceki maçlar
            if match_idx >= current_match_idx:
                continue
            
            match_info = match_data['match_info']
            
            # Bu takım bu maçta oynadı mı?
            is_home = (match_info['home'] == team_name)
            is_away = (match_info['away'] == team_name)
            
            if is_home or is_away:
                team_matches.append({
                    'match_idx': match_idx,
                    'is_home': is_home,
                    'opponent': match_info['away'] if is_home else match_info['home'],
                    'result': match_info['actual_result'],
                    'score': match_info.get('actual_score'),
                    'total_tweets': match_info.get('total_tweets'),
                    'sentiment': match_info.get('sentiment_score'),
                    'home_support': match_info.get('home_support'),
                    'away_support': match_info.get('away_support'),
                    'date': match_info.get('date')
                })
        
        # Son N maçı al (tarihe göre sırala)
        team_matches.sort(key=lambda x: x['match_idx'], reverse=True)
        recent = team_matches[:last_n]
        recent.reverse()  # Eskiden yeniye (kronolojik)
        
        if len(recent) == 0:
            return {
                'scored': [],
                'conceded': [],
                'results': [],
                'form': 0,
                'avg_scored': 0.0,
                'avg_conceded': 0.0,
                'hype_trend': 'unknown',
                'avg_hype': 0.5,
                'matches_found': 0
            }
        
        # GOL VERİLERİNİ ÇIKAR
        scored = []
        conceded = []
        results = []
        hype_values = []
        
        for m in recent:
            if m['score'] is not None:
                home_g, away_g = m['score']
                
                if m['is_home']:
                    scored.append(home_g)
                    conceded.append(away_g)
                else:
                    scored.append(away_g)
                    conceded.append(home_g)
                
                # Sonuç (bu takım için!)
                if m['is_home']:
                    if 'home' in m['result'].lower():
                        results.append('WIN')
                    elif 'away' in m['result'].lower():
                        results.append('LOSS')
                    else:
                        results.append('DRAW')
                else:
                    if 'away' in m['result'].lower():
                        results.append('WIN')
                    elif 'home' in m['result'].lower():
                        results.append('LOSS')
                    else:
                        results.append('DRAW')
            
            # HYPE VERİSİ
            if m['is_home'] and m['home_support'] is not None:
                hype_values.append(m['home_support'])
            elif not m['is_home'] and m['away_support'] is not None:
                hype_values.append(m['away_support'])
        
        # FORM HESAPLA (Win=+1, Draw=0, Loss=-1)
        form = sum([1 if r == 'WIN' else (-1 if r == 'LOSS' else 0) for r in results])
        
        # HYPE TREND (artıyor mu azalıyor mu?)
        if len(hype_values) >= 3:
            first_half_hype = np.mean(hype_values[:len(hype_values)//2])
            second_half_hype = np.mean(hype_values[len(hype_values)//2:])
            
            if second_half_hype > first_half_hype + 0.1:
                hype_trend = 'increasing'
            elif second_half_hype < first_half_hype - 0.1:
                hype_trend = 'decreasing'
            else:
                hype_trend = 'stable'
        else:
            hype_trend = 'unknown'
        
        return {
            'scored': scored,
            'conceded': conceded,
            'results': results,
            'form': form,  # +5 = çok iyi, -3 = kötü
            'avg_scored': float(np.mean(scored)) if scored else 0.0,
            'avg_conceded': float(np.mean(conceded)) if conceded else 0.0,
            'hype_trend': hype_trend,
            'avg_hype': float(np.mean(hype_values)) if hype_values else 0.5,
            'matches_found': len(recent)
        }
    
    def get_h2h_history(self, team1: str, team2: str, last_n: int = 5,
                       current_match_idx: int = 999999) -> Dict:
        """
        🆚 İKİ TAKIMIN KARŞILAŞMA GEÇMİŞİ!
        
        LoRA'lar H2H (Head to Head) verilerini buradan öğrensin!
        
        Returns:
            {
                'team1_wins': 3,
                'team2_wins': 1,
                'draws': 1,
                'team1_avg_goals': 1.8,
                'team2_avg_goals': 1.2,
                'last_5_scores': [(2,1), (0,1), (3,3), (1,0), (2,1)],
                'matches_found': 5
            }
        """
        h2h_matches = []
        
        for match_key, match_data in self.memory.items():
            # Maç index
            try:
                match_idx = int(match_key.split('_')[1])
            except:
                continue
            
            if match_idx >= current_match_idx:
                continue
            
            match_info = match_data['match_info']
            
            # Bu iki takım oynadı mı?
            is_match = (
                (match_info['home'] == team1 and match_info['away'] == team2) or
                (match_info['home'] == team2 and match_info['away'] == team1)
            )
            
            if is_match:
                team1_is_home = (match_info['home'] == team1)
                
                h2h_matches.append({
                    'match_idx': match_idx,
                    'team1_is_home': team1_is_home,
                    'score': match_info.get('actual_score'),
                    'result': match_info['actual_result'],
                    'date': match_info.get('date')
                })
        
        # Son N maçı al
        h2h_matches.sort(key=lambda x: x['match_idx'], reverse=True)
        recent_h2h = h2h_matches[:last_n]
        recent_h2h.reverse()  # Kronolojik
        
        if len(recent_h2h) == 0:
            return {
                'team1_wins': 0,
                'team2_wins': 0,
                'draws': 0,
                'team1_avg_goals': 0.0,
                'team2_avg_goals': 0.0,
                'last_5_scores': [],
                'matches_found': 0
            }
        
        # İSTATİSTİKLER
        team1_wins = 0
        team2_wins = 0
        draws = 0
        team1_goals = []
        team2_goals = []
        scores = []
        
        for m in recent_h2h:
            if m['score'] is not None:
                home_g, away_g = m['score']
                
                if m['team1_is_home']:
                    team1_goals.append(home_g)
                    team2_goals.append(away_g)
                    scores.append((home_g, away_g))
                else:
                    team1_goals.append(away_g)
                    team2_goals.append(home_g)
                    scores.append((away_g, home_g))
                
                # Kazananı belirle
                if m['team1_is_home']:
                    if 'home' in m['result'].lower():
                        team1_wins += 1
                    elif 'away' in m['result'].lower():
                        team2_wins += 1
                    else:
                        draws += 1
                else:
                    if 'away' in m['result'].lower():
                        team1_wins += 1
                    elif 'home' in m['result'].lower():
                        team2_wins += 1
                    else:
                        draws += 1
        
        return {
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'draws': draws,
            'team1_avg_goals': float(np.mean(team1_goals)) if team1_goals else 0.0,
            'team2_avg_goals': float(np.mean(team2_goals)) if team2_goals else 0.0,
            'last_5_scores': scores,
            'matches_found': len(recent_h2h)
        }
    
    def update_match_hype_data(self, match_idx: int, total_tweets: float,
                              sentiment_score: float, home_support: float,
                              away_support: float):
        """
        🔥 HYPE VERİLERİNİ GÜNCELLE!
        
        Maç kaydedildikten sonra hype verilerini ekle.
        LoRA'lar bir sonraki maçta bunu görecek!
        """
        match_key = f"match_{match_idx}"
        
        if match_key in self.memory:
            self.memory[match_key]['match_info']['total_tweets'] = total_tweets
            self.memory[match_key]['match_info']['sentiment_score'] = sentiment_score
            self.memory[match_key]['match_info']['home_support'] = home_support
            self.memory[match_key]['match_info']['away_support'] = away_support
    
    def interpret_based_on_temperament(self, lora, collective_data: Dict) -> Dict:
        """
        LoRA kişiliğine göre ortak hafızayı yorumla
        
        Args:
            lora: LoRA objesi (temperament'ı var)
            collective_data: Ortak hafıza verisi
        
        Returns:
            {
                'strategy': 'follow_consensus' / 'trust_self' / 'follow_experts' / 'oppose',
                'confidence_modifier': 0.8 - 1.3 arası çarpan
            }
        """
        temp = lora.temperament
        
        # 1) BAĞIMSIZ (independence > 0.7)
        if temp['independence'] > 0.7:
            return {
                'strategy': 'trust_self',
                'confidence_modifier': 1.1,  # Kendi düşüncesine daha çok güven
                'reason': 'Bağımsız kişilik: Kendi analizime güveniyorum'
            }
        
        # 2) SOSYAL ZEKİ (social_intelligence > 0.7)
        elif temp['social_intelligence'] > 0.7:
            return {
                'strategy': 'follow_experts',
                'confidence_modifier': 1.2,  # Güvenilir kaynaklara uy
                'reason': 'Sosyal zeki: Güvenilir LoRA\'ları takip ediyorum'  # ✅ Escape!
            }
        
        # 3) SÜRÜ PSİKOLOJİSİ (herd_tendency > 0.6)
        elif temp['herd_tendency'] > 0.6:
            return {
                'strategy': 'follow_consensus',
                'confidence_modifier': 0.9,  # Çoğunluğa uy
                'reason': 'Sürü psikolojisi: Çoğunluk ne diyorsa o!'
            }
        
        # 4) KARŞIT GÖRÜŞ (contrarian_score > 0.6)
        elif temp['contrarian_score'] > 0.6:
            return {
                'strategy': 'oppose',
                'confidence_modifier': 1.15,  # Karşıt görüşe güven
                'reason': 'Karşıt görüş: Çoğunluğun tersi doğrudur!'
            }
        
        # 5) DENGELİ
        else:
            return {
                'strategy': 'balanced',
                'confidence_modifier': 1.0,
                'reason': 'Dengeli yaklaşım'
            }


# Global instance
collective_memory = CollectiveMemory()

