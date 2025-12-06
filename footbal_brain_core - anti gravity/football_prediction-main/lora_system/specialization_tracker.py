"""
🎯 UZMANLIK TAKİP SİSTEMİ
=========================

Her LoRA'nın hangi pattern'lerde iyi olduğunu takip eder.
Uzmanlıklar değişebilir (nadir ama mümkün - mutasyon gibi)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SpecializationHistory:
    """Uzmanlık geçmişi"""
    specialization: str
    start_match: int
    end_match: Optional[int] = None
    success_rate: float = 0.0


class SpecializationTracker:
    """
    Uzmanlık takip sistemi
    
    Pattern türleri:
    - derby_match: Derbi maçlar
    - high_hype: Yüksek hype
    - odds_surprise: Odds sürprizi
    - underdog: Küçük takım galibiyeti
    - high_scoring: Gollü maçlar
    - low_scoring: Az gollü
    - season_end: Sezon sonu
    - league_specific: Lig bazlı
    """
    
    PATTERN_TYPES = [
        'derby_match',
        'high_hype',
        'odds_surprise',
        'underdog',
        'favorite',
        'high_scoring',
        'low_scoring',
        'season_end',
        'season_start',
        'midweek',
        'weekend',
        'home_advantage',
        'away_advantage'
    ]
    
    SPECIALIZATION_THRESHOLDS = {
        'min_matches': 15,        # En az 15 maç
        'min_success_rate': 0.70, # En az %70 başarı
        'evolution_chance': 0.05  # %5 şans (mutasyon gibi nadir)
    }
    
    def __init__(self):
        self.match_count = 0
    
    def detect_match_patterns(self, match_data) -> List[str]:
        """
        Maçın hangi pattern'lere ait olduğunu tespit et
        
        Returns:
            Liste: ['derby_match', 'high_hype', ...]
        """
        patterns = []
        
        # Hype yüksek mi?
        hype = match_data.get('total_tweets', 0) + match_data.get('hype_score', 0)
        if hype > 50000:  # Threshold
            patterns.append('high_hype')
        
        # Derbi mi? (basit kontrol: takım isimleri benzer)
        home = str(match_data.get('home_team', '')).lower()
        away = str(match_data.get('away_team', '')).lower()
        
        # Şehir isimleri paylaşıyor mu?
        cities = ['london', 'manchester', 'madrid', 'milan', 'liverpool', 'istanbul', 'rome']
        for city in cities:
            if city in home and city in away:
                patterns.append('derby_match')
                break
        
        # Underdog mu? (odds varsa)
        home_odds = match_data.get('home_win', 0)
        away_odds = match_data.get('away_win', 0)
        
        if home_odds > 3.0 or away_odds > 3.0:
            patterns.append('underdog')
        
        if home_odds < 1.5 or away_odds < 1.5:
            patterns.append('favorite')
        
        # Gol farkı (varsa)
        if 'home_xG' in match_data and 'away_xG' in match_data:
            total_xg = match_data.get('home_xG', 0) + match_data.get('away_xG', 0)
            if total_xg > 3.5:
                patterns.append('high_scoring')
            elif total_xg < 2.0:
                patterns.append('low_scoring')
        
        # Varsayılan
        if len(patterns) == 0:
            patterns.append('general')
        
        return patterns
    
    def update_lora_patterns(self, lora, match_patterns: List[str], was_correct: bool):
        """
        LoRA'nın pattern performansını güncelle
        """
        if not hasattr(lora, 'pattern_performance'):
            lora.pattern_performance = {pattern: {'correct': 0, 'total': 0} for pattern in self.PATTERN_TYPES}
            lora.pattern_performance['general'] = {'correct': 0, 'total': 0}
        
        for pattern in match_patterns:
            if pattern not in lora.pattern_performance:
                lora.pattern_performance[pattern] = {'correct': 0, 'total': 0}
            
            lora.pattern_performance[pattern]['total'] += 1
            if was_correct:
                lora.pattern_performance[pattern]['correct'] += 1
    
    def detect_specialization(self, lora, current_match: int) -> Optional[str]:
        """
        LoRA'nın uzmanlığını tespit et
        
        Returns:
            Uzmanlık adı veya None
        """
        if not hasattr(lora, 'pattern_performance'):
            return None
        
        best_pattern = None
        best_rate = 0.0
        
        for pattern, stats in lora.pattern_performance.items():
            if stats['total'] < self.SPECIALIZATION_THRESHOLDS['min_matches']:
                continue
            
            success_rate = stats['correct'] / stats['total']
            
            if success_rate > best_rate and success_rate >= self.SPECIALIZATION_THRESHOLDS['min_success_rate']:
                best_rate = success_rate
                best_pattern = pattern
        
        if best_pattern:
            return self._get_specialization_name(best_pattern, best_rate)
        
        return None
    
    def check_specialization_evolution(self, lora, current_match: int) -> Optional[Dict]:
        """
        Uzmanlık değişti mi kontrol et
        
        Returns:
            Eğer değiştiyse event dict, yoksa None
        """
        old_spec = getattr(lora, 'specialization', None)
        new_spec = self.detect_specialization(lora, current_match)
        
        # Uzmanlık değişti mi?
        if old_spec != new_spec and new_spec is not None:
            # Uzmanlık geçmişine ekle
            if not hasattr(lora, 'specialization_history'):
                lora.specialization_history = []
            
            # Eski uzmanlığı kapat
            if old_spec and len(lora.specialization_history) > 0:
                lora.specialization_history[-1].end_match = current_match
            
            # Yeni uzmanlık
            lora.specialization_history.append(
                SpecializationHistory(
                    specialization=new_spec,
                    start_match=current_match
                )
            )
            
            lora.specialization = new_spec
            
            return {
                'type': 'specialization_evolution',
                'lora_name': lora.name,
                'old_specialization': old_spec,
                'new_specialization': new_spec,
                'match': current_match,
                'evolved': True
            }
        
        # İlk uzmanlık tespiti
        elif old_spec is None and new_spec is not None:
            lora.specialization = new_spec
            
            if not hasattr(lora, 'specialization_history'):
                lora.specialization_history = []
            
            lora.specialization_history.append(
                SpecializationHistory(
                    specialization=new_spec,
                    start_match=current_match
                )
            )
            
            return {
                'type': 'specialization_discovered',
                'lora_name': lora.name,
                'specialization': new_spec,
                'match': current_match
            }
        
        return None
    
    def _get_specialization_name(self, pattern: str, success_rate: float) -> str:
        """Pattern'den uzmanlık adı"""
        names = {
            'derby_match': '⚔️ Derbi Uzmanı',
            'high_hype': '📢 Hype Uzmanı',
            'odds_surprise': '🎲 Odds Sürpriz Avcısı',
            'underdog': '🎯 Underdog Avcısı',
            'favorite': '👑 Favori Uzmanı',
            'high_scoring': '⚽ Gollü Maç Uzmanı',
            'low_scoring': '🛡️ Az Gollü Uzmanı',
            'season_end': '🏁 Sezon Sonu Uzmanı',
            'season_start': '🌱 Sezon Başı Uzmanı',
            'general': '🌍 Genel Uzman'
        }
        
        name = names.get(pattern, f'🔹 {pattern.title()} Uzmanı')
        
        # Başarı oranı çok yüksekse "Süper" ekle
        if success_rate > 0.85:
            name = "⭐ SÜPER " + name
        
        return name
    
    def get_lora_specialization_summary(self, lora) -> str:
        """LoRA'nın uzmanlık özetini getir"""
        
        if not hasattr(lora, 'pattern_performance'):
            return "Yeni LoRA (henüz veri yok)"
        
        summary = []
        
        for pattern, stats in lora.pattern_performance.items():
            if stats['total'] > 5:  # En az 5 maç
                rate = stats['correct'] / stats['total']
                summary.append(f"{pattern}: {rate*100:.0f}% ({stats['correct']}/{stats['total']})")
        
        return " | ".join(summary[:3]) if summary else "Henüz yeterli veri yok"
    
    def get_evolved_loras(self, population) -> List[Dict]:
        """Evrim geçiren LoRA'ları bul"""
        evolved = []
        
        for lora in population:
            if hasattr(lora, 'specialization_history') and len(lora.specialization_history) > 1:
                evolved.append({
                    'lora': lora,
                    'history': lora.specialization_history,
                    'evolution_count': len(lora.specialization_history) - 1
                })
        
        return evolved




