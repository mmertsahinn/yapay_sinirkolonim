"""
🎖️ UZMANLIK SİSTEMİ
===================

Her LoRA'nın pattern başarısını takip eder.
Otomatik uzmanlık atar ve evrimini izler.

Uzmanlıklar:
- Derbi Uzmanı
- Hype Uzmanı
- Odds Uzmanı
- Underdog Avcısı
- Favori Avcısı
- Gollü Maç Uzmanı
- Az Gollü Uzmanı
- Sezon Sonu Uzmanı
- Kaos Uzmanı
- vs.

Uzmanlık değişebilir (nadir ama olabilir):
- Travmadan sonra
- Mutasyondan sonra
- Doğal evrim ile
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PatternStats:
    """Pattern istatistikleri"""
    correct: int = 0
    total: int = 0
    
    @property
    def rate(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class SpecializationHistory:
    """Uzmanlık geçmişi"""
    specialization: str
    start_match: int
    end_match: Optional[int] = None
    success_rate: float = 0.0


class SpecializationSystem:
    """
    Uzmanlık tespit ve takip sistemi
    """
    
    # Uzmanlık kriterleri
    SPECIALIZATIONS = {
        'derby': {
            'name': 'Derbi Uzmanı',
            'min_rate': 0.75,
            'min_count': 10,
            'emoji': '⚔️'
        },
        'high_hype': {
            'name': 'Hype Uzmanı',
            'min_rate': 0.70,
            'min_count': 15,
            'emoji': '📢'
        },
        'odds_surprise': {
            'name': 'Odds Uzmanı',
            'min_rate': 0.70,
            'min_count': 12,
            'emoji': '🎲'
        },
        'underdog': {
            'name': 'Underdog Avcısı',
            'min_rate': 0.72,
            'min_count': 15,
            'emoji': '🦊'
        },
        'favorite': {
            'name': 'Favori Avcısı',
            'min_rate': 0.70,
            'min_count': 15,
            'emoji': '👑'
        },
        'high_scoring': {
            'name': 'Gollü Maç Uzmanı',
            'min_rate': 0.68,
            'min_count': 12,
            'emoji': '⚽'
        },
        'low_scoring': {
            'name': 'Az Gollü Uzmanı',
            'min_rate': 0.68,
            'min_count': 12,
            'emoji': '🛡️'
        },
        'season_end': {
            'name': 'Sezon Sonu Uzmanı',
            'min_rate': 0.75,
            'min_count': 8,
            'emoji': '🏁'
        },
        'chaos': {
            'name': 'Kaos Uzmanı',
            'min_rate': 0.65,
            'min_count': 10,
            'emoji': '🌪️'
        },
        'general': {
            'name': 'Genel Uzman',
            'min_rate': 0.68,
            'min_count': 50,
            'emoji': '⭐'
        }
    }
    
    @staticmethod
    def initialize_lora_specialization(lora):
        """LoRA'ya uzmanlık tracking başlat"""
        if not hasattr(lora, 'pattern_stats'):
            lora.pattern_stats = {
                pattern: PatternStats() 
                for pattern in SpecializationSystem.SPECIALIZATIONS.keys()
            }
        
        if not hasattr(lora, 'specialization'):
            lora.specialization = None
        
        if not hasattr(lora, 'specialization_history'):
            lora.specialization_history = []
        
        if not hasattr(lora, 'is_evolved'):
            lora.is_evolved = False  # Uzmanlık değişti mi?
    
    @staticmethod
    def update_pattern_stats(lora, match_features: Dict, correct: bool):
        """
        Maç sonrası pattern istatistiklerini güncelle
        
        match_features: {
            'is_derby': bool,
            'is_high_hype': bool,
            'is_odds_surprise': bool,
            'is_underdog': bool,
            'is_high_scoring': bool,
            ...
        }
        """
        if not hasattr(lora, 'pattern_stats'):
            SpecializationSystem.initialize_lora_specialization(lora)
        
        # Her pattern için güncelle
        for pattern, is_active in match_features.items():
            if is_active and pattern in lora.pattern_stats:
                lora.pattern_stats[pattern].total += 1
                if correct:
                    lora.pattern_stats[pattern].correct += 1
    
    @staticmethod
    def detect_specialization(lora, match_count: int) -> Optional[str]:
        """
        LoRA'nın uzmanlığını otomatik tespit et
        
        Returns:
            Yeni uzmanlık (varsa)
        """
        if not hasattr(lora, 'pattern_stats'):
            return None
        
        old_specialization = lora.specialization
        best_specialization = None
        best_score = 0.0
        
        # Her pattern için kontrol
        for pattern_key, criteria in SpecializationSystem.SPECIALIZATIONS.items():
            stats = lora.pattern_stats.get(pattern_key)
            
            if stats and stats.total >= criteria['min_count']:
                rate = stats.rate
                
                if rate >= criteria['min_rate']:
                    # Skor = başarı oranı × log(maç sayısı)
                    score = rate * np.log1p(stats.total)
                    
                    if score > best_score:
                        best_score = score
                        best_specialization = pattern_key
        
        # Yeni uzmanlık
        if best_specialization:
            new_spec_name = SpecializationSystem.SPECIALIZATIONS[best_specialization]['name']
            
            # Değişti mi?
            if old_specialization != new_spec_name:
                # UZMANLIK DEĞİŞTİ!
                
                # Geçmişe ekle
                if old_specialization:
                    # Eski uzmanlığı kapat
                    if len(lora.specialization_history) > 0:
                        lora.specialization_history[-1].end_match = match_count
                    
                    lora.is_evolved = True  # EVRİM ETİKETİ!
                
                # Yeni uzmanlık başlat
                lora.specialization_history.append(
                    SpecializationHistory(
                        specialization=new_spec_name,
                        start_match=match_count,
                        success_rate=SpecializationSystem._get_pattern_rate(lora, best_specialization)
                    )
                )
                
                lora.specialization = new_spec_name
                
                return new_spec_name  # Değişti!
        
        return None  # Değişmedi
    
    @staticmethod
    def _get_pattern_rate(lora, pattern_key: str) -> float:
        """Pattern başarı oranı"""
        if pattern_key in lora.pattern_stats:
            return lora.pattern_stats[pattern_key].rate
        return 0.0
    
    @staticmethod
    def get_specialization_display(lora) -> str:
        """Uzmanlık gösterimi (emoji ile)"""
        if not hasattr(lora, 'specialization') or not lora.specialization:
            return ""
        
        # Emoji bul
        for pattern_key, criteria in SpecializationSystem.SPECIALIZATIONS.items():
            if criteria['name'] == lora.specialization:
                emoji = criteria['emoji']
                
                # Evrim etiketi
                evolved = " ⚡" if getattr(lora, 'is_evolved', False) else ""
                
                return f"{emoji} {lora.specialization}{evolved}"
        
        return lora.specialization
    
    @staticmethod
    def get_specialization_evolution_log(lora) -> str:
        """Uzmanlık evrim geçmişi"""
        if not hasattr(lora, 'specialization_history') or len(lora.specialization_history) == 0:
            return "Henüz uzmanlık yok"
        
        log = []
        for i, history in enumerate(lora.specialization_history):
            end_text = f"Maç #{history.end_match}" if history.end_match else "Devam ediyor"
            log.append(
                f"  {i+1}. {history.specialization} "
                f"(Maç #{history.start_match} - {end_text}) "
                f"[Başarı: {history.success_rate*100:.1f}%]"
            )
        
        return "\n".join(log)
    
    @staticmethod
    def classify_match(match_data) -> Dict[str, bool]:
        """
        Maçı sınıflandır (hangi pattern'lere ait?)
        
        Returns:
            {'derby': True, 'high_hype': False, ...}
        """
        features = {}
        
        # Derby (takım isimleri benzerse)
        home = match_data.get('home_team', '').lower()
        away = match_data.get('away_team', '').lower()
        features['derby'] = any(word in home and word in away 
                               for word in ['united', 'city', 'fc', 'real', 'milan'])
        
        # High Hype (total_tweets yüksek)
        hype = match_data.get('total_tweets', 0)
        features['high_hype'] = hype > 20000
        
        # Odds surprise
        if 'home_odds' in match_data and match_data.get('home_odds', 0) > 0:
            home_implied = 1.0 / match_data['home_odds']
            away_implied = 1.0 / match_data.get('away_odds', 1.5)
            features['odds_surprise'] = abs(home_implied - away_implied) > 0.3
        else:
            features['odds_surprise'] = False
        
        # Underdog (odds'a göre)
        features['underdog'] = match_data.get('home_odds', 2.0) > 3.0
        features['favorite'] = match_data.get('home_odds', 2.0) < 1.5
        
        # Gollü maç
        total_goals = match_data.get('home_goals', 0) + match_data.get('away_goals', 0)
        features['high_scoring'] = total_goals >= 4
        features['low_scoring'] = total_goals <= 1
        
        # Sezon sonu (ay kontrolü)
        try:
            import pandas as pd
            date = pd.to_datetime(match_data.get('date'))
            features['season_end'] = date.month in [4, 5]  # Nisan-Mayıs
        except:
            features['season_end'] = False
        
        # Kaos (chaos_index yüksekse)
        features['chaos'] = match_data.get('chaos_index', 0) > 0.6
        
        # Genel (her zaman True)
        features['general'] = True
        
        return features

