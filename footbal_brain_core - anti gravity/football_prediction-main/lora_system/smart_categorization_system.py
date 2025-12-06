"""
🧠 AKILLI KATEGORİZASYON SİSTEMİ
=================================

LoRA'ları otomatik olarak en uygun kategorilere yerleştirir:

KATEGORİ HIYERARŞISI:
1. GENEL UZMAN → Tüm maçlarda iyi (geniş beceri)
2. TAKIM SPESIFIK → Belirli takımlarda iyi (dar beceri)
3. LIGA SPESIFIK → Belirli ligalarda iyi
4. HYPE SPESIFIK → Hype seviyesine göre iyi
5. VS SPESIFIK → Belirli eşleşmelerde iyi (en dar!)

MANTIK:
- Eğer LoRA HER YERDE iyi → GENEL
- Eğer SADECE Manchester'da iyi → MANCHESTER UZMAN
- Eğer SADECE Man vs Liv'de iyi → VS UZMAN
- Eğer hem genel hem özel iyi → HYBRID!

SPESİFİKLİK SKORU:
- 1.0 = Tam genel (tüm maçlarda iyi)
- 0.5 = Karma (bazı yerlerde iyi)
- 0.0 = Tam spesifik (sadece bir yerde iyi)
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class SmartCategorizationSystem:
    """
    LoRA'ları akıllıca kategorilere yerleştirir
    """
    
    def __init__(self):
        self.categorization_history = []
        print("🧠 Smart Categorization System başlatıldı")
    
    def analyze_lora_specialization(self,
                                    lora,
                                    team_performance: Dict[str, List[bool]],
                                    global_performance: List[bool],
                                    hype_performance: Dict[str, List[bool]],
                                    match_count: int) -> Dict:
        """
        LoRA'nın uzmanlık alanını akıllıca analiz et
        
        Args:
            lora: LoRA instance
            team_performance: {team_name: [True, False, True, ...]}  # Her takımda doğru/yanlış
            global_performance: [True, False, True, ...]  # Tüm maçlarda
            hype_performance: {'high': [...], 'medium': [...], 'low': [...]}
            match_count: Toplam maç sayısı
        
        Returns:
            {
                'primary_category': 'GENERAL' | 'TEAM_SPECIFIC' | 'VS_SPECIFIC' | 'HYPE_SPECIFIC',
                'team_specializations': [(team_name, score), ...],  # En iyi 5
                'vs_specializations': [((team1, team2), score), ...],  # En iyi 5
                'hype_specialization': ('high' | 'medium' | 'low', score),
                'specificity_score': 0.0-1.0,  # 1.0 = tam genel, 0.0 = tam spesifik
                'placement_reason': str
            }
        """
        
        # 1) GENEL PERFORMANS
        if len(global_performance) > 0:
            global_accuracy = sum(global_performance) / len(global_performance)
        else:
            global_accuracy = 0.0
        
        # 2) TAKIM BAZLI PERFORMANS
        team_accuracies = {}
        for team_name, results in team_performance.items():
            if len(results) >= 5:  # Minimum 5 maç
                team_accuracies[team_name] = sum(results) / len(results)
        
        # 3) HYPE BAZLI PERFORMANS
        hype_accuracies = {}
        for hype_level, results in hype_performance.items():
            if len(results) >= 5:
                hype_accuracies[hype_level] = sum(results) / len(results)
        
        # 4) UZMANLIK TESPİTİ (Akıllı analiz!)
        specialization_analysis = self._detect_specialization_pattern(
            global_accuracy,
            team_accuracies,
            hype_accuracies
        )
        
        # 5) KATEGORI BELİRLE
        category_result = self._determine_primary_category(
            lora,
            global_accuracy,
            team_accuracies,
            hype_accuracies,
            specialization_analysis
        )
        
        return category_result
    
    def _detect_specialization_pattern(self,
                                       global_acc: float,
                                       team_accs: Dict[str, float],
                                       hype_accs: Dict[str, float]) -> Dict:
        """
        Uzmanlık paternini tespit et (İstatistiksel analiz!)
        """
        
        # 1) TAKIM VARYANS ANALİZİ
        # Eğer bazı takımlarda çok iyi, bazılarında kötüyse → Spesifik!
        # Eğer hepsinde benzer → Genel!
        
        if len(team_accs) > 0:
            team_values = list(team_accs.values())
            team_variance = np.var(team_values)
            team_mean = np.mean(team_values)
            
            # Varyasyon katsayısı (CV = std / mean)
            if team_mean > 0:
                team_cv = np.sqrt(team_variance) / team_mean
            else:
                team_cv = 0.0
        else:
            team_variance = 0.0
            team_cv = 0.0
        
        # 2) HYPE VARYANS ANALİZİ
        if len(hype_accs) > 0:
            hype_values = list(hype_accs.values())
            hype_variance = np.var(hype_values)
        else:
            hype_variance = 0.0
        
        # 3) GLOBAL vs SPECIFIC FARK
        # Eğer bazı takımlarda global'den çok daha iyiyse → Spesifik!
        team_outperformance = {}
        for team_name, team_acc in team_accs.items():
            outperformance = team_acc - global_acc
            if outperformance > 0.10:  # +10% fark
                team_outperformance[team_name] = outperformance
        
        return {
            'team_variance': team_variance,
            'team_cv': team_cv,  # Yüksek CV = Spesifik
            'hype_variance': hype_variance,
            'team_outperformance': team_outperformance,
            'is_specialist': team_cv > 0.15  # CV > 15% → Uzman!
        }
    
    def _determine_primary_category(self,
                                    lora,
                                    global_acc: float,
                                    team_accs: Dict[str, float],
                                    hype_accs: Dict[str, float],
                                    pattern: Dict) -> Dict:
        """
        Ana kategoriyi belirle (Akıllı karar ağacı!)
        """
        
        # 1) GENEL MI UZMAN MI?
        if global_acc >= 0.70 and not pattern['is_specialist']:
            # Genel olarak iyi VE spesifik bir yeri yok
            primary_category = 'GENERAL'
            placement_reason = f"Genel başarı yüksek (%{global_acc*100:.0f}) ve tüm takımlarda dengeli (CV: {pattern['team_cv']:.2f})"
            specificity = 1.0  # Tam genel
        
        elif len(pattern['team_outperformance']) > 0:
            # Bazı takımlarda ÇOK daha iyi → TAKIM UZMAN!
            primary_category = 'TEAM_SPECIFIC'
            best_teams = sorted(pattern['team_outperformance'].items(), 
                               key=lambda x: x[1], reverse=True)[:3]
            teams_str = ', '.join([t[0] for t in best_teams])
            placement_reason = f"Bu takımlarda özel başarı: {teams_str}"
            specificity = 0.3  # Orta spesifik
        
        elif pattern['hype_variance'] > 0.05:
            # Hype seviyelerine göre farklı → HYPE UZMAN!
            primary_category = 'HYPE_SPECIFIC'
            best_hype = max(hype_accs.items(), key=lambda x: x[1]) if hype_accs else ('medium', 0.5)
            placement_reason = f"Hype bazlı uzmanlık ({best_hype[0]}: %{best_hype[1]*100:.0f})"
            specificity = 0.5  # Orta
        
        else:
            # Çok spesifik veya henüz net değil
            primary_category = 'DEVELOPING'
            placement_reason = f"Henüz net uzmanlık belirlenmedi (Genel: %{global_acc*100:.0f})"
            specificity = 0.5
        
        # En iyi takımları sırala
        team_specializations = sorted(team_accs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Hype uzmanlığı
        hype_specialization = None
        if hype_accs:
            best_hype = max(hype_accs.items(), key=lambda x: x[1])
            hype_specialization = best_hype
        
        return {
            'primary_category': primary_category,
            'team_specializations': team_specializations,
            'vs_specializations': [],  # VS analizini ayrı yapacağız
            'hype_specialization': hype_specialization,
            'specificity_score': specificity,
            'placement_reason': placement_reason,
            'global_accuracy': global_acc,
            'pattern_analysis': pattern
        }
    
    def get_placement_recommendations(self, categorization: Dict) -> List[str]:
        """
        LoRA'nın hangi klasörlere yerleştirilmesi gerektiğini öner
        
        Returns:
            ['en_iyi_loralar/🌍_GENEL_UZMANLAR/🎯_WIN_EXPERTS/', 
             'en_iyi_loralar/takım_uzmanlıkları/Manchester_United/🎯_WIN_EXPERTS/', ...]
        """
        placements = []
        
        primary = categorization['primary_category']
        
        # 1) GENEL UZMAN
        if primary == 'GENERAL':
            # Genel klasörlere yerleştir
            placements.append('en_iyi_loralar/🌍_GENEL_UZMANLAR/🎯_WIN_EXPERTS/')
            placements.append('en_iyi_loralar/🌍_GENEL_UZMANLAR/⚽_GOAL_EXPERTS/')
        
        # 2) TAKIM UZMAN
        elif primary == 'TEAM_SPECIFIC':
            # En iyi 3 takıma yerleştir
            top_teams = categorization['team_specializations'][:3]
            for team_name, score in top_teams:
                if score >= 0.70:  # Minimum %70 başarı
                    safe_name = team_name.replace(' ', '_')
                    placements.append(f'en_iyi_loralar/takım_uzmanlıkları/{safe_name}/🎯_WIN_EXPERTS/')
        
        # 3) HYPE UZMAN
        elif primary == 'HYPE_SPECIFIC':
            hype_spec = categorization['hype_specialization']
            if hype_spec:
                hype_level, score = hype_spec
                if score >= 0.70:
                    placements.append(f'en_iyi_loralar/🌍_GENEL_UZMANLAR/🔥_HYPE_EXPERTS/')
        
        # 4) HYBRID (Hem genel hem spesifik iyi!)
        if categorization['global_accuracy'] >= 0.65 and len(categorization['team_specializations']) > 0:
            # Hem genel hem spesifik başarılı → HYBRID!
            placements.append('en_iyi_loralar/🌈_HYBRID_SPECIALISTS/')
        
        return placements
    
    def generate_placement_report(self, lora, categorization: Dict) -> str:
        """
        LoRA için yerleştirme raporu oluştur
        """
        report = []
        report.append(f"\n{'='*100}")
        report.append(f"🧠 {lora.name} - AKILLI KATEGORİZASYON RAPORU")
        report.append(f"{'='*100}")
        
        # Ana kategori
        report.append(f"\n📍 ANA KATEGORİ: {categorization['primary_category']}")
        report.append(f"   Sebep: {categorization['placement_reason']}")
        report.append(f"   Spesifiklik: {categorization['specificity_score']:.2f} (1.0=Genel, 0.0=Çok Spesifik)")
        
        # Performans detayları
        report.append(f"\n📊 PERFORMANS ANALİZİ:")
        report.append(f"   Genel Başarı: %{categorization['global_accuracy']*100:.1f}")
        
        # En iyi takımlar
        if categorization['team_specializations']:
            report.append(f"\n🏆 EN İYİ TAKIMLAR (Top 5):")
            for i, (team, score) in enumerate(categorization['team_specializations'][:5], 1):
                report.append(f"   {i}. {team:30s}: %{score*100:.1f}")
        
        # Hype uzmanlığı
        if categorization['hype_specialization']:
            hype_level, score = categorization['hype_specialization']
            report.append(f"\n🔥 HYPE UZMANLIĞI:")
            report.append(f"   Seviye: {hype_level.upper()}")
            report.append(f"   Başarı: %{score*100:.1f}")
        
        # Önerilen yerleştirmeler
        placements = self.get_placement_recommendations(categorization)
        if placements:
            report.append(f"\n📁 ÖNERİLEN YERLEŞTİRMELER ({len(placements)}):")
            for i, path in enumerate(placements, 1):
                report.append(f"   {i}. {path}")
        
        report.append(f"\n{'='*100}\n")
        
        return '\n'.join(report)


# Global instance
smart_categorization = SmartCategorizationSystem()

