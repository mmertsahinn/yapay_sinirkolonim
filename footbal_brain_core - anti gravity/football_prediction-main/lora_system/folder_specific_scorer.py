📂 FOLDER SPECIFIC SCORER (Klasöre Özel Puanlama)
=================================================

⚠️ ÖNEMLİ: Bu modül "MASTER_CONTEXT_RULES.md" kurallarına sıkı sıkıya bağlıdır.
Her klasörün (uzmanlık alanının) kendi "doğrusu" vardır!

- Manchester Klasörü: Manchester maçlarındaki başarıya bakar.
- Einstein Klasörü: Lazarus Lambda (potansiyel) ve Fisher bilgisine bakar.
- Hybrid Klasörü: Hem başarı hem istikrara bakar.
- Genel Klasör: Standart Advanced Score kullanır.

Bu modül, hedef klasöre göre doğru puanlama formülünü seçer.
"""

from .advanced_score_calculator import AdvancedScoreCalculator

class FolderSpecificScorer:
    """
    Hedef klasöre göre özelleşmiş puanlama sistemi
    """
    
    @staticmethod
    def calculate_score_for_folder(lora, folder_name: str, match_count: int = None, collective_memory=None) -> float:
        """
        Belirtilen klasör tipi için skor hesapla
        
        Args:
            lora: LoRA objesi
            folder_name: Hedef klasör adı (örn: "Manchester", "EINSTEIN", "HYBRID")
            match_count: Mevcut maç sayısı
            collective_memory: Ortak hafıza (Maç verileri için şart!)
            
        Returns:
            0-1 arası skor
        """
        folder_upper = folder_name.upper()
        
        # 1. EINSTEIN / MUCIZE KLASÖRÜ (Potansiyel Odaklı)
        if "EINSTEIN" in folder_upper or "MUCIZE" in folder_upper:
            return FolderSpecificScorer._calculate_einstein_score(lora)
            
        # 2. HYBRID KLASÖRÜ (Denge Odaklı)
        elif "HYBRID" in folder_upper:
            return FolderSpecificScorer._calculate_hybrid_score(lora, match_count)
            
        # 3. TAKIM KLASÖRLERİ (Örn: Manchester, Inter, Real Madrid)
        # Takım ismi klasör adında geçiyorsa, o takımdaki başarısına bak!
        elif FolderSpecificScorer._is_team_folder(folder_name):
            return FolderSpecificScorer._calculate_team_score(lora, folder_name, match_count, collective_memory)
            
        # 4. DEFAULT (Standart Advanced Score)
        else:
            return AdvancedScoreCalculator.calculate_advanced_score(lora, match_count)

    @staticmethod
    def _calculate_einstein_score(lora) -> float:
        """
        Einstein Formülü:
        - %60 Lazarus Lambda (Potansiyel)
        - %20 Fisher Information (Öğrenme Kapasitesi)
        - %20 Mevcut Başarı
        """
        lazarus = getattr(lora, '_lazarus_lambda', 0.5)
        fitness = lora.get_recent_fitness()
        
        # Einstein Skoru
        score = (lazarus * 0.80) + (fitness * 0.20)
        return min(1.0, score)

    @staticmethod
    def _calculate_hybrid_score(lora, match_count) -> float:
        """
        Hybrid Formülü:
        - %40 Advanced Score
        - %40 Consistency (İstikrar)
        - %20 Temperament Balance (Dengeli Mizaç)
        """
        adv_score = AdvancedScoreCalculator.calculate_advanced_score(lora, match_count)
        consistency = AdvancedScoreCalculator._calculate_consistency(lora)
        
        # Mizaç dengesi (Social + Independence dengesi)
        temp = getattr(lora, 'temperament', {})
        indep = temp.get('independence', 0.5)
        social = temp.get('social_intelligence', 0.5)
        balance = 1.0 - abs(indep - social)  # Birbirine ne kadar yakın?
        
        score = (adv_score * 0.40) + (consistency * 0.40) + (balance * 0.20)
        return min(1.0, score)

    @staticmethod
    def _calculate_team_score(lora, team_name: str, match_count, collective_memory) -> float:
        """
        Takım Özel Formülü (BAĞLAM TABANLI):
        - Sadece o takımın maçlarındaki performansa bakar.
        - Global skor önemsizdir.
        """
        if not collective_memory:
            # Fallback: Memory yoksa eski usul (Genel + Bonus)
            base_score = AdvancedScoreCalculator.calculate_advanced_score(lora, match_count)
            specialization = getattr(lora, 'specialization', '')
            bonus = 0.30 if specialization and team_name.lower() in specialization.lower() else -0.20
            return max(0.0, min(1.0, base_score + bonus))

        # 1. İlgili maçları bul
        relevant_matches = []
        # Takım adını temizle (örn: "Team_Real_Madrid" -> "Real Madrid")
        clean_team_name = team_name.replace("Team_", "").replace("_", " ")
        
        for match_key, match_data in collective_memory.memory.items():
            info = match_data['match_info']
            # Bu takım ev sahibi veya deplasman mı?
            if info['home'] == clean_team_name or info['away'] == clean_team_name:
                relevant_matches.append(match_data)
        
        if not relevant_matches:
            return 0.0 # Hiç maç yoksa skor 0
            
        # 2. LoRA'nın bu maçlardaki performansını ölç
        correct_count = 0
        total_confidence = 0.0
        participated_matches = 0
        match_history = [] # True/False history for trend analysis
        
        for m in relevant_matches:
            pred = next((p for p in m['lora_thoughts'] if p['lora_id'] == lora.id), None)
            if pred:
                participated_matches += 1
                is_correct = (pred['result'] == 'CORRECT')
                if is_correct:
                    correct_count += 1
                total_confidence += pred['confidence']
                match_history.append(is_correct)
        
        if participated_matches < 3:
            return 0.0 # Yetersiz veri (En az 3 maç)
            
        # 🧠 DEEP MATH EVALUATION (Bayesian Wilson Score)
        from lora_system.deep_evaluator import DeepEvaluator
        
        base_score = DeepEvaluator.calculate_bayesian_score(
            correct=correct_count, 
            total=participated_matches, 
            total_confidence=total_confidence
        )
        
        # Trend Bonusu (Öğrenme İvmesi)
        trend_bonus = DeepEvaluator.calculate_trend_bonus(match_history)
        
        final_score = base_score + trend_bonus
        
        return max(0.0, min(1.0, final_score))

    @staticmethod
    def _is_team_folder(folder_name: str) -> bool:
        """
        Klasör adı bir takım mı?
        (Basit kontrol: EINSTEIN, HYBRID, MUCIZE değilse ve büyük harf değilse muhtemelen takımdır)
        """
        reserved = ["EINSTEIN", "HYBRID", "MUCIZE", "GENEL", "AKTIF", "ARSIV"]
        upper_name = folder_name.upper()
        
        for r in reserved:
            if r in upper_name:
                return False
                
        return True

    @staticmethod
    def calculate_h2h_score(lora, team1: str, team2: str, collective_memory) -> float:
        """
        H2H (İki Takım Arası) Skor Hesapla
        
        Kriterler:
        1. Bu iki takımın maçlarındaki başarısı (Accuracy)
        2. Güven seviyesi (Confidence)
        3. Hype/Deplasman/Ev sahibi performansı (Bonus)
        """
        # Ortak hafızadan bu iki takımın maçlarını bul
        # (collective_memory.get_h2h_history sadece maçları verir, LoRA tahminlerini vermez)
        # Bu yüzden manuel tarama yapacağız.
        
        relevant_matches = []
        for match_key, match_data in collective_memory.memory.items():
            info = match_data['match_info']
            if (info['home'] == team1 and info['away'] == team2) or \
               (info['home'] == team2 and info['away'] == team1):
                relevant_matches.append(match_data)
                
        if not relevant_matches:
            return 0.0
            
        correct_count = 0
        total_confidence = 0.0
        total_matches = 0
        
        for m in relevant_matches:
            # LoRA'nın tahminini bul
            prediction = next((p for p in m['lora_thoughts'] if p['lora_id'] == lora.id), None)
            if prediction:
                total_matches += 1
                if prediction['result'] == 'CORRECT':
                    correct_count += 1
                total_confidence += prediction['confidence']
                
        if total_matches < 3:
            return 0.0  # Yetersiz veri (En az 3 maç!)
            
        accuracy = correct_count / total_matches
        avg_confidence = total_confidence / total_matches
        
        # Skor: Accuracy (%70) + Confidence (%30)
        score = (accuracy * 0.7) + (avg_confidence * 0.3)
        
        return score

    @staticmethod
    def get_h2h_details(lora, team1: str, team2: str, collective_memory) -> dict:
        """
        H2H Detaylı Analiz (Txt dosyası için)
        """
        relevant_matches = []
        for match_key, match_data in collective_memory.memory.items():
            info = match_data['match_info']
            if (info['home'] == team1 and info['away'] == team2) or \
               (info['home'] == team2 and info['away'] == team1):
                relevant_matches.append(match_data)
        
        stats = {
            'total_matches': 0,
            'correct': 0,
            'home_correct': 0, # Team1 evindeyken
            'away_correct': 0, # Team1 deplasmandayken
            'high_hype_correct': 0, # Hype yüksekken
            'avg_confidence': 0.0
        }
        
        total_conf = 0.0
        
        for m in relevant_matches:
            pred = next((p for p in m['lora_thoughts'] if p['lora_id'] == lora.id), None)
            if pred:
                stats['total_matches'] += 1
                total_conf += pred['confidence']
                
                is_correct = (pred['result'] == 'CORRECT')
                if is_correct:
                    stats['correct'] += 1
                
                # Home/Away analizi (Team1 referans)
                is_team1_home = (m['match_info']['home'] == team1)
                
                if is_team1_home:
                    if is_correct: stats['home_correct'] += 1
                else:
                    if is_correct: stats['away_correct'] += 1
                    
                # Hype analizi (>0.7 hype)
                hype = m['match_info'].get('home_support', 0.5) if is_team1_home else m['match_info'].get('away_support', 0.5)
                if hype and hype > 0.7 and is_correct:
                    stats['high_hype_correct'] += 1

        if stats['total_matches'] > 0:
            stats['avg_confidence'] = total_conf / stats['total_matches']
            stats['accuracy'] = stats['correct'] / stats['total_matches']
        else:
            stats['accuracy'] = 0.0
            
        return stats

# Global instance
folder_specific_scorer = FolderSpecificScorer()
