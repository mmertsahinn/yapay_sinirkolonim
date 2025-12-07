"""
🧠 REPLAY BUFFER - Hafıza Sistemi
==================================

Önemli maçları saklar:
- Modelin yanıldığı maçlar (yüksek loss)
- Aşırı sürpriz skorlar (7-0, vs.)
- Yüksek hype + beklenmedik sonuç
- Lig/sezon dengesi
"""

import numpy as np
from typing import List, Dict, Optional
import random


class ReplayBuffer:
    """
    Deneyim hafızası (Experience Replay)
    Önemli maçları saklar ve online öğrenme için kullanır
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.storage: List[Dict] = []
        
        # İstatistikler
        self.total_added = 0
        self.total_pruned = 0
    
    def add(self, example: Dict):
        """
        Yeni deneyim ekle
        
        example = {
            'features': np.array (58,),
            'base_proba': np.array (3,),
            'lora_proba': np.array (3,),
            'actual_class_idx': int,
            'actual_result': str,
            'loss': float,
            'surprise': float,  # 1 - p(actual)
            'hype': float,      # total_tweets veya hype_score
            'goal_diff': int,
            'match_date': str,
            'home_team': str,
            'away_team': str,
            'league': str,
            'predicted_class': str,
            'correct': bool
        }
        """
        # Önem skoru hesapla
        importance = self._calculate_importance(example)
        example['importance'] = importance
        
        self.storage.append(example)
        self.total_added += 1
        
        # Limit aşıldıysa, en az önemliyi at
        if len(self.storage) > self.max_size:
            self._prune()
    
    def _calculate_importance(self, example: Dict) -> float:
        """
        Maçın önemi (ne kadar hatırlamaya değer?)
        
        Yüksek önem kriterleri:
        - Yüksek loss (model çok yanıldı)
        - Yüksek surprise (beklenmedik sonuç)
        - Aşırı skor farkı (7-0, vs.)
        - Yüksek hype maçlar
        """
        importance = 0.0
        
        # 1) Loss (0-1 arası normalize)
        loss = example.get('loss', 0.5)
        importance += min(loss, 2.0) * 0.3  # Max 0.6 katkı
        
        # 2) Surprise (0-1 arası)
        surprise = example.get('surprise', 0.0)
        importance += surprise * 0.3  # Max 0.3 katkı
        
        # 3) Gol farkı (3+ fark = çok önemli)
        goal_diff = abs(example.get('goal_diff', 0))
        if goal_diff >= 5:
            importance += 0.3
        elif goal_diff >= 3:
            importance += 0.2
        elif goal_diff >= 2:
            importance += 0.1
        
        # 4) Hype (normalize edilmiş)
        hype = example.get('hype', 0.0)
        # Varsayalım hype 0-100k arası
        normalized_hype = min(hype / 50000, 1.0)
        importance += normalized_hype * 0.2  # Max 0.2 katkı
        
        return importance
    
    def _prune(self):
        """En az önemli örnekleri çıkar"""
        # Importance'a göre sırala
        self.storage.sort(key=lambda x: x.get('importance', 0.0), reverse=True)
        
        # En önemli max_size kadarını tut
        removed = len(self.storage) - self.max_size
        self.storage = self.storage[:self.max_size]
        self.total_pruned += removed
    
    def sample(self, batch_size: int = 16) -> List[Dict]:
        """
        Rastgele örnek çek (online learning için)
        Önem skoruna göre ağırlıklı örnekleme
        """
        if len(self.storage) == 0:
            return []
        
        # Önem skorlarını ağırlık olarak kullan
        importances = np.array([ex.get('importance', 0.5) for ex in self.storage])
        importances = np.clip(importances, 0.01, 10.0)  # Güvenlik
        probs = importances / importances.sum()
        
        # Ağırlıklı örnekleme
        sample_size = min(batch_size, len(self.storage))
        indices = np.random.choice(
            len(self.storage),
            size=sample_size,
            replace=False,
            p=probs
        )
        
        return [self.storage[i] for i in indices]
    
    def sample_uniform(self, batch_size: int = 16) -> List[Dict]:
        """Uniform (eşit olasılıklı) örnekleme"""
        if len(self.storage) == 0:
            return []
        
        sample_size = min(batch_size, len(self.storage))
        return random.sample(self.storage, sample_size)
    
    def sample_situational(self, criteria: Dict, batch_size: int = 16) -> List[Dict]:
        """
        🌊 DURUMSAL ÖRNEKLEME (Situational Sampling)
        
        Mevcut duruma uygun geçmiş maçları getir!
        Örn: Şu anki maç 'High Hype' ise, geçmişteki 'High Hype' maçları getir.
        
        Args:
            criteria: Filtreleme kriterleri (örn: {'high_hype': True})
            batch_size: Kaç örnek?
        """
        if len(self.storage) == 0:
            return []
        
        # 1. Kriterlere uyanları bul
        candidates = []
        for ex in self.storage:
            match = True
            for key, val in criteria.items():
                # Hype kontrolü (özel mantık)
                if key == 'high_hype':
                    # Hype > 0.7 ise high hype say
                    ex_hype = ex.get('hype', 0)
                    is_high = (ex_hype > 50000) if ex_hype > 100 else (ex_hype > 0.7)
                    if is_high != val:
                        match = False
                        break
                # Gol farkı kontrolü
                elif key == 'high_goal_diff':
                    diff = abs(ex.get('goal_diff', 0))
                    is_high = (diff >= 3)
                    if is_high != val:
                        match = False
                        break
                # Normal eşleşme
                elif key in ex:
                    if ex[key] != val:
                        match = False
                        break
            
            if match:
                candidates.append(ex)
        
        # 2. Yeterli aday var mı?
        if len(candidates) < batch_size // 2:
            # Yeterli yoksa, karışık getir (yarı situational, yarı random)
            needed = batch_size - len(candidates)
            # Use dictionary identity or manual check instead of direct 'not in' with numpy arrays inside dictionaries
            # because (dict_a == dict_b) can fail if values are arrays.
            # Just use IDs or object identity if possible, but here 'ex' are dicts.
            # Robust way: compare Python object IDs
            candidate_ids = {id(c) for c in candidates}
            others = [ex for ex in self.storage if id(ex) not in candidate_ids]
            
            if others:
                # Kalanı önem sırasına göre doldur
                other_probs = np.array([ex.get('importance', 0.5) for ex in others])
                other_probs = other_probs / other_probs.sum()
                
                chosen_others_idx = np.random.choice(
                    len(others), 
                    size=min(needed, len(others)), 
                    replace=False, 
                    p=other_probs
                )
                chosen_others = [others[i] for i in chosen_others_idx]
                candidates.extend(chosen_others)
            
            return candidates
        
        # 3. Adaylardan önem ağırlıklı seç
        importances = np.array([ex.get('importance', 0.5) for ex in candidates])
        probs = importances / importances.sum()
        
        indices = np.random.choice(
            len(candidates),
            size=min(batch_size, len(candidates)),
            replace=False,
            p=probs
        )
        
        return [candidates[i] for i in indices]
    
    def get_top_k(self, k: int = 10) -> List[Dict]:
        """En önemli K örneği döndür"""
        sorted_storage = sorted(self.storage, key=lambda x: x.get('importance', 0.0), reverse=True)
        return sorted_storage[:k]
    
    def get_stats(self) -> Dict:
        """Buffer istatistikleri"""
        if len(self.storage) == 0:
            return {
                'size': 0,
                'total_added': self.total_added,
                'total_pruned': self.total_pruned
            }
        
        importances = [ex.get('importance', 0.0) for ex in self.storage]
        losses = [ex.get('loss', 0.0) for ex in self.storage]
        surprises = [ex.get('surprise', 0.0) for ex in self.storage]
        
        return {
            'size': len(self.storage),
            'max_size': self.max_size,
            'total_added': self.total_added,
            'total_pruned': self.total_pruned,
            'avg_importance': np.mean(importances),
            'max_importance': np.max(importances),
            'avg_loss': np.mean(losses),
            'avg_surprise': np.mean(surprises),
            'high_importance_count': sum(1 for x in importances if x > 0.7)
        }
    
    def filter_by_criteria(self, **criteria) -> List[Dict]:
        """
        Kriterlere göre filtrele
        
        Örnek:
        buffer.filter_by_criteria(goal_diff=5, correct=False)
        → 5 gol farkla yanlış tahmin edilen maçlar
        """
        results = []
        
        for ex in self.storage:
            match = True
            for key, value in criteria.items():
                if key not in ex:
                    match = False
                    break
                
                if isinstance(value, (int, float)):
                    if abs(ex[key] - value) > 0.01:
                        match = False
                        break
                else:
                    if ex[key] != value:
                        match = False
                        break
            
            if match:
                results.append(ex)
        
        return results
    
    def add_user_selected_matches(self, matches: List[Dict]):
        """
        Kullanıcının seçtiği özel maçları ekle
        (Ani değişiklikler, özel durumlar vs.)
        """
        for match in matches:
            # Kullanıcı seçimiyse önem otomatik yüksek
            if 'importance' not in match:
                match['importance'] = 1.0
            
            self.storage.append(match)
            self.total_added += 1
        
        print(f"✅ {len(matches)} kullanıcı seçimli maç buffer'a eklendi")
        
        # Limit kontrolü
        if len(self.storage) > self.max_size:
            self._prune()
    
    def save(self, filepath: str):
        """Buffer'ı diske kaydet"""
        import joblib
        joblib.dump({
            'storage': self.storage,
            'max_size': self.max_size,
            'total_added': self.total_added,
            'total_pruned': self.total_pruned
        }, filepath)
        print(f"💾 Buffer kaydedildi: {filepath}")
    
    def load(self, filepath: str):
        """Buffer'ı diskten yükle"""
        import joblib
        try:
            data = joblib.load(filepath)
            self.storage = data['storage']
            self.max_size = data['max_size']
            self.total_added = data.get('total_added', len(self.storage))
            self.total_pruned = data.get('total_pruned', 0)
            print(f"📂 Buffer yüklendi: {filepath} ({len(self.storage)} örnek)")
        except FileNotFoundError:
            print(f"⚠️ Buffer dosyası bulunamadı: {filepath}")
    
    def clear(self):
        """Buffer'ı temizle"""
        self.storage.clear()
        print("🗑️ Buffer temizlendi")
    
    def __len__(self):
        return len(self.storage)
    
    def __repr__(self):
        stats = self.get_stats()
        return f"ReplayBuffer(size={stats['size']}/{self.max_size}, avg_importance={stats.get('avg_importance', 0):.3f})"




