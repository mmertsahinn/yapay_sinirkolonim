"""
🧠 LoRA ADAPTER - Low-Rank Adaptation
======================================

LoRA (Low-Rank Adaptation) implementasyonu:
- Ana ağırlıklar donuk (frozen)
- Sadece küçük A, B matrislerini eğitiyoruz
- Rank=16, Alpha=16 (güçlü konfigürasyon)

Input: 61 boyut (58 feature + 3 base_proba)
Output: 3 boyut (home_win, draw, away_win probabilities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import uuid


class LoRALinear(nn.Module):
    """
    LoRA ile donatılmış Linear katman.
    W = W_frozen + (B @ A) * (alpha / rank)
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0, device='cpu'):
        super().__init__()
        
        self.device = device
        
        # Ana ağırlık (DONUK - train edilmiyor) - DOĞRU DEVICE'DA YARAT!
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        nn.init.xavier_uniform_(self.weight)  # Daha stabil
        self.weight.requires_grad = False
        
        # LoRA matrisleri (BUNLAR train ediliyor) - DOĞRU DEVICE'DA YARAT!
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)  # Küçük değerler
        nn.init.zeros_(self.lora_B)
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Ana ağırlıkla hesapla
        base_output = F.linear(x, self.weight)
        
        # LoRA delta'sını ekle
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        
        return base_output + lora_output * self.scaling
    
    def get_lora_params(self):
        """Sadece LoRA parametrelerini döndür"""
        return {'lora_A': self.lora_A.data.clone(), 'lora_B': self.lora_B.data.clone()}
    
    def set_lora_params(self, params: Dict):
        """LoRA parametrelerini ayarla (device-aware)"""
        # Parametreleri doğru device'a taşıyarak kopyala
        target_device = self.lora_A.device if hasattr(self, 'device') else 'cpu'
        self.lora_A.data = params['lora_A'].clone().to(target_device)
        self.lora_B.data = params['lora_B'].clone().to(target_device)


class LoRAAdapter(nn.Module):
    """
    Tam LoRA Adapter Ağı:
    Input (78) → LoRA Linear (128) → ReLU → LoRA Linear (64) → ReLU → LoRA Linear (3) → Softmax
    
    🔧 78 BOYUT = 60 features + 3 base_proba + 15 tarihsel (gol + hype + h2h)
    
    TARİHSEL VERİLER (15 boyut):
    - Home son 5 maç: avg_scored, avg_conceded, form, avg_hype, hype_trend (5)
    - Away son 5 maç: avg_scored, avg_conceded, form, avg_hype, hype_trend (5)
    - H2H geçmiş: team1_avg_goals, team2_avg_goals, win_rate, draw_rate (4)
    - Data quality: 1 (ne kadar veri var?)
    """
    
    def __init__(self, input_dim: int = 78, hidden_dim: int = 128, rank: int = 16, alpha: float = 16.0, device='cpu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.alpha = alpha
        self.device = device  # 🔧 DEVICE ATTRIBUTE'UNU SET ET!
        
        # 3 katmanlı ağ (DOĞRU DEVICE'DA YARAT!)
        self.fc1 = LoRALinear(input_dim, hidden_dim, rank=rank, alpha=alpha, device=device)
        self.fc2 = LoRALinear(hidden_dim, 64, rank=rank, alpha=alpha, device=device)
        self.fc3 = LoRALinear(64, 3, rank=rank, alpha=alpha, device=device)
        
        # Dropout (regularization)
        self.dropout = nn.Dropout(0.1)
        
        # 🔧 TÜM KATMANLARI DEVICE'A TAŞI!
        self.to(device)
        
        # Metadata
        self.id = str(uuid.uuid4())[:8]
        self.name = f"LoRA_{self.id}"
        self.generation = 0
        self.parents = []
        self.birth_match = 0
        
        # Performans metrikleri
        self.match_history = []
        self.fitness_history = []
        self.specialization = None  # "hype_expert", "odds_expert", vs.
        
        # Sosyal ve psikolojik özellikler (Koloni mantığı için)
        self.pattern_attractions = {}  # Pattern çekimleri
        self.social_bonds = {}  # Diğer LoRA'larla bağlar
        self.main_goal = None  # Ana hedef
        self.trauma_history = []  # Travma geçmişi
        
        # 🎭 KİŞİLİK ÖZELLİKLERİ (Genetik olarak geçer!)
        self.temperament = self._initialize_random_temperament()
        
        # 🎭 DUYGU ARKETİPİ (Kimlik etiketi!)
        self.emotional_archetype = self._determine_emotional_archetype()
        
        # 🍀 HAYATTA KALMA (Etiketler için)
        self.lucky_survivals = 0  # Kaç kez şanslı kurtuldu
        self.resurrection_count = 0  # Kaç kez dirildi
        self.children_count = 0  # Kaç çocuk doğurdu
        
        # 🧠 KİŞİSEL HAFIZA (Her LoRA'nın kendi öğrenme tarihi!)
        self.personal_memory = {
            'learned_patterns': {},  # Pattern bazlı öğrenmeler
            'learning_history': [],  # Ne zaman ne öğrendi
            'observed_others': {},   # Başkalarından ne öğrendi
            'adjustments': []        # Kendi değişimleri
        }
    
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, 63) tensor (60 features + 3 base_proba)
        returns: (batch_size, 3) probabilities
        """
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        
        logits = self.fc3(h2)
        proba = F.softmax(logits, dim=-1)
        
        return proba
    
    def predict(self, features_np: np.ndarray, base_proba_np: np.ndarray, device='cpu'):
        """
        Numpy array'lerden tahmin yap
        
        Args:
            features_np: 75 boyut (60 base + 15 tarihsel)
            base_proba_np: 3 boyut (ensemble tahmini)
        
        TOPLAM INPUT: 78 boyut
        """
        # Debug: Boyutları yazdır
        # print(f"   predict() - features: {features_np.shape}, base_proba: {base_proba_np.shape}")
        
        # Input oluştur: [features (75) + base_proba (3)] = 78 boyut
        x = np.concatenate([features_np, base_proba_np]).astype(np.float32)
        # print(f"   predict() - combined: {x.shape}")
        
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
        # print(f"   predict() - tensor: {x_tensor.shape}")
        
        with torch.no_grad():
            proba = self.forward(x_tensor).cpu().numpy()[0]
        
        return proba
    
    def predict_score(self, home_xg: float, away_xg: float, 
                     features_np: np.ndarray = None) -> tuple:
        """
        xG verilerinden + LoRA düzeltmelerinden skor tahmini
        
        Args:
            home_xg: Ev sahibi xG
            away_xg: Deplasman xG
            features_np: Maç özellikleri (opsiyonel, düzeltme için)
            
        Returns:
            (home_goals, away_goals): Tahmin edilen skor
        """
        from .score_predictor import score_predictor
        
        # Base xG'den skor tahmini
        base_home, base_away = score_predictor.predict_score_from_xg(home_xg, away_xg)
        
        # LoRA düzeltmesi (opsiyonel)
        # Eğer LoRA güçlü bir avantaj görüyorsa skoru ayarla
        if features_np is not None and hasattr(self, 'get_confidence'):
            confidence = self.get_confidence()
            
            # Yüksek confidence varsa LoRA'nın tahminini kullan
            if confidence > 0.7:
                # Örnek: HOME ağırlığı yüksekse ev sahibine +1 gol
                pass  # Şimdilik base skorları kullan
        
        return base_home, base_away
    
    def get_all_lora_params(self):
        """Tüm LoRA parametrelerini al (çiftleşme/klonlama için)"""
        return {
            'fc1': self.fc1.get_lora_params(),
            'fc2': self.fc2.get_lora_params(),
            'fc3': self.fc3.get_lora_params()
        }
    
    def set_all_lora_params(self, params: Dict):
        """Tüm LoRA parametrelerini ayarla"""
        self.fc1.set_lora_params(params['fc1'])
        self.fc2.set_lora_params(params['fc2'])
        self.fc3.set_lora_params(params['fc3'])
    
    def _initialize_random_temperament(self) -> Dict:
        """
        Rastgele kişilik özellikleri oluştur
        
        🎭 15 KİŞİLİK ÖZELLİĞİ (Psikolojik Derinlik!):
        
        TEMEL:
        1. independence: Bağımsızlık
        2. social_intelligence: Sosyal zeka
        3. herd_tendency: Sürü eğilimi
        4. contrarian_score: Karşıt görüş
        
        DUYGUSAL:
        5. emotional_depth: Duygusal derinlik (kayıplara tepki)
        6. empathy: Empati (başkalarının acısını hissetme)
        7. anger_tendency: Sinirlilik eğilimi
        
        PERFORMANS:
        8. ambition: Hırs (başarı tutkusu)
        9. competitiveness: Rekabetçilik
        10. resilience: Dayanıklılık (travmalardan toparlanma)
        11. will_to_live: Yaşam isteği (ölüme direnç)
        
        DAVRANIŞSAL:
        12. patience: Sabır
        13. impulsiveness: Dürtüsellik
        14. stress_tolerance: Stres toleransı
        15. risk_appetite: Risk iştahı
        """
        import random
        
        return {
            # TEMEL (4)
            'independence': random.uniform(0.3, 0.9),
            'social_intelligence': random.uniform(0.3, 0.9),
            'herd_tendency': random.uniform(0.1, 0.8),
            'contrarian_score': random.uniform(0.0, 0.7),
            
            # DUYGUSAL (3)
            'emotional_depth': random.uniform(0.2, 0.9),     # ✨ YENİ!
            'empathy': random.uniform(0.2, 0.9),             # ✨ YENİ!
            'anger_tendency': random.uniform(0.1, 0.9),      # ✨ YENİ!
            
            # PERFORMANS (4)
            'ambition': random.uniform(0.3, 0.95),           # ✨ YENİ!
            'competitiveness': random.uniform(0.2, 0.9),     # ✨ YENİ!
            'resilience': random.uniform(0.3, 0.9),          # ✨ YENİ!
            'will_to_live': random.uniform(0.4, 0.95),       # ✨ YENİ!
            
            # DAVRANIŞSAL (4)
            'patience': random.uniform(0.3, 0.9),
            'impulsiveness': random.uniform(0.1, 0.8),
            'stress_tolerance': random.uniform(0.4, 0.9),
            'risk_appetite': random.uniform(0.2, 0.9)
        }
    
    def _determine_emotional_archetype(self) -> str:
        """
        Mizaçtan duygu arketipini belirle (Kimlik etiketi!)
        
        ARKETIPLER:
        - Hırslı Savaşçı (Ambition + Competitiveness yüksek)
        - Duygusal Empatik (Emotional depth + Empathy yüksek)
        - Sakin Bilge (Stress tolerance + Resilience yüksek)
        - Sinirli Ateşli (Anger + Impulsiveness yüksek)
        - Sosyal Lider (Social intelligence + Empathy yüksek)
        - Bağımsız Yalnız (Independence + Contrarian yüksek)
        - Dengeli
        """
        temp = self.temperament
        
        # SKORLAMA
        ambitious_warrior = (temp['ambition'] + temp['competitiveness']) / 2
        emotional_empath = (temp['emotional_depth'] + temp['empathy']) / 2
        calm_sage = (temp['stress_tolerance'] + temp['resilience']) / 2
        angry_hothead = (temp['anger_tendency'] + temp['impulsiveness']) / 2
        social_leader = (temp['social_intelligence'] + temp['empathy']) / 2
        independent_lone = (temp['independence'] + temp['contrarian_score']) / 2
        
        # EN YÜKSEK SKOR
        archetypes = {
            'Hırslı Savaşçı': ambitious_warrior,
            'Duygusal Empatik': emotional_empath,
            'Sakin Bilge': calm_sage,
            'Sinirli Ateşli': angry_hothead,
            'Sosyal Lider': social_leader,
            'Bağımsız Yalnız': independent_lone
        }
        
        max_archetype = max(archetypes, key=archetypes.get)
        max_score = archetypes[max_archetype]
        
        # Eşik kontrolü (min 0.60)
        if max_score >= 0.60:
            return max_archetype
        else:
            return "Dengeli"
    
    def crossover(self, partner: 'LoRAAdapter') -> 'LoRAAdapter':
        """
        🧬 NÖRAL ÇAPRAZLAMA (Deep Neural Crossover)
        
        İki ebeveynin (self ve partner) genlerini ve BEYİNLERİNİ birleştirir.
        
        1. Parametre Karışımı: LoRA ağırlıkları karışır.
        2. İçgüdü Transferi (Neural Mating): ReactionNet ağırlıkları karışır.
        3. Hafıza Füzyonu (Ancestral Wisdom): En önemli anılar aktarılır.
        4. Tabula Rasa: Mizaç nötr başlar (0.5).
        """
        import random
        import copy
        
        # 1. Yeni Çocuk Oluştur
        child = LoRAAdapter(self.input_dim, self.hidden_dim, self.rank, self.alpha, device=self.device)
        child.generation = max(self.generation, partner.generation) + 1
        child.parents = [self.id, partner.id]
        
        # 2. Parametre Karışımı (LoRA Ağırlıkları)
        # Basit ortalama + Gürültü (Veya genetik parça değişimi)
        child_params = {}
        my_params = self.get_all_lora_params()
        partner_params = partner.get_all_lora_params()
        
        for key in my_params: # fc1, fc2, fc3
            child_params[key] = {}
            for subkey in my_params[key]: # lora_A, lora_B
                # %50 şansla anneden, %50 babadan (Genetik Dominantlık)
                if random.random() < 0.5:
                    child_params[key][subkey] = my_params[key][subkey].clone()
                else:
                    child_params[key][subkey] = partner_params[key][subkey].clone()
                    
                # Hafif mutasyon (%10)
                if random.random() < 0.1:
                    noise = torch.randn_like(child_params[key][subkey]) * 0.02
                    child_params[key][subkey] += noise
                    
        child.set_all_lora_params(child_params)
        
        # 3. İÇGÜDÜ TRANSFERİ (ReactionNet Mating)
        # Ebeveynlerin ReactionNet'leri varsa karıştır
        if hasattr(self, 'reaction_net') and hasattr(partner, 'reaction_net'):
            child.reaction_net = ReactionNet(input_dim=10, output_dim=5).to(self.device)
            
            # Ağırlıkları karıştır (Weighted Average)
            dominance = random.random() # 0.0 (Baba) - 1.0 (Anne)
            
            with torch.no_grad():
                # FC1
                child.reaction_net.fc1.weight.data = (
                    self.reaction_net.fc1.weight.data * dominance + 
                    partner.reaction_net.fc1.weight.data * (1 - dominance)
                )
                child.reaction_net.fc1.bias.data = (
                    self.reaction_net.fc1.bias.data * dominance + 
                    partner.reaction_net.fc1.bias.data * (1 - dominance)
                )
                
                # FC2
                child.reaction_net.fc2.weight.data = (
                    self.reaction_net.fc2.weight.data * dominance + 
                    partner.reaction_net.fc2.weight.data * (1 - dominance)
                )
                child.reaction_net.fc2.bias.data = (
                    self.reaction_net.fc2.bias.data * dominance + 
                    partner.reaction_net.fc2.bias.data * (1 - dominance)
                )
        
        # 4. HAFIZA FÜZYONU (Ancestral Wisdom)
        # Ebeveynlerin en önemli anılarını al (Core Memory)
        child.personal_memory_buffer = PersonalMemory()
        
        # Ebeveyn hafızalarını birleştir
        all_memories = []
        if hasattr(self, 'personal_memory_buffer'):
            all_memories.extend(self.personal_memory_buffer.buffer)
        if hasattr(partner, 'personal_memory_buffer'):
            all_memories.extend(partner.personal_memory_buffer.buffer)
            
        # En travmatik 5 anı (En yüksek loss)
        traumas = sorted(all_memories, key=lambda x: x.get('loss', 0), reverse=True)[:5]
        
        # En zafer dolu 5 anı (En düşük loss - eğer varsa)
        victories = sorted(all_memories, key=lambda x: x.get('loss', 100))[:5]
        
        # Çocuğa yükle
        for mem in traumas + victories:
            # Anı kopyala (Referans olmasın)
            child.personal_memory_buffer.add(copy.deepcopy(mem))
            
        # 5. TABULA RASA (Mizaç Nötr Başlar)
        # Çocuk "Hırslı" doğmaz, "Hırslanmaya meyilli" doğar (ReactionNet sayesinde)
        neutral_temp = {k: 0.5 for k in self.temperament.keys()}
        child.temperament = neutral_temp
        
        return child
    
    def get_recent_fitness(self, window: int = 50):
        """Son N maçtaki fitness ortalaması"""
        if len(self.fitness_history) == 0:
            return 0.5  # Başlangıç değeri
        
        recent = self.fitness_history[-window:]
        return np.mean(recent)
    
    def update_fitness(self, correct: bool, confidence: float):
        """
        Fitness güncelle
        correct: Doğru tahmin mi?
        confidence: Ne kadar emindi?
        """
        # Fitness = doğruluk + güven dengesi
        if correct:
            fitness = 0.5 + 0.5 * confidence  # 0.5 - 1.0 arası
        else:
            fitness = 0.5 * (1 - confidence)  # 0.0 - 0.5 arası
        
        self.fitness_history.append(fitness)

    def evolve_temperament(self, correct: bool, loss: float, is_trauma: bool = False):
        """
        🎭 MİZAÇ EVRİMİ (Neural & Dynamic)
        
        Artık sabit kurallar yok!
        ReactionNet (Yapay Sinir Ağı) olayı analiz eder ve mizaç değişimine karar verir.
        """
        # ReactionNet yoksa oluştur (Eski LoRA'lar için)
        if not hasattr(self, 'reaction_net'):
            self.reaction_net = ReactionNet(input_dim=10, output_dim=5).to(self.device)
            
        # Input vektörü oluştur
        # [Success, Loss, Trauma, Confidence, Ambition, Stress, Resilience, Patience, Risk, Random]
        temp = self.temperament
        
        input_vec = torch.tensor([
            1.0 if correct else 0.0,
            loss,
            1.0 if is_trauma else 0.0,
            temp.get('confidence_level', 0.5),
            temp.get('ambition', 0.5),
            temp.get('stress_tolerance', 0.5),
            temp.get('resilience', 0.5),
            temp.get('patience', 0.5),
            temp.get('risk_appetite', 0.5),
            torch.rand(1).item()  # Kaos faktörü
        ], dtype=torch.float32).to(self.device)
        
        # Neural karar
        with torch.no_grad():
            delta = self.reaction_net(input_vec).cpu().numpy()
            
        # Delta: [ΔConfidence, ΔAmbition, ΔStress, ΔPatience, ΔRisk]
        
        # Değişimleri uygula (Clip 0.0 - 1.0)
        temp['confidence_level'] = max(0.0, min(1.0, temp.get('confidence_level', 0.5) + delta[0]))
        temp['ambition'] = max(0.0, min(1.0, temp.get('ambition', 0.5) + delta[1]))
        temp['stress_tolerance'] = max(0.0, min(1.0, temp.get('stress_tolerance', 0.5) + delta[2]))
        temp['patience'] = max(0.0, min(1.0, temp.get('patience', 0.5) + delta[3]))
        temp['risk_appetite'] = max(0.0, min(1.0, temp.get('risk_appetite', 0.5) + delta[4]))
        
        # Travma kaydı
        if is_trauma:
            self.trauma_history.append({
                'loss': loss, 
                'match': len(self.match_history),
                'severity': loss, # Loss is the severity
                'type': 'loss_trauma'
            })
            
        self.temperament = temp

    def get_status_tags(self) -> List[str]:
        """
        🏷️ LoRA DURUM ETİKETLERİ (Status Tags)
        
        LoRA'nın o anki durumunu özetleyen etiketler.
        Panelde ve loglarda gösterilir.
        """
        tags = []
        
        # 1. Potansiyel (Lazarus Lambda)
        lazarus = getattr(self, '_lazarus_lambda', 0.5)
        if lazarus > 0.8:
            tags.append("🌟 Rising Star")
        elif lazarus > 0.6:
            tags.append("✨ High Potential")
            
        # 2. Yaş / Tecrübe
        match_count = len(self.match_history)
        if match_count > 100:
            tags.append("👴 Veteran")
        elif match_count < 10:
            tags.append("🐣 Rookie")
            
        # 3. Mizaç (Temperament)
        temp = self.temperament
        if temp.get('impulsiveness', 0.5) > 0.8:
            tags.append("💣 Loose Cannon")
        if temp.get('patience', 0.5) > 0.8:
            tags.append("🧘 Zen Master")
        if temp.get('ambition', 0.5) > 0.8:
            tags.append("🦁 Ambitious")
        if temp.get('fear', 0.5) > 0.7:
            tags.append("😨 Traumatized")
            
        # 4. Performans
        fitness = self.get_recent_fitness()
        if fitness > 0.8:
            tags.append("🔥 On Fire")
        elif fitness < 0.3:
            tags.append("❄️ Cold")
            
        # 5. Özel Durumlar
        if getattr(self, 'resurrection_count', 0) > 0:
            tags.append(f"🧟 Resurrected x{self.resurrection_count}")
        if getattr(self, 'lucky_survivals', 0) > 0:
            tags.append("🍀 Survivor")
            
        return tags

class ReactionNet(nn.Module):
    """
    🧠 REACTION NET (Sinirsel Kişilik)
    
    Her LoRA'nın olaylara nasıl tepki vereceğini belirleyen mini beyin.
    Sabit kurallar yerine (if success -> confidence++), bu ağ öğrenir.
    
    Bazı LoRA'lar başarıdan şımarır (Confidence++, Patience--),
    Bazıları başarıdan ders alır (Confidence++, Patience++).
    """
    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)
        self.tanh = nn.Tanh()  # -1 ile +1 arası değişim için
        
        # Rastgele başlat (Her LoRA farklı doğar!)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = self.tanh(self.fc2(h)) * 0.1  # Max değişim 0.1 (Stabilite için)
        return out

class PersonalMemory:
    """
    📔 KİŞİSEL HAFIZA (Subjective Memory)
    
    Her LoRA neyi hatırlayacağına kendi karar verir.
    Hype uzmanı hype maçlarını, defansif uzman 0-0 maçlarını tutar.
    """
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.buffer = []
        # Neyi önemsediğini öğrenen ağırlıklar (Feature Importance)
        self.relevance_weights = np.random.rand(78) 
        
    def should_remember(self, features: np.ndarray, loss: float) -> bool:
        """
        Bu maçı günlüğüme yazmalı mıyım?
        """
        # 1. İlgi düzeyi (Dot product)
        # features (78) * weights (78)
        if len(features) != len(self.relevance_weights):
            return True # Boyut uyuşmazsa güvenli taraf
            
        relevance = np.dot(features, self.relevance_weights)
        
        # 2. Duygusal etki (Loss yüksekse travmatik, düşükse zafer)
        emotional_impact = abs(loss - 0.5) * 2  # 0.5'ten sapma
        
        # Toplam skor
        score = relevance + emotional_impact
        
        # Eşik (Rastgelelik içerir, bazen önemsiz şeyleri de hatırlar)
        threshold = 0.5 + np.random.normal(0, 0.1)
        
        return score > threshold
        
    def add(self, item):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0) # En eskiyi sil
        self.buffer.append(item)
    


    def __repr__(self):
        fitness = self.get_recent_fitness()
        tags = self.get_status_tags()
        tag_str = f" | {' '.join(tags)}" if tags else ""
        return f"LoRA(id={self.id[:8]}, name={self.name}, fitness={fitness:.3f}, gen={self.generation}{tag_str}, matches={len(self.match_history)})"


class OnlineLoRALearner:
    """
    Online öğrenme wrapper'ı
    Her maçta LoRA'yı günceller
    """
    
    def __init__(self, lora: LoRAAdapter, learning_rate: float = 0.0001, device='cpu'):
        self.lora = lora.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Sadece LoRA parametrelerini optimize et
        lora_params = [p for p in self.lora.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(lora_params, lr=learning_rate)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def learn(self, features_np: np.ndarray, base_proba_np: np.ndarray, actual_class_idx: int):
        """
        Bir maçtan öğren
        """
        # Input oluştur
        x = np.concatenate([features_np, base_proba_np]).astype(np.float32)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        
        # Target
        y_tensor = torch.tensor([actual_class_idx], dtype=torch.long, device=self.device)
        
        # Forward + backward
        self.optimizer.zero_grad()
        proba = self.lora(x_tensor)
        loss = self.criterion(proba, y_tensor)
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())
    
    def learn_batch(self, batch_data: List[Dict]):
        """
        Bir batch'ten öğren (yeni maç + buffer)
        """
        if len(batch_data) == 0:
            return 0.0
        
        # Batch oluştur
        x_list = []
        y_list = []
        
        for data in batch_data:
            x = np.concatenate([data['features'], data['base_proba']]).astype(np.float32)
            x_list.append(x)
            y_list.append(data['actual_class_idx'])
        
        x_batch = torch.from_numpy(np.stack(x_list)).to(self.device)
        y_batch = torch.tensor(y_list, dtype=torch.long, device=self.device)
        
        # Forward + backward
        self.optimizer.zero_grad()
        proba = self.lora(x_batch)
        loss = self.criterion(proba, y_batch)
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())

