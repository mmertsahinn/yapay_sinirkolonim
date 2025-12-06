# 🧬 SİSTEM ANATOMİSİ - TABANLIK DÖKÜMAN

**Sistemin 5 ana bileşeninin tam anatomisi.**

Sırasıyla:
1. LoRA Sınıfı (Birey)
2. Evrim Motoru (Doğum/Ölüm/Çiftleşme)
3. Scoreboard (Adalet Sistemi)
4. Meta-LoRA (Kolektif Bilinç)
5. Replay Buffer (Kolektif Hafıza)

---

## 1️⃣ LoRA SINIFI (`lora_adapter.py`)

**Dosya:** `lora_system/lora_adapter.py`

### 📌 LoRA NEDİR?

LoRA = **Low-Rank Adaptation**

Bir sinir ağının **ağırlıklarını dondurup**, sadece **küçük ek matrisler (A, B)** ekleyerek öğrenme yapar.

```
W_final = W_frozen + (B @ A) * (alpha / rank)
```

**Avantajlar:**
- Çok az parametre (rank=16 → çok küçük!)
- Hızlı eğitim
- Evrimsel çiftleşme kolay (sadece A, B karıştırılır)

---

### 🧱 MİMARİ

```python
Input (63 boyut)
  ↓
LoRALinear(63 → 128, rank=16)  # fc1
  ↓
ReLU + Dropout(0.1)
  ↓
LoRALinear(128 → 64, rank=16)  # fc2
  ↓
ReLU + Dropout(0.1)
  ↓
LoRALinear(64 → 3, rank=16)    # fc3
  ↓
Softmax
  ↓
Output (3 boyut: HOME, DRAW, AWAY)
```

**Parametreler:**
- Rank: 16 (LoRA matrislerin boyutu)
- Alpha: 16.0 (scaling faktörü)
- Hidden: 128 (gizli katman boyutu)

---

### 🧬 LoRA'NIN ANATOMİSİ (Özellikler)

#### A) TEMEL ÖZELLİKLER

```python
self.id              # Benzersiz ID (8 karakter)
self.name            # İsim (örn: "LoRA_Gen3_abc123")
self.generation      # Nesil (0, 1, 2, ...)
self.parents         # Anne-Baba ID'leri
self.birth_match     # Hangi maçta doğdu?
```

#### B) PERFORMANS METRİKLERİ

```python
self.match_history     # Maç geçmişi
self.fitness_history   # Fitness skoru geçmişi (0-1 arası)
self.specialization    # Uzmanlık (örn: "hype_expert")
```

**Fitness Hesaplaması:**

```python
def update_fitness(self, correct: bool, confidence: float):
    if correct:
        fitness = 0.5 + 0.5 * confidence  # 0.5 - 1.0
    else:
        fitness = 0.5 * (1 - confidence)  # 0.0 - 0.5
    
    self.fitness_history.append(fitness)
```

**Mantık:**
- Doğru tahmin → 0.5-1.0 (yüksek güvenle doğru = 1.0)
- Yanlış tahmin → 0.0-0.5 (çok eminken yanılma = 0.0!)

#### C) SOSYAL ÖZELLİKLER

```python
self.pattern_attractions = {}  # Pattern çekimleri
self.social_bonds = {}         # Diğer LoRA'larla bağlar
self.main_goal = None          # Ana hedef
self.trauma_history = []       # Travma geçmişi
```

#### D) 🎭 KİŞİLİK ÖZELLİKLERİ (Genetik!)

```python
self.temperament = {
    'independence': 0.3-0.9,        # Bağımsızlık
    'social_intelligence': 0.3-0.9, # Sosyal zeka
    'herd_tendency': 0.1-0.8,       # Sürü eğilimi
    'contrarian_score': 0.0-0.7,    # Karşıt görüş
    'confidence_level': 0.4-0.9,    # Özgüven
    'risk_appetite': 0.2-0.9,       # Risk iştahı
    'patience': 0.3-0.9,            # Sabır
    'impulsiveness': 0.1-0.8,       # Dürtüsellik
    'stress_tolerance': 0.4-0.9     # Stres toleransı
}
```

**GENETİK AKTARIM:**
- Anne + Baba kişilikleri karışır (ortalama + ±%20 mutasyon)
- %5 şans: Tamamen yeni kişilik (alien bebek!)

#### E) 🧠 KİŞİSEL HAFIZA (YENİ!)

```python
self.personal_memory = {
    'learned_patterns': {},    # Kendi öğrenmeleri
    'learning_history': [],    # Ne zaman ne öğrendi
    'observed_others': {},     # Başkalarından ne gördü
    'adjustments': []          # Kendi değişimleri
}
```

Bu, LoRA'nın **kendi öğrenme tarihi**!

#### F) 🍀 HAYATTA KALMA

```python
self.lucky_survivals = 0     # Kaç kez şanslı kurtuldu
self.resurrection_count = 0  # Kaç kez dirildi
self.children_count = 0      # Kaç çocuk doğurdu
```

---

### 🔄 ÖNEMLİ METODLAR

#### 1) `predict(features, base_proba, device)`

**Ne yapar:** Maç tahmini yapar.

**Input:**
- `features`: 60 boyutlu özellik vektörü
- `base_proba`: Ensemble'dan gelen 3 boyutlu tahmin

**Output:**
- `proba`: 3 boyutlu tahmin (HOME, DRAW, AWAY)

**Nasıl:**
```python
# Input: [60 features + 3 base_proba] = 63 boyut
x = concat(features, base_proba)
x = forward(x)  # Ağdan geçir
return softmax(x)  # 0-1 arası normalize
```

#### 2) `clone()`

**Ne yapar:** Bu LoRA'nın kopyasını oluşturur.

**Kullanım:** Evrimde mutasyon için.

```python
new_lora = lora.clone()
new_lora.temperament = lora.temperament.copy()  # Kişilik kopyalanır!
```

#### 3) `get_recent_fitness(window=50)`

**Ne yapar:** Son N maçtaki ortalama fitness.

```python
recent = self.fitness_history[-50:]
return np.mean(recent)
```

#### 4) `get_all_lora_params()` / `set_all_lora_params()`

**Ne yapar:** Çiftleşme ve klonlama için LoRA parametrelerini al/kur.

```python
params = lora.get_all_lora_params()
# params = {
#     'fc1': {'lora_A': tensor, 'lora_B': tensor},
#     'fc2': {...},
#     'fc3': {...}
# }

child.set_all_lora_params(params)
```

---

### 🎓 ONLINELoRALEARNER (Öğrenme Wrapper)

**Ne yapar:** Her maçtan sonra LoRA'yı günceller (gradient descent).

```python
learner = OnlineLoRALearner(lora, learning_rate=0.0001)
loss = learner.learn_batch(batch)
```

**Nasıl:**
- Sadece LoRA parametrelerini (A, B) optimize eder
- Ana ağırlıklar DONUK kalır
- Adam optimizer
- CrossEntropyLoss

---

## 2️⃣ EVRİM MOTORU (`chaos_evolution.py`)

**Dosya:** `lora_system/chaos_evolution.py`

### 📌 EVRİM MOTORU NEDİR?

Popülasyonun **doğal seleksiyon** ile evrimini yöneten motor.

**3 Ana Süreç:**
1. **Ölüm** (Fitness < threshold → ölüm + şanslı kurtuluş)
2. **Üreme** (Fitness > threshold → partner bul → çiftleş)
3. **Spontane Doğum** (Rastgele alien LoRA doğar!)

---

### ⚙️ PARAMETReler (Config)

```yaml
population:
  min_population: 5        # Minimum popülasyon (yoksa diriltme!)
  max_population: null     # Üst limit YOK! (doğa dengeyi kurar)

death:
  threshold: 0.05          # Fitness < 0.05 → ölüm riski
  lucky_survival_chance: 0.50  # %50 şanslı kurtuluş!

reproduction:
  fitness_threshold: 0.60  # Fitness > 0.60 → üreme hakkı
  chance_per_match: 0.06   # Her maçta %6 üreme şansı
  
  partner_selection:
    random: 0.30           # %30 rastgele partner (KAOS!)
    strongest: 0.30        # %30 en güçlü
    weakest: 0.20          # %20 en zayıf (sürpriz!)
    complementary: 0.20    # %20 tamamlayıcı (farklı özellik)

noise:
  crossover:
    base_noise_max: 0.3    # Çiftleşmede gürültü max %30
    mega_noise_chance: 0.10  # %10 MEGA gürültü!
  
  mutation:
    param_mutation_chance: 0.15  # %15 mutasyon
    shock_mutation_chance: 0.05  # %5 şok mutasyon!
  
  spontaneous_birth:
    chance_per_match: 0.04  # %4 spontane doğum (alien!)
```

---

### 💀 ÖLÜM SİSTEMİ

#### Ölüm Kriterleri:

```python
def evolution_step():
    for lora in population:
        fitness = lora.get_recent_fitness()
        
        if fitness < self.death_threshold:  # 0.05
            # ÖLÜM RİSKİ!
            
            if random() < self.lucky_survival_chance:  # 0.50
                # 🍀 ŞANSLI KURTULUŞ!
                lora.lucky_survivals += 1
                survivors.append(lora)
            else:
                # 💀 ÖLÜM!
                death_reason = _determine_death_reason(lora, fitness)
                # Ölüm loglanır
```

#### Ölüm Sebepleri:

```python
def _determine_death_reason(lora, fitness):
    if fitness < 0.02:
        return "Kritik düşük fitness"
    
    if hasattr(lora, 'goalless_death_risk') and risk > 0.5:
        return "Hedefsizlik sürüklenmesi"
    
    if len(lora.trauma_history) > 5:
        return "Aşırı travma"
    
    return "Düşük performans"
```

---

### 👶 ÜREME SİSTEMİ (DOĞAL BAĞ BAZLI!)

#### Doğal Üreme Şansı:

```python
def _calculate_natural_reproduction_chance(lora, population_size, alarm_info):
    # 1) SOSYAL BAĞ (40%)
    if lora.social_bonds:
        max_bond = max(lora.social_bonds.values())
        bond_score = max_bond * 0.40
    else:
        bond_score = 0.10  # Bağsız = düşük şans
    
    # 2) FITNESS (30%)
    fitness_score = lora.get_recent_fitness() * 0.30
    
    # 3) HIRSLIK (15%)
    ambition_score = lora.temperament['ambition'] * 0.15
    
    # 4) NÜFUS FAKTÖRÜ (15%) - Dünya gibi artar!
    # 50 LoRA: 0.50, 100: 0.60, 200: 0.75, 400: 0.95
    population_factor = 0.50 + min(population_size / 1000, 0.45)
    population_score = population_factor * 0.15
    
    # TOPLAM
    natural_chance = bond_score + fitness_score + ambition_score + population_score
    
    # Alarm varsa (soy azalırsa) alarm_chance ile karşılaştır
    if alarm_info:
        alarm_chance = base_chance * alarm_info['reproduction_multiplier']
        final_chance = max(natural_chance, alarm_chance)
    
    return min(0.95, final_chance)  # Max %95
```

**Mantık:**
- Güçlü sosyal bağ → Daha çok çocuk!
- Sağlıklı → Daha çok çocuk!
- Hırslı → Daha çok çocuk!
- Kalabalık dünya → Doğum oranı artar! (gerçek dünya gibi!)

#### Partner Seçimi:

```python
def select_partner(lora):
    rand = random()
    
    # %30: Tamamen rastgele (KAOS!)
    if rand < 0.30:
        return random.choice(others)
    
    # %30: En güçlü (klasik evrim)
    elif rand < 0.60:
        return max(others, key=lambda x: x.get_recent_fitness())
    
    # %20: En zayıf (sürpriz potansiyeli!)
    elif rand < 0.80:
        return min(others, key=lambda x: x.get_recent_fitness())
    
    # %20: Tamamlayıcı (farklı özellik)
    else:
        return _find_complementary(lora, others)
```

**Tamamlayıcı Partner:**
- En FARKLI parametre yapısına sahip LoRA seçilir.
- Amaç: Çeşitliliği artırmak!

---

### 🌪️ KAOTİK ÇİFTLEŞME (Crossover)

```python
def chaotic_crossover(parent1, parent2):
    child = LoRAAdapter()
    
    for layer in ['fc1', 'fc2', 'fc3']:
        for matrix in ['lora_A', 'lora_B']:
            # Her parametrede FARKLI gürültü!
            noise_level = random(0, 0.3)  # %0-%30 gürültü
            
            # Anne veya baba?
            if random() < 0.5:
                base = parent1.params[layer][matrix]
            else:
                base = parent2.params[layer][matrix]
            
            # Gürültü ekle
            result = base + randn_like(base) * noise_level
            
            # %10: MEGA GÜRÜLTÜ!
            if random() < 0.10:
                avg = (parent1.params + parent2.params) / 2
                mega_noise = randn_like(avg) * 0.5
                result = avg + mega_noise
            
            child.params[layer][matrix] = result
    
    # 🎭 KİŞİLİK GENETİĞİ
    child.temperament = _inherit_temperament(parent1, parent2)
    
    # Anne-Baba çocuk sayısını artır
    parent1.children_count += 1
    parent2.children_count += 1
    
    return child
```

**Kişilik Kalıtımı:**

```python
def _inherit_temperament(parent1, parent2):
    # %5 şans: TAM YENİ KİŞİLİK (alien bebek!)
    if random() < 0.05:
        return random_temperament()
    
    # Anne + Baba ortalaması + ±%20 mutasyon
    child_temp = {}
    for trait in temperament.keys():
        p1_val = parent1.temperament[trait]
        p2_val = parent2.temperament[trait]
        
        avg = (p1_val + p2_val) / 2
        mutation = random(-0.2, 0.2)
        final_val = clamp(avg + mutation, 0, 1)
        
        child_temp[trait] = final_val
    
    return child_temp
```

---

### 👽 SPONTANE DOĞUM (Alien LoRA)

```python
def spawn_random_lora():
    alien = LoRAAdapter()
    alien.name = f"LoRA_Alien_{id}"
    alien.generation = 0
    alien.parents = []
    
    # 👽 EKSTREM KİŞİLİK!
    alien.temperament = {
        'independence': 0.7-1.0,        # ÇOK bağımsız!
        'social_intelligence': 0.0-0.5, # Sosyal zeka düşük
        'herd_tendency': 0.0-0.3,       # Sürüye uymaz!
        'contrarian_score': 0.5-1.0,    # ÇOK karşıt!
        'confidence_level': 0.6-1.0,    # Aşırı özgüvenli
        'risk_appetite': 0.7-1.0,       # Risk sever!
        'patience': 0.1-0.5,            # Sabırsız
        'impulsiveness': 0.6-1.0,       # Dürtüsel
        'stress_tolerance': 0.3-0.8
    }
    
    return alien
```

**Alien LoRA Özellikleri:**
- Hiç ebeveyn yok!
- Ekstrem kişilik (bağımsız, karşıt, risk sever!)
- Sıfır nesil (Gen0)
- Sürpriz potansiyeli yüksek!

---

### 🔄 EVRİM ADIMI (Her Maç Sonrası)

```python
def post_match_update(alarm_info=None):
    match_count += 1
    
    # 1) ÖLÜMLER (fitness < threshold)
    # 2) ÜREMELER (fitness > threshold + doğal şans)
    # 3) SPONTANE DOĞUM (%4 şans)
    # 4) SOY TÜKENMESİ KONTROLÜ (population == 0 → diriltme!)
    
    return events  # birth, death, lucky_survival, spontaneous_birth
```

---

## 3️⃣ SCOREBOARD FORMÜLÜ (`advanced_score_calculator.py`)

**Dosya:** `lora_system/advanced_score_calculator.py`

### 📌 SCOREBOARD NEDİR?

LoRA'ları **adil bir şekilde sıralamak** için kullanılan gelişmiş formül.

**Amaç:**
- Genç yetenekler yaşlıları geçebilsin!
- Trend önemli (yükseliyor mu?)
- İstikrar ödüllendirilsin
- Yaşa göre normalize (genç = daha az beklenti)

---

### 🧮 FORMÜL

```
ADVANCED_SCORE = 
  (Weighted_Recent × 0.30) +      # Son performans (ağırlıklı)
  (Age_Normalized × 0.25) +       # Yaşa göre normalize başarı
  (Peak_Performance × 0.20) +     # En iyi dönem
  (Momentum × 0.15) +             # Trend (yükseliyor mu?)
  (Consistency × 0.10)            # İstikrar
```

**Toplam: 1.00** (0-1 arası normalize)

---

### 📊 BİLEŞENLER

#### 1) WEIGHTED RECENT (30%)

**Ne:** Son performans (exponential weighted average)

**Nasıl:**
```python
history = lora.fitness_history[-50:]  # Son 50 maç

# Exponential ağırlıklar
weights = []
for i in range(len(history)):
    weight = exp(i / len(history))  # Son maç = en yüksek ağırlık
    weights.append(weight)

weights = weights / sum(weights)  # Normalize

weighted_avg = dot(history, weights)
```

**Mantık:** Son maçlar daha önemli!

---

#### 2) AGE-NORMALIZED SUCCESS (25%)

**Ne:** Yaşa göre normalize başarı

**Nasıl:**
```python
success_rate = count(fitness > 0.5) / len(history)
age = match_count - lora.birth_match

# Beklenen başarı (yaşa göre artar!)
# 0-50 maç: %50 beklenir
# 50-100: %55
# 100-200: %60
# 200+: %65
expected = 0.50 + min(age / 400, 0.15)  # Max +15%

# Normalize: Gerçek / Beklenen
normalized = success_rate / expected
```

**Mantık:**
- Genç LoRA: %60 başarı = 0.90 skor (**ÇOK İYİ!**)
- Yaşlı LoRA: %60 başarı = 0.70 skor (**ORTA**)

**Genç yetenekler avantajlı!**

---

#### 3) PEAK PERFORMANCE (20%)

**Ne:** En iyi 20 maçlık dönem performansı

**Nasıl:**
```python
# 20 maçlık sliding window
best_avg = 0.0
for i in range(len(history) - 19):
    window = history[i:i+20]
    window_avg = mean(window)
    best_avg = max(best_avg, window_avg)

return best_avg
```

**Mantık:** Potansiyeli gösterir! (Bir dönem ne kadar iyi olmuş?)

---

#### 4) MOMENTUM (15%)

**Ne:** Trend (yükseliyor mu düşüyor mu?)

**Nasıl:**
```python
# Son 20 maç vs Önceki 20 maç
recent_20 = history[-20:]
previous_20 = history[-40:-20]

recent_avg = mean(recent_20)
previous_avg = mean(previous_20)

momentum_ratio = recent_avg / previous_avg

# 0.5-1.5 arası → 0-1 arası normalize
# 0.5 → düşüş (0.0)
# 1.0 → sabit (0.5)
# 1.5 → artış (1.0)
normalized_momentum = (momentum_ratio - 0.5) / 1.0
normalized_momentum = clamp(normalized_momentum + 0.5, 0, 1)
```

**Mantık:** Yükseliş trendi ödüllendirilir!

---

#### 5) CONSISTENCY (10%)

**Ne:** İstikrar (Variance ne kadar düşük?)

**Nasıl:**
```python
history = fitness_history[-50:]  # Son 50 maç

mean = sum(history) / len(history)
variance = sum((f - mean)^2 for f in history) / len(history)
std = sqrt(variance)

# Düşük std = yüksek consistency
# std: 0.0 → 1.0, 0.3+ → 0.0
consistency = max(0, 1.0 - (std / 0.3))
```

**Mantık:** İstikrarlı LoRA > Kararsız LoRA

---

### 🎯 ÖRNEK HESAPLAMA

**LoRA_Einstein:**
- Fitness history: 50 maç, avg=0.75
- Yaş: 120 maç
- Son 20 maç: 0.80 avg
- Önceki 20 maç: 0.70 avg
- Peak 20 maç: 0.85
- Std: 0.12

**Hesaplama:**

```
Weighted Recent: 0.78 * 0.30 = 0.234
Age Normalized: (0.75 / 0.59) * 0.25 = 0.318
  (expected = 0.50 + 120/400*0.15 = 0.545 ≈ 0.59)
Peak: 0.85 * 0.20 = 0.170
Momentum: ((0.80/0.70 - 0.5) / 1.0 + 0.5) * 0.15 = 0.105
  (momentum_ratio = 1.14 → normalized = 0.64)
Consistency: (1 - 0.12/0.3) * 0.10 = 0.060

TOPLAM = 0.234 + 0.318 + 0.170 + 0.105 + 0.060 = 0.887
```

**Einstein'ın Advanced Score: 0.887 / 1.00 (%88.7)**

---

## 4️⃣ META-LoRA (`meta_lora.py`)

**Dosya:** `lora_system/meta_lora.py`

### 📌 META-LoRA NEDİR?

**"Hangi LoRA'yı dinleyelim?"** kararını veren üst akıl.

**Mekanizma:** Attention (Dikkat)

**Analoji:**
- LoRA'lar = Uzmanlar
- Meta-LoRA = Moderatör
- Her maç = Farklı uzmanlar dinlenir!

---

### 🧠 ATTENTION MEKANİZMASI

```
Maç Özellikleri (63 boyut)
  ↓
Query Network
  ↓
Query Vektörü (16 boyut)
  ↓
Her LoRA → Key Vektörü (16 boyut)
  ↓
Attention Scores = Query @ Keys^T
  ↓
Softmax → Attention Weights (0-1 arası, toplam=1)
  ↓
Weighted Average (LoRA tahminlerini ağırlıklandır)
  ↓
Final Prediction
```

---

### 🔑 KEY VEKTÖRÜ (LoRA'nın İmzası)

```python
def get_lora_key(lora):
    params = lora.get_all_lora_params()
    
    features = []
    for layer in ['fc1', 'fc2', 'fc3']:
        for matrix in ['lora_A', 'lora_B']:
            # İstatistikler: mean, std
            features.extend([
                params[layer][matrix].mean(),
                params[layer][matrix].std()
            ])
    
    # 12 özellik → 16 boyuta pad
    key = tensor(features[:16])
    return key
```

**Key = LoRA'nın imzası** (parametrelerinden çıkarılır)

---

### ⚖️ ATTENTION WEIGHTS HESAPLAMA

```python
def forward(match_features, lora_population):
    # Query: Maç özelliklerinden
    query = query_net(match_features)  # (1, 16)
    
    # Keys: Her LoRA'dan
    keys = [get_lora_key(lora) for lora in lora_population]
    keys = stack(keys)  # (num_loras, 16)
    
    # Attention scores
    scores = query @ keys.T  # (1, num_loras)
    
    # Softmax
    attention_weights = softmax(scores)  # Toplam = 1.0
    
    return attention_weights
```

---

### 🎯 AGGREGATE PREDICTIONS (Nihai Tahmin)

```python
def aggregate_predictions(match_features, base_proba, lora_population):
    # 1) Attention weights hesapla
    attention_weights = forward(match_features, lora_population)
    
    # 2) Her LoRA'dan tahmin al
    individual_probas = []
    for lora in lora_population:
        lora_proba = lora.predict(match_features, base_proba)
        individual_probas.append(lora_proba)
    
    # 3) Weighted average
    aggregated_proba = sum(
        individual_probas * attention_weights[:, None],
        axis=0
    )
    
    # 4) Normalize
    aggregated_proba /= aggregated_proba.sum()
    
    return aggregated_proba
```

---

### 📊 ÖRNEK

**Maç:** Galatasaray - Fenerbahçe

**Popülasyon:**
- LoRA_Hype (derbi uzmanı): Fitness=0.80
- LoRA_Odds (oran uzmanı): Fitness=0.70
- LoRA_Alien (kaotik): Fitness=0.50

**Attention Weights:**
```
LoRA_Hype: 0.60   (Derbi maçı → Hype uzmanı öne çıkıyor!)
LoRA_Odds: 0.30
LoRA_Alien: 0.10
```

**Individual Predictions:**
```
LoRA_Hype:  [0.30, 0.20, 0.50]  (AWAY ağırlıklı)
LoRA_Odds:  [0.50, 0.30, 0.20]  (HOME ağırlıklı)
LoRA_Alien: [0.10, 0.10, 0.80]  (AWAY çok ağırlıklı!)
```

**Aggregated:**
```
Final = 0.60*[0.30, 0.20, 0.50] + 0.30*[0.50, 0.30, 0.20] + 0.10*[0.10, 0.10, 0.80]
      = [0.18, 0.12, 0.30] + [0.15, 0.09, 0.06] + [0.01, 0.01, 0.08]
      = [0.34, 0.22, 0.44]
```

**Final Prediction: AWAY (44%)**

**Meta-LoRA derbi uzmanını dinledi!**

---

### 🆚 SimpleMetaLoRA (Alternatif)

**Basitleştirilmiş versiyon:**
- Attention yok!
- Sadece fitness bazlı ağırlıklandırma

```python
def aggregate_predictions(match_features, base_proba, lora_population):
    # Her LoRA'dan tahmin + fitness al
    probas = []
    fitnesses = []
    
    for lora in lora_population:
        probas.append(lora.predict(match_features, base_proba))
        fitnesses.append(lora.get_recent_fitness())
    
    # Fitness'i ağırlık olarak kullan (softmax)
    weights = exp(fitnesses * 5)
    weights /= weights.sum()
    
    # Weighted average
    aggregated = sum(probas * weights[:, None], axis=0)
    
    return aggregated
```

**Basit ama etkili!**

---

## 5️⃣ REPLAY BUFFER (`replay_buffer.py`)

**Dosya:** `lora_system/replay_buffer.py`

### 📌 REPLAY BUFFER NEDİR?

**Önemli maçları saklar** ve online öğrenme için kullanır.

**Amaç:**
- Modelin yanıldığı maçları hatırla!
- Sürpriz skorları hatırla!
- Yüksek hype maçları hatırla!

---

### 📦 BUFFER YAPISI

```python
storage = [
    {
        'features': np.array (60,),
        'base_proba': np.array (3,),
        'lora_proba': np.array (3,),
        'actual_class_idx': int,
        'actual_result': str,
        'loss': float,
        'surprise': float,  # 1 - p(actual)
        'hype': float,      # total_tweets
        'goal_diff': int,
        'match_date': str,
        'home_team': str,
        'away_team': str,
        'league': str,
        'predicted_class': str,
        'correct': bool,
        'importance': float  # 0-1 arası
    },
    ...
]
```

**Max size:** 1000 maç

---

### 🎯 IMPORTANCE (Önem Skoru)

**Ne kadar önemli?**

```python
def _calculate_importance(example):
    importance = 0.0
    
    # 1) LOSS (30%)
    loss = example['loss']
    importance += min(loss, 2.0) * 0.3  # Max 0.6 katkı
    
    # 2) SURPRISE (30%)
    surprise = example['surprise']  # 1 - p(actual)
    importance += surprise * 0.3
    
    # 3) GOL FARKI (30%)
    goal_diff = abs(example['goal_diff'])
    if goal_diff >= 5:
        importance += 0.3  # 5+ fark = ÇOK önemli!
    elif goal_diff >= 3:
        importance += 0.2
    elif goal_diff >= 2:
        importance += 0.1
    
    # 4) HYPE (10%)
    hype = example['hype']
    normalized_hype = min(hype / 50000, 1.0)
    importance += normalized_hype * 0.2
    
    return importance
```

**Toplam: 0-1.0 arası**

**Yüksek önem:**
- Model çok yanıldı (yüksek loss)
- Beklenmedik sonuç (yüksek surprise)
- Aşırı skor farkı (7-0, 5-1, vs.)
- Çok hype'lı maç (derbi, vs.)

---

### 🎲 SAMPLING (Örnekleme)

#### 1) WEIGHTED SAMPLING (Ağırlıklı)

```python
def sample(batch_size=16):
    # Önem skorlarını ağırlık olarak kullan
    importances = [ex['importance'] for ex in storage]
    probs = importances / sum(importances)
    
    # Ağırlıklı örnekleme
    indices = np.random.choice(
        len(storage),
        size=batch_size,
        replace=False,
        p=probs
    )
    
    return [storage[i] for i in indices]
```

**Önemli maçlar daha sık örneklenir!**

#### 2) UNIFORM SAMPLING (Eşit)

```python
def sample_uniform(batch_size=16):
    return random.sample(storage, batch_size)
```

---

### 🗑️ PRUNING (Temizleme)

**Buffer dolarsa:**

```python
def _prune():
    # Importance'a göre sırala
    storage.sort(key=lambda x: x['importance'], reverse=True)
    
    # En önemli max_size kadarını tut
    storage = storage[:max_size]
```

**En az önemliler atılır!**

---

### 💾 KAYDETME / YÜKLEME

```python
def save(filepath):
    joblib.dump({
        'storage': storage,
        'max_size': max_size,
        'total_added': total_added,
        'total_pruned': total_pruned
    }, filepath)

def load(filepath):
    data = joblib.load(filepath)
    storage = data['storage']
    # ...
```

---

### 🔍 FILTERING (Filtreleme)

```python
def filter_by_criteria(**criteria):
    # Örnek: goal_diff=5, correct=False
    results = []
    
    for ex in storage:
        match = True
        for key, value in criteria.items():
            if ex[key] != value:
                match = False
                break
        
        if match:
            results.append(ex)
    
    return results
```

**Örnek kullanım:**

```python
# 5 gol farkla yanlış tahmin edilen maçlar
buffer.filter_by_criteria(goal_diff=5, correct=False)

# Derbi maçları
buffer.filter_by_criteria(league='Süper Lig', hype_threshold=50000)
```

---

### 📊 İSTATİSTİKLER

```python
def get_stats():
    return {
        'size': len(storage),
        'max_size': max_size,
        'total_added': total_added,
        'total_pruned': total_pruned,
        'avg_importance': mean(importances),
        'max_importance': max(importances),
        'avg_loss': mean(losses),
        'avg_surprise': mean(surprises),
        'high_importance_count': count(importance > 0.7)
    }
```

---

## 🎯 ÖZET: 5 BİLEŞEN VE ROLLERI

| Bileşen | Rol | Analoji |
|---------|-----|---------|
| **LoRA** | Bireysel uzman, tahmin yapar | İnsan |
| **Evrim Motoru** | Doğal seleksiyon, üreme, ölüm | Doğa |
| **Scoreboard** | Adil sıralama, genç yetenek tespiti | Hakem |
| **Meta-LoRA** | Uzmanları ağırlıklandır, en iyiyi seç | Moderatör |
| **Replay Buffer** | Önemli maçları hatırla, öğren | Hafıza |

---

## ❓ KRITIK SORULAR

### 1) LoRA hakkında:

**Q:** LoRA neden bu kadar küçük?  
**A:** Çünkü rank=16! Ana ağırlıklar donuk, sadece 2 küçük matris (A, B) eğitiliyor. Hızlı + Az hafıza.

**Q:** Kişilik genetik olarak geçiyor, ama nasıl kullanılıyor?  
**A:** Şu an **Collective Memory** yorumlamasında! Bağımsız LoRA başkalarını az dinler, Sosyal Zeki çok dinler!

**Q:** Personal memory tam olarak ne işe yarar?  
**A:** Her LoRA kendi öğrenme tarihini tutar. Başkaları bu öğrenmeyi görebilir ve kendi mizacına göre yorumlayabilir!

---

### 2) Evrim hakkında:

**Q:** Neden %50 şanslı kurtuluş?  
**A:** Çünkü düşük fitness = kötü şans + kötü performans karışımı olabilir. %50 şans veririz, belki düzelir!

**Q:** Alien LoRA neden önemli?  
**A:** Çeşitlilik! Eğer tüm LoRA'lar aynı tip kişiliğe sahipse, alien LoRA farklı bakış açısı getirir.

**Q:** Partner seçimi neden %30 rastgele?  
**A:** KAOS! Bazen en güçlü + en zayıf = süper çocuk çıkabilir! Rastgelelik sürprizlere kapı açar.

---

### 3) Scoreboard hakkında:

**Q:** Neden yaş normalize ediliyor?  
**A:** Genç LoRA'lara şans vermek için! Yoksa yaşlı LoRA'lar hep üstte olur, genç yetenekler hiç yükselemez.

**Q:** Momentum neden %15?  
**A:** Trend önemli ama yeterince uzun veri gerektirir (40 maç). Bu yüzden ağırlığı düşük.

**Q:** İstikrar neden sadece %10?  
**A:** Çünkü istikrar önemli ama çok ödüllendirirsek riskten kaçınırlar. Risk almak da gerekiyor!

---

### 4) Meta-LoRA hakkında:

**Q:** Attention vs Simple Meta, hangisi daha iyi?  
**A:** Attention daha sofistike (maç özelliklerine göre dinamik). Simple daha basit (sadece fitness). İkisi de iyi, Attention hafif daha iyi.

**Q:** Attention weights kaç LoRA'ya dağılır?  
**A:** Hepsine! Ama genelde top 3-5 LoRA %70-80 ağırlığı alır.

**Q:** Meta-LoRA eğitiliyor mu?  
**A:** **HAYIR!** Şu an statik. Gelecekte eğitilebilir (meta-learning).

---

### 5) Replay Buffer hakkında:

**Q:** 1000 maç yeterli mi?  
**A:** Şu an yeterli. Gerekirse 2000-5000'e çıkarılabilir.

**Q:** Buffer'dan ne sıklıkla örnekleniyor?  
**A:** Her maçta! Yeni maç + buffer'dan 16 örnek = toplam 17 örnek ile LoRA öğrenir.

**Q:** Buffer'ı manuel olarak düzenleyebilir miyiz?  
**A:** EVET! `add_user_selected_matches()` ile özel maçlar eklenebilir (örn: ani değişiklikler, özel durumlar).

---

## 🚀 GELECEK GELİŞTİRMELER

1. **Meta-LoRA Eğitimi:** Attention weights öğrenilir hale gelebilir.
2. **Buffer Intelligence:** Kullanıcı buffer'a "Turning Point" maçları ekleyebilir.
3. **LoRA Self-Awareness:** LoRA kendi wallet'ını okuyup kendini optimize edebilir (şu an var, daha da geliştirilecek!).
4. **Aşk & Evlilik Sistemi:** Sosyal bağ = %100 → evlilik!
5. **AI Psikolog Raporu:** LoRA'ların psikolojik durumu analiz edilir.

---

## 📚 KAYNAKLAR

- **LoRA Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **Attention:** "Attention Is All You Need" (Vaswani et al., 2017)
- **Replay Buffer:** "Experience Replay in Deep Reinforcement Learning" (Mnih et al., 2015)
- **Genetic Algorithms:** "Genetic Algorithms in Search, Optimization, and Machine Learning" (Goldberg, 1989)

---

**🎉 ANATOMİ TAMAM!**

Şimdi sistemin her hücresini biliyorsun!

**Soru varsa sor, kod değişikliği istersen söyle!** 🚀

