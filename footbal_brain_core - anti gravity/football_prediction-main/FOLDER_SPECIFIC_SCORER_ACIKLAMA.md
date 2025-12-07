# 🎯 FOLDER SPECIFIC SCORER - AÇIKLAMA

## 📋 GENEL BAKIŞ

`FolderSpecificScorer`, LoRA'ları farklı klasörlere (Einstein, Takım, H2H) göre **özel puanlama** yapan bir sistemdir.

**Mantık:** Her klasör tipi için farklı kriterler kullanılır!

---

## 🧠 EINSTEIN KLASÖRÜ

**Amaç:** Zeka ve potansiyel odaklı LoRA'ları bulmak

**Puanlama Formülü:**
```python
score = (lazarus * 0.7) + (fitness * 0.3)
```

**Açıklama:**
- **Lazarus Lambda (%70):** LoRA'nın öğrenme potansiyeli
  - Yüksek Lazarus = Yüksek potansiyel
  - Gelecekte çok iyi olabilir!
  
- **Fitness (%30):** Mevcut performans
  - Şu anki başarı oranı
  - Ama potansiyel daha önemli!

**Örnek:**
- LoRA A: Lazarus=0.9, Fitness=0.6 → Score = (0.9*0.7) + (0.6*0.3) = **0.81**
- LoRA B: Lazarus=0.5, Fitness=0.8 → Score = (0.5*0.7) + (0.8*0.3) = **0.59**

**Sonuç:** LoRA A daha yüksek puan (potansiyel daha önemli!)

---

## ⚽ TAKIM KLASÖRÜ

**Amaç:** Belirli takımlar için uzman LoRA'ları bulmak

**Puanlama:**
```python
if team_name in lora.specialization:
    score = fitness * 1.5  # Uzman bonus!
else:
    score = fitness * 0.5  # Uzman değil, düşük puan
```

**Açıklama:**
- **Uzman LoRA:** Takım adı specialization'da varsa → **1.5x bonus**
- **Normal LoRA:** Uzman değilse → **0.5x puan** (düşük)

**Örnek:**
- LoRA A: Specialization="Real_Madrid", Fitness=0.7 → Score = 0.7 * 1.5 = **1.05**
- LoRA B: Specialization="Barcelona", Fitness=0.8 → Score = 0.8 * 0.5 = **0.40**

**Sonuç:** LoRA A daha yüksek puan (Real Madrid uzmanı!)

---

## 🆚 H2H (HEAD-TO-HEAD) KLASÖRÜ

**Amaç:** İki takım arası maçlarda başarılı LoRA'ları bulmak

**Puanlama:**
```python
# Şimdilik placeholder: Genel fitness
score = lora.get_recent_fitness()
```

**Gelecek Geliştirme:**
- Collective memory'den bu iki takım arası maçlardaki performansı al
- Örnek: "Real Madrid vs Barcelona" maçlarında %80 başarı → Yüksek puan!

---

## 📊 KULLANIM YERLERİ

### 1. Top LoRA Exporter

`top_lora_exporter.py` içinde kullanılır:

```python
from lora_system.folder_specific_scorer import folder_specific_scorer

# Einstein klasörü için puan hesapla
score = folder_specific_scorer.calculate_score_for_folder(
    lora, 
    "EINSTEIN", 
    match_count=100
)

# Takım klasörü için puan hesapla
score = folder_specific_scorer.calculate_score_for_folder(
    lora, 
    "Team_Real_Madrid", 
    match_count=100
)
```

### 2. LoRA Sıralama

Farklı klasörler için farklı sıralama:
- **Einstein:** Potansiyel yüksek olanlar önce
- **Takım:** O takımın uzmanları önce
- **H2H:** O maç tipinde başarılı olanlar önce

---

## 🔧 GELECEK GELİŞTİRMELER

### 1. Collective Memory Entegrasyonu

```python
# Takım performansı
team_performance = collective_memory.get_team_performance(
    lora.id, 
    team_name
)
score = team_performance * fitness
```

### 2. H2H Detayları

```python
h2h_details = folder_specific_scorer.get_h2h_details(
    lora, 
    "Real_Madrid", 
    "Barcelona", 
    collective_memory
)
# Returns: {"matches": 10, "wins": 8, "score": 0.8}
```

### 3. Dinamik Ağırlıklar

Her klasör için ağırlıklar ayarlanabilir:
- Einstein: Lazarus %80, Fitness %20
- Takım: Uzmanlık %60, Fitness %40

---

## ✅ SONUÇ

**Folder Specific Scorer:**
- ✅ Her klasör için özel puanlama
- ✅ Einstein: Potansiyel odaklı
- ✅ Takım: Uzmanlık odaklı
- ✅ H2H: Gelecekte performans odaklı

**Kullanım:** `top_lora_exporter.py` içinde LoRA'ları klasörlere göre sıralamak için!

