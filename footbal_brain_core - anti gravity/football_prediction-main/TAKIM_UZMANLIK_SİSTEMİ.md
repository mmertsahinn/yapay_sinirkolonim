# 🏆 TAKIM UZMANLIK SİSTEMİ

## Genel Bakış

Einstein/Newton/Darwin dışında, **takım bazlı uzmanlık sistemi!**

Her takım için:
- Win Experts (Kazanan tahmin)
- Goal Experts (Gol tahmin)
- Hype Experts (Hype doğruluk)
- VS Experts (Rakip bazlı)

---

## 📁 Klasör Yapısı

```
takım_uzmanlıkları/
├── Manchester_United/
│   ├── 🎯_WIN_EXPERTS/
│   │   ├── {lora_id_1}.pt
│   │   ├── {lora_id_2}.pt
│   │   ├── {lora_id_3}.pt
│   │   ├── {lora_id_4}.pt
│   │   ├── {lora_id_5}.pt
│   │   └── 🎯_win_experts_top5.txt
│   │
│   ├── ⚽_GOAL_EXPERTS/
│   │   ├── {lora_id}.pt (Top 5)
│   │   └── ⚽_goal_experts_top5.txt
│   │
│   ├── 🔥_HYPE_EXPERTS/
│   │   ├── {lora_id}.pt (Top 5)
│   │   └── 🔥_hype_experts_top5.txt
│   │
│   ├── 🆚_VS_Liverpool/
│   │   ├── {lora_id}.pt (Top 5)
│   │   └── 🆚_vs_liverpool_top5.txt
│   │
│   ├── 🆚_VS_Arsenal/
│   ├── 🆚_VS_Chelsea/
│   ├── ... (her rakip için)
│   │
│   └── manchester_united_MASTER.txt  ← Tüm özet!
│
├── Liverpool/
├── Real_Madrid/
├── ... (her takım için)
```

---

## 🎯 Uzmanlık Tipleri

### 1. Win Expert (🎯)
- **Kriter:** Takımın maçlarında kazanan (Home/Draw/Away) doğru tahmin eder
- **Minimum:** 20 maç
- **Sıralama:** Advanced Score (Başarı + Deneyim + İstikrar + Peak + Momentum + Maç Sayısı)

### 2. Goal Expert (⚽)
- **Kriter:** Takımın atacağı golleri doğru tahmin eder
- **Minimum:** 20 maç
- **Metrik:** MAE (Mean Absolute Error) - Düşük MAE = İyi

### 3. Hype Expert (🔥)
- **Kriter:** Takım hype'lıyken (home_support > 0.7) doğru tahmin yapar
- **Minimum:** 20 maç
- **Özellik:** Upset detection (hype yanlışsa sezebilir!)

### 4. VS Expert (🆚)
- **Kriter:** İki takımın eşleşmesinde uzman
- **Minimum:** 5 maç (daha az eşleşme olur)
- **Özellik:** H2H (Head to Head) uzmanı

---

## 📊 Advanced Score Formülü

```python
SKOR = 
  Accuracy      × 0.30 +  # Başarı oranı
  Age           × 0.20 +  # Deneyim (yaş)
  Consistency   × 0.15 +  # İstikrar (varyans düşük)
  Peak          × 0.15 +  # En iyi dönem
  Momentum      × 0.10 +  # Trend (yükseliyor mu?)
  Match Count   × 0.10    # Maç sayısı bonusu
```

### Örnek Hesaplama:

```
LoRA_abc123, Manchester Win Expert:

Accuracy: %92.5 → 0.925 × 0.30 = 0.278
Age: 187 maç → 0.9 × 0.20 = 0.180
Consistency: Variance 0.05 → 0.95 × 0.15 = 0.143
Peak: En iyi 10 maç %98 → 0.98 × 0.15 = 0.147
Momentum: +%8 trend → 0.89 × 0.10 = 0.089
Match Count: 45 maç → 0.6 × 0.10 = 0.060

TOPLAM SKOR: 0.897
```

---

## ☠️ Ölümsüzlük Sistemi

### Çoklu Uzmanlık = Ölümsüzlük!

```python
10+ uzmanlık → %98 ölümsüz (Tanrı!)
7+ uzmanlık  → %95 ölümsüz (Efsane!)
5+ uzmanlık  → %90 ölümsüz (Süper uzman!)
3+ uzmanlık  → %70 ölümsüz (Çok uzman)
2 uzmanlık   → %50 ölümsüz (İkili uzman)
1 uzmanlık   → %25 ölümsüz (Tekli uzman)
0 uzmanlık   → %0 ölümsüz (Normal LoRA)
```

### Örnek:

```
LoRA_super:
  • Manchester_Win_Expert (Top #1)
  • Liverpool_Goal_Expert (Top #2)
  • Manchester_vs_Liverpool_Expert (Top #1)
  • Arsenal_Hype_Expert (Top #3)
  • Manchester_Goal_Expert (Top #4)

Toplam: 5 uzmanlık → %90 ölümsüz!

Base Ölüm Riski: %30
Gerçek Ölüm Riski: %30 × (1 - 0.90) = %3 ← Neredeyse ölmez!
```

### Uzmanlık Kaybı:

```
LoRA_declining:
  Eski: 5 uzmanlık → %90 ölümsüz
  Şimdi: 2 uzmanlık → %50 ölümsüz (3 uzmanlık kaybetti!)

Base Ölüm Riski: %30
Gerçek Ölüm Riski: %30 × (1 - 0.50) = %15

→ Ölüm riski arttı! (%3 → %15)
→ Yavaş yavaş normal LoRA seviyesine iniyor
```

---

## 🔄 Güncelleme Sıklığı

### Her 50 maçta:
1. Tüm LoRA'ların takım bazlı skorları hesaplanır
2. Her takım için Top 5 belirlenir
3. .pt dosyaları kaydedilir (ID bazlı)
4. .txt dosyaları güncellenir (senkronize!)
5. Ölümsüzlük seviyeleri güncellenir

### Her maç:
- Tahminler kaydedilir (win, goal, hype)
- Accuracy, MAE, hype doğruluk takip edilir

---

## 🎯 Kullanım

### Bir takım için uzmanları bul:

```bash
# Manchester United için tüm uzmanlar
cat takım_uzmanlıkları/Manchester_United/manchester_united_MASTER.txt

# Sadece win experts
cat takım_uzmanlıkları/Manchester_United/🎯_WIN_EXPERTS/🎯_win_experts_top5.txt

# Manchester vs Liverpool uzmanları
cat takım_uzmanlıkları/Manchester_United/🆚_VS_Liverpool/🆚_vs_liverpool_top5.txt
```

### Bir LoRA'nın tüm uzmanlıklarını bul:

```python
# LoRA_abc123 hangi takımlarda uzman?
# Her takım klasörüne bak, ID'si var mı?

Sonuç:
  • Manchester_United/🎯_WIN_EXPERTS/ ✅
  • Liverpool/⚽_GOAL_EXPERTS/ ✅
  • Manchester_United/🆚_VS_Liverpool/ ✅
  
Toplam: 3 uzmanlık → %70 ölümsüz!
```

---

## 💡 Özel Durumlar

### Çoklu Kopyalama:
- Bir LoRA birden fazla uzmanlıkta olabilir
- Aynı .pt dosyası birden fazla klasörde olabilir
- **HATA ÇIKAMAZ!** Copy işlemi güvenli

### Minimum Maç:
- Win/Goal/Hype: 20 maç
- VS: 5 maç (daha az eşleşme olur)

### Top 5 Güncellemesi:
- Yeni LoRA Top 5'e girerse → .pt dosyası eklenir
- Eski LoRA Top 5'ten düşerse → .pt dosyası kalır (arşiv)
- Txt dosyası her seferinde yeniden oluşturulur (senkronize)

---

## 🧬 Evrim Etkisi

Çoklu uzman LoRA'lar:
- Daha az enerji kaybeder
- Daha uzun yaşar
- Daha fazla çiftleşir
- Daha fazla bilgi aktarır

→ **En iyi genetik özelliklerin korunması garanti!**


