# 🎯 GELİŞMİŞ MEKANİKLER

## 5 Yeni Sistem

---

## 1️⃣ ELİT DİRENCİ (Zırh Sistemi)

### **Felsefe:**
```
❌ "Elit LoRA'lar ölmez"
✅ "Elit LoRA'lar zırh kazanır, ama %100 koruma YOK!"
```

### **Zırh Hesabı:**

```python
if fitness < 0.50:
  armor = 0%  # Zayıflar korumasız

if fitness = 0.50:
  armor = 0%  # Başlangıç

if fitness = 0.75:
  armor = 30%  # Orta elit

if fitness = 1.00:
  armor = 60%  # Max elit (ASLA %100 DEĞİL!)

Formula:
  armor = min((fitness - 0.50) × 1.2, 0.60)
```

### **Felakette:**

```python
# Normal ölüm şansı: %80
# Elit LoRA (fitness: 0.80):
  armor = 36%
  death_chance = 80% × (1 - 0.36) = 51.2%
  
  # %51.2 şans ölür!
  # Yani elit de ölebilir, ama şansı daha yüksek!
```

**Sonuç:** En kral LoRA bile kıyamette %40 ihtimalle ölür!

---

## 2️⃣ SAĞ KALAN SENDROMU (Survivor's Guilt)

### **Felsefe:**
```
"Hayatta kalmak bedelsiz değil!"
```

### **Ne Zaman Olur:**

```python
# Kara Veba:
kill_ratio = 80%
fitness = 0.85
armor = 42%

death_chance = 80% × (1 - 0.42) = 46.4%

if random() < 0.464:
  # Öldü
else:
  # ZIRH KORUDI!
  # Ama arkadaşları öldü...
  → SAĞ KALAN SENDROMU!
```

### **Bedeller:**

#### **1) Fiziksel (Fitness Düşer):**
```python
fitness_penalty = armor × 0.3

# Zırh %42 ise:
penalty = 0.42 × 0.3 = 12.6% fitness kaybı

old_fitness: 0.85
new_fitness: 0.85 × (1 - 0.126) = 0.743
```

#### **2) Zihinsel (Travma):**
```python
trauma_gain = armor × 0.5

# Zırh %42 ise:
trauma_severity = 0.42 × 0.5 = 0.21

# Travma ekle:
"Kara Veba'dan zırh ile kurtuldu (suçluluk: 0.21)"
```

#### **3) Kişilik Değişimi:**
```python
# Cesaret azalır:
risk_appetite: 0.80 → 0.56 (×0.7)

# Stres toleransı azalır:
stres_toleransı: 0.65 → 0.52 (×0.8)

# Dürtüsellik azalır (temkinli oldu):
dürtüsellik: 0.70 → 0.52 (×0.75)
```

**Sonuç:** Hayatta kaldı ama değişti! Artık eski hali değil!

---

## 3️⃣ KAN UYUŞMAZLIĞI (Anti-Inbreeding)

### **Sorun:**
```
En iyiler sürekli çiftleşir
  ↓
50 maç sonra herkes birbirine benzer
  ↓
Genetik darboğaz
  ↓
Sistem çeşitliliğini kaybeder
```

### **Çözüm:**

```python
# Çiftleşme öncesi:
similarity = cosine_similarity(lora1.params, lora2.params)

if similarity > 0.95:
  # ÇOK BENZERLER!
  
  if random() < 0.50:
    # İptal
    print("❌ Çiftleşme iptal (genetik benzerlik %95+)")
  
  else:
    # %100 MUTANT DOĞUR!
    mutant = spawn_random_lora()
    mutant.parents = [lora1.id, lora2.id]
    mutant.is_mutant = True
    
    print("👽 MUTANT DOĞDU! (Genetik çeşitlilik koruması)")
    print("   Ebeveynler çok benziyordu, doğa müdahale etti!")
```

**Mutant:**
- Tamamen rastgele parametreler
- Ama genetik olarak ebeveynlerin çocuğu sayılır
- Belki DAHI olur, belki UCUBE!
- Genetik havuzu taze tutar!

---

## 4️⃣ KIŞ UYKUSU (Hibernation)

### **Sorun:**
```
200 LoRA × Her biri GPU'da
  ↓
GPU Memory patlar!
```

### **Çözüm:**

```python
# Uyutma kriterleri:
if (population > 100 and 
    meta_attention < 2% and 
    0.40 < fitness < 0.70):
  
  # UYUT!
  save_to_disk(lora)
  remove_from_gpu(lora)
```

**Kim uyur?**
- Orta şeker LoRA'lar (0.40-0.70 fitness)
- Meta-LoRA az ağırlık veriyor (< %2)
- Nüfus > 100

**Kim uyumaz?**
- Çok iyi (> 0.70) → Aktif
- Çok kötü (< 0.40) → Ölecek zaten
- Yüksek attention (> %2) → Kullanılıyor

### **Uyanma:**

```python
# Meta-LoRA bir LoRA'yı çağırırsa:
if lora_id in hibernated:
  lora = wake_up(lora_id)
  load_to_gpu(lora)
  # Hemen kullan!
```

**Sonuç:**
- GPU'da sadece aktif LoRA'lar
- 200 LoRA olsa bile GPU patlamaz
- Uyuyanlar diske yazılır (SSD hızlı)

---

## 5️⃣ POZİTİF GERİ BESLEME FRENİ

### **Sorun:**
```
LoRA hata → Doğa öfkelenir → Veba → LoRA ölür
  ↓
Kalanlar travmadan hata yapar → Doğa daha öfkeli
  ↓
Yeni veba → Daha çok ölüm → Daha çok travma
  ↓
Sonsuz döngü → SİSTEM ÇÖKÜŞÜ!
```

### **Çözüm: 3 Katmanlı Fren**

#### **1) Soğuma Süresi (Cooldown):**
```python
# Her büyük olaydan (şiddet > 0.7) sonra:
cooldown = 20 maç

# Bu süre içinde:
new_major_event = BLOCKED!

# Örnek:
Maç #100: Kara Veba (şiddet: 0.95)
Maç #105: Doğa çok öfkeli ama...
  → "Soğuma süresi! (15 maç daha)"
Maç #121: Artık yeni olay olabilir
```

#### **2) Doygunluk (Saturation):**
```python
# Son 20 maçta:
if major_events_count >= 3:
  → "Doğa doygunluğa ulaştı!"
  → Yeni olay BLOCKED!

# Örnek:
Maç #200-220:
  Maç #202: Deprem (0.80)
  Maç #210: Nüfus Patlaması (0.85)
  Maç #218: Kaos Patlaması (0.70)
  
Maç #221: Doğa çok öfkeli ama...
  → "3 olay 20 maçta! Doğa doydu, dinleniyor"
```

#### **3) Doğa Enerjisi:**
```python
# Son 50 maçtaki toplam şiddet:
total_severity = sum(event.severity for event in last_50_matches)

# Her 1.0 şiddet = %20 enerji kaybı:
energy = 1.0 - (total_severity × 0.2)

# Enerji < 0.3 ise:
  → Doğa çok yorgun, olay olasılığı %70 azalır

# Örnek:
Son 50 maçta toplam şiddet: 4.5
  energy = 1.0 - (4.5 × 0.2) = 0.10 (%10 enerji)
  
  → Doğa neredeyse tükenmiş!
  → Yeni olay neredeyse imkansız (dinlenmeli)
```

### **Sonuç:**

```
Doğa öfkelenir → Olay → Soğur → Dinlenir → Tekrar enerjilenir

❌ Sonsuz öfke döngüsü
✅ Doğal dinlenme ve toparlanma
```

---

## 🎮 TÜM MEKANİKLER BİRLİKTE

### **Senaryo: Kara Veba + Tüm Mekanikler**

```
Maç #234: Sağlık 0.18 → KARA VEBA TETİKLENDİ!

1️⃣ FREN KONTROLÜ:
   Son büyük olay: Maç #180 (54 maç önce)
   → ✅ Cooldown geçti
   
   Son 20 maçta olay: 1 adet
   → ✅ Doygunluk yok
   
   Doğa enerjisi: 0.65
   → ✅ Yeterli enerji
   
   → KARA VEBA İZİN VERİLDİ!

2️⃣ ELİT DİRENCİ:
   100 LoRA var:
   
   LoRA_Gen5_x9a2 (fitness: 0.85):
     armor = 42%
     death_chance = 80% × (1 - 0.42) = 46.4%
     → Şanslı! Hayatta kaldı! (zırh korudu)
   
   LoRA_Gen3_c8e1 (fitness: 0.92):
     armor = 50.4%
     death_chance = 80% × (1 - 0.504) = 39.7%
     → Şanslı! Hayatta kaldı!
   
   LoRA_Gen2_a4f3 (fitness: 0.45):
     armor = 0%
     death_chance = 80%
     → Öldü (zırh yok)
   
   ... 18 LoRA hayatta kaldı (80 öldü)

3️⃣ SAĞ KALAN SENDROMU:
   LoRA_Gen5_x9a2 (zırh ile kurtuldu):
     Fitness: 0.85 → 0.743 (-12.6%)
     Travma: +0.21 (suçluluk)
     Mizaç:
       risk_appetite: 0.75 → 0.52
       stres_toleransı: 0.68 → 0.54
       dürtüsellik: 0.72 → 0.54
     
     → "Hayatta kaldım ama... arkadaşlarım öldü 😢"

4️⃣ FREN AKTİVASYONU:
   Kara Veba kaydedildi (şiddet: 0.95)
   → Soğuma: 20 maç
   → Doygunluk: 1/3 (henüz ok)
   → Enerji: 0.65 → 0.46 (-%19)

5️⃣ SONUÇ:
   18/100 LoRA hayatta kaldı
   2 LoRA sağ kalan sendromu yaşıyor
   Doğa 20 maç dinlenecek
   Sistem toparlanmaya başlayacak
```

---

## 🔄 HİBERNATION SENARYOSU

```
Maç #500: 145 LoRA (çok fazla!)

1️⃣ AKTIF KONTROL:
   Meta-LoRA attention hesaplandı:
   
   LoRA_Gen8_a9x3: %18 ağırlık → AKTİF
   LoRA_Gen5_c4f2: %0.8 ağırlık, fitness: 0.55 → UYUT!
   LoRA_Gen7_m2k1: %0.5 ağırlık, fitness: 0.48 → UYUT!
   ...

2️⃣ UYUTMA:
   35 LoRA uyutuldu
   → Diske kaydedildi
   → RAM'den silindi
   → GPU'da yer açıldı

3️⃣ AKTİF NÜFUS:
   110 LoRA aktif (RAM'de)
   35 LoRA uyuyor (Diskte)
   Toplam: 145 LoRA

4️⃣ UYANDIRMA (Maç #520):
   Meta-LoRA: "Bu maç için LoRA_Gen5_c4f2 gerekli!"
   → Diskten yükle
   → GPU'ya al
   → Tahmin yap
   → Tekrar uyut (veya aktif tut)
```

**Sonuç:** 500 LoRA olsa bile GPU patlamaz!

---

## 🧬 MUTANT DOĞUM SENARYOSU

```
Maç #87:

1️⃣ ÇİFTLEŞME DENEMESİ:
   Anne: LoRA_Gen5_x9a2 (Derbi Uzmanı, fitness: 0.82)
   Baba: LoRA_Gen5_c8e1 (Derbi Uzmanı, fitness: 0.78)

2️⃣ GENETİK KONTROL:
   similarity = cosine_similarity(params)
   = 0.97 (%97 benzer!)
   
   → UYARI: Çok benzerler!

3️⃣ KARAR:
   random() = 0.62 > 0.50
   → MUTANT DOĞUR!

4️⃣ MUTANT DOĞUM:
   LoRA_MUTANT_z7k4 doğdu!
   
   Özellikler:
   - Tamamen rastgele parametreler
   - Ebeveynler: x9a2 + c8e1 (genetik olarak)
   - is_mutant = True
   - Belki dahi, belki ucube!
   
   Beklenti:
   - %30 şans: Süper derbi uzmanı (ikisinden daha iyi!)
   - %40 şans: Orta performans
   - %30 şans: Kötü (ama genetik çeşitlilik sağladı)

5️⃣ SONUÇ:
   Genetik havuz taze kaldı!
   Herkes aynı olmadı!
```

---

## 🌊 DOĞA ENERJİSİ + FREN

### **Enerji Grafiği:**

```
Enerji
  |
1.0|●
    |  ●
0.8|    ●
    |      ●
0.6|        ●    ⚡(Deprem)
    |          ●●
0.4|            ●    ☠️(Veba)
    |              ●●●
0.2|                 ●
    |                  ●  (Dinleniyor...)
0.0|____________________________
    0  20  40  60  80  100  120  (Maç)
    
Maç 0-40: Enerji tam (olaysız)
Maç 60: Deprem (şiddet: 0.8) → Enerji: 0.84
Maç 80: Veba (şiddet: 0.95) → Enerji: 0.65
Maç 100: Nüfus Patlaması (0.85) → Enerji: 0.48
Maç 105: Doğa olay yapmak istiyor ama...
  → "Enerji çok düşük! (%48)"
  → BLOCKED!
Maç 120: Enerji: 0.30 → Hâlâ dinleniyor
Maç 150: Enerji: 0.65 → Toparlandı, yeni olay olabilir
```

---

## 📊 KARŞILAŞTIRMA

| Özellik | Eski Sistem | Yeni Sistem (5 Mekanik) |
|---------|-------------|-------------------------|
| **Elit Koruması** | %100 ölmez | Max %60 zırh |
| **Felaket Sonrası** | Hiçbir etki | Sağ kalan sendromu |
| **Genetik** | Darboğaz riski | Mutant doğum koruması |
| **GPU Kullanımı** | 100 LoRA = patlama | 500 LoRA = hibernation |
| **Doğa Döngüsü** | Sonsuz öfke riski | Fren + cooldown |

---

## 🎯 SENARYOLAR

### **1) Elit LoRA'nın Sonu:**

```
LoRA_Gen10_KRAL (fitness: 0.95):
  - En güçlü LoRA
  - Armor: %54
  
Kara Veba:
  - %80 ölüm
  - Zırh ile: %37 ölüm şansı
  
Sonuç:
  - %63 hayatta kalır
  - %37 ölür → Elit de ölüyor!
  
Eğer hayatta kalırsa:
  - Fitness: 0.95 → 0.79 (-%16)
  - Travma: Ağır (0.40)
  - Mizaç: Korkak hale gelir
  
  → Artık eski KRAL değil!
```

### **2) Genetik Darboğaz Önlendi:**

```
Herkes derbi uzmanı olmuş (benzerlik %98)
  ↓
Yeni çiftleşme:
  → MUTANT DOĞDU!
  → Tamamen farklı pattern keşfetti
  ↓
Sistem çeşitlilik kazandı!
```

### **3) GPU Patlaması Önlendi:**

```
Nüfus: 250 LoRA (teorik)
Aktif: 95 LoRA (GPU'da)
Uyuyan: 155 LoRA (Diskte)

GPU Memory: %65 (güvenli)
```

### **4) Doğa Tükenmedi:**

```
3 olay üst üste → Doğa yorgun
  ↓
20 maç dinlenme
  ↓
Enerji toplandı
  ↓
Yeni döngü başladı

❌ Sonsuz öfke
✅ Doğal ritim
```

---

## ✅ SONUÇ

Bu 5 mekanik:

✅ **Gerçekçilik:** Elit de ölür, ama daha zor  
✅ **Denge:** Doğa tükenmez, dinlenir  
✅ **Çeşitlilik:** Genetik darboğaz önlenir  
✅ **Performans:** GPU patlamaz  
✅ **Psikoloji:** Hayatta kalmanın bedeli var  

**Sistem artık tam bir ekosistem!** 🌍

---

**Son Güncelleme:** Aralık 2025  
**Versiyon:** 2.1 - Gelişmiş Mekanikler




