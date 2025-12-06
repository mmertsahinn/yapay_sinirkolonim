# 😴 HİBERNATION SİSTEMİ - TAM RAPOR

**Tarih:** 2025-12-04  
**Durum:** ✅ **KRİTİK DÜZELTMELER YAPILDI!**

---

## ⚠️ **KRİTİK KURALLAR:**

### 1. **%20 LİMİT! ASLA AŞILMAZ!**

```
Toplam Popülasyon = Aktif LoRA + Uyuyan LoRA

Maksimum Uyuyan = Toplam × 20%

Örnek:
  • 200 LoRA toplam → Max 40 uyuyan
  • 100 LoRA toplam → Max 20 uyuyan
  • 50 LoRA toplam → Max 10 uyuyan
```

**Amaç:** Evrim ve gelişim! Yük taşımak değil!  
**Kural:** Toplumun %80'i HEP AKTİF olmalı!

---

### 2. **UYUMA KRİTERLERİ:**

Bir LoRA uyutulabilir EĞER:
- ✅ Popülasyon > 100
- ✅ Meta-LoRA ağırlığı < %2 (kullanılmıyor)
- ✅ Fitness 0.40-0.70 arası (orta seviye)
- ✅ **%20 limiti aşılmamış**

**Uyumaz:**
- ❌ Çok iyi LoRA'lar (fitness > 0.70)
- ❌ Çok kötü LoRA'lar (fitness < 0.40)
- ❌ Aktif kullanılanlar (ağırlık > %2)
- ❌ Limit dolduysa

---

### 3. **UYUYANLAR YAŞAYAN EXCEL'DE!**

**Dosya:** `YASAYAN_LORALAR_CANLI.xlsx`

**Şimdi İçerir:**
- ✅ Aktif LoRA'lar (normal verilerle)
- ✅ **Uyuyan LoRA'lar (😴 UYUYOR durumunda)**

**Uyuyanlar İçin:**
- Durum: `😴 UYUYOR`
- TES: `-` (hesaplanamaz)
- Fizik: `-` (hesaplanamaz)
- Fitness: Son bilinen değer
- Etiket: `😴 UYUYAN`

---

## 🔍 **DEBUG SİSTEMİ:**

### Her 10 Maçta:

```
😴 HİBERNATION DEBUG (Maç #50):
   • Toplam Popülasyon: 150 (Aktif: 125, Uyuyan: 25)
   • Şu An Uyuma Oranı: 16.7%
   • Maksimum İzin: 20.0% (30 LoRA)
   • Kalan Slot: 5 LoRA
```

**Durumlar:**
- `⏸️  Uyutma yapılmıyor` → Nüfus ≤ 100
- `🛑 LİMİT AŞILDI!` → %20 doldu
- `⚠️  UYARI: Limit yakın!` → %18+ yaklaşıyor

---

## 📊 **UYUMA MEKANİĞİ:**

### Adım 1: Adayları Bul
- Tüm popülasyonu tarar
- Uyutulabilir LoRA'ları listeler

### Adım 2: Fitness'a Göre Sırala
- En düşük fitness önce uyur
- Orta seviye LoRA'lar seçilir

### Adım 3: Limit'e Kadar Uyut
- %20 limitine kadar uyut
- Limit dolunca DURDUR!

### Adım 4: Disk'e Kaydet
- `hibernated_loras/LoRA_NAME.pt`
- RAM'den sil
- GPU'dan çıkar

---

## 📁 **DOSYA YAPISI:**

```
hibernated_loras/
├── LoRA_Gen5_a3b2.pt
├── LoRA_Gen4_c8d1.pt
└── ... (uyuyanlar)

evolution_logs/
├── YASAYAN_LORALAR_CANLI.xlsx  # Aktif + Uyuyan!
└── ... (diğer loglar)
```

---

## 🔄 **UYANDIRMA:**

Uyuyanlar şu durumlarda uyandırılır:
1. Popülasyon çok azaldıysa
2. Belirli bir LoRA'ya ihtiyaç varsa
3. Intelligent wake-up sistemi devreye girdiyse

**Uyandırıldığında:**
- RAM'e yüklenir
- Aktif popülasyona eklenir
- Hibernated listesinden çıkarılır

---

## ✅ **YAPILAN DÜZELTMELER:**

1. ✅ **%20 Limit Eklendi** - Artık asla aşılmaz!
2. ✅ **Debug Logları** - Her 10 maçta durum gösterilir
3. ✅ **Uyuyanlar Excel'de** - Artık görünüyorlar!
4. ✅ **Limit Kontrolü** - Durdurma mekanizması çalışıyor
5. ✅ **Fitness Sıralama** - Düşük fitness önce uyur

---

## 🎯 **ÖZET:**

**Öncesi:**
- ❌ Sınırsız uyutma
- ❌ Uyuyanlar görünmüyordu
- ❌ Kontrol yoktu

**Sonrası:**
- ✅ Maksimum %20 uyutma
- ✅ Uyuyanlar Excel'de
- ✅ Sürekli kontrol ve debug
- ✅ Evrim ve gelişim odaklı!

**AMAÇ:** Toplumun %80'i aktif, %20'si uyuyan!  
**KURAL:** Asla %20'den fazla uyumasın!  
**FELSEFE:** Evrim ve gelişim, yük taşımak değil! 🚀

