# 🐛 BİLİNEN SORUNLAR VE KUSURLAR

**Son Güncelleme:** 2025-12-04

---

## ⚠️ YÜKSEK ÖNCELİK

### 1. Fisher Information Matrix Hesaplaması Çalışmıyor

**Sorun:**
- K-FAC Fisher hesaplaması hep default değere (`1e-10`) düşüyor
- Tüm LoRA'lar aynı Fisher değerini alıyor
- Lazarus Lambda herkeste aynı (`0.482`)

**Etkilenen Sistemler:**
- ✅ Lazarus Potential (Diriltme)
- ✅ Nature Trigger System
- ✅ TES Scoreboard (indirect)

**Gözlem:**
```
Her LoRA için:
• Determinant: 1.00e-10 (hep default!)
• Fisher term: 0.619 (hep aynı!)
• Entropy: 0.5000 (hep aynı!)
• Lazarus Λ: 0.482 (hep aynı!)
```

**Sebep:**
- K-FAC Fisher matrisi hesaplaması karmaşık
- Hata oluşunca default değere düşüyor
- `kfac_fisher.compute_fisher_kfac(lora)` çalışmıyor

**Etki:**
- ❌ Çeşitlilik yok - hepsi aynı potansiyele sahip
- ✅ Sistem çalışıyor - ama ayrım yapamıyor
- ✅ Diriltme çalışıyor - ama öncelik yok

**Çözüm Önerileri:**
1. **Basitleştirilmiş Fisher:** Gradient magnitude kullan
2. **Alternatif Metrik:** TES skorları + fitness geçmişi
3. **Parametre Çeşitliliği:** Doğrudan parametre std hesapla
4. **K-FAC'i Düzelt:** Hesaplama hatasını bul ve düzelt

**Geçici Çözüm:**
- Fisher yerine TES skorları kullanılabilir
- Fitness geçmişi + yaş diriltme için yeterli
- Sistem Fisher olmadan da çalışıyor

**Dosya:**
- `lora_system/lazarus_potential.py`
- `lora_system/kfac_fisher.py`

---

## 📋 ORTA ÖNCELİK

### 2. Log Dosyaları Güncellenmiyor (Tespit Aşamasında)

**Sorun:**
- Population History maç #0'da kalıyor
- Dynamic Relocation boş

**Durum:** Araştırılıyor (debug eklendi)

---

## ℹ️ DÜŞÜK ÖNCELİK

### 3. Unicode/Emoji Sorunları (Çözüldü)

**Sorun:** Windows terminal emoji desteklemiyor

**Çözüm:** ASCII karakterlere çevrildi

**Durum:** ✅ Çözüldü

---

## 📊 İSTATİSTİKLER

- **Toplam Sorun:** 2 aktif
- **Kritik:** 1
- **Çözüldü:** 1

---

## 🔗 İLGİLİ DOSYALAR

- `lora_system/lazarus_potential.py` - Fisher hesaplama
- `lora_system/kfac_fisher.py` - K-FAC implementasyonu
- `lora_system/resurrection_system_v2.py` - Diriltme sistemi
- `run_evolutionary_learning.py` - Ana loop

---

## 📝 NOTLAR

Fisher sorunu kritik değil çünkü:
1. Sistem başka metriklerle çalışıyor
2. Diriltme için alternatifler var (TES, fitness, yaş)
3. Genetik çeşitlilik zaten koloni mantığıyla yönetiliyor

**İleride yapılacak:** Fisher hesaplamasını basitleştir veya alternatif kullan.

