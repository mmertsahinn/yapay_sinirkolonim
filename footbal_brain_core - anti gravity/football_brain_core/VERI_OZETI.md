# 📊 Veritabanı Veri Özeti

## 🎯 Genel Bakış

Veritabanınızda şu anda **3,926 maç** verisi var! İşte detaylı özet:

---

## 📋 Tablolar ve İçerikleri

### 1. **LİGLER (Leagues)** - 7 Lig
- Premier League (İngiltere)
- La Liga (İspanya)
- Serie A (İtalya)
- Bundesliga (Almanya)
- Ligue 1 (Fransa)
- Liga Portugal (Portekiz)
- Süper Lig (Türkiye)

**Her ligde saklanan bilgiler:**
- Lig adı
- Ülke
- Lig kodu
- Oluşturulma tarihi

---

### 2. **TAKIMLAR (Teams)** - 156 Takım
Her ligden takımlar saklanıyor.

**Her takımda saklanan bilgiler:**
- Takım adı
- Hangi ligde oynadığı (league_id)
- Takım kodu
- Oluşturulma tarihi

---

### 3. **MAÇLAR (Matches)** - 3,926 Maç ⭐
**Bu en önemli tablo!**

**Her maçta saklanan bilgiler:**
- **match_id**: API'den gelen benzersiz maç ID'si
- **league_id**: Hangi ligde oynandığı
- **home_team_id**: Ev sahibi takım
- **away_team_id**: Deplasman takımı
- **match_date**: Maç tarihi (2021-08-13 - 2024-06-02 arası)
- **home_score**: Ev sahibi takımın golleri
- **away_score**: Deplasman takımın golleri
- **status**: Maç durumu (tamamlandı, ertelendi, vs.)
- **created_at**: Veritabanına eklenme tarihi
- **updated_at**: Son güncelleme tarihi

**Örnek maçlar:**
- Brentford vs Arsenal (2-?)
- Manchester United vs Leeds (5-1)
- Watford vs Aston Villa (3-2)
- vs.

---

### 4. **İSTATİSTİKLER (Stats)** - 0 Kayıt
Şu anda istatistik verisi yok, ama sistem hazır.

**İstatistiklerde saklanacak bilgiler:**
- Hangi maça ait
- Hangi takıma ait
- İstatistik türü (gol, pas, top kontrolü, şut, vs.)
- İstatistik değeri (sayısal)

---

### 5. **MARKETLER (Markets)** - 0 Kayıt
Bahis piyasaları için hazır.

**Marketlerde saklanacak bilgiler:**
- Market adı (örn: "Match Result", "BTTS", "Over/Under 2.5")
- Açıklama

---

### 6. **TAHMİNLER (Predictions)** - 0 Kayıt
Model tahminleri için hazır.

**Tahminlerde saklanacak bilgiler:**
- Hangi maça ait
- Hangi market için
- Tahmin edilen sonuç (örn: "1", "X", "2", "Yes", "No")
- Olasılık değeri (p_hat)
- Hangi model versiyonu ile yapıldı
- Tahmin zamanı

---

### 7. **SONUÇLAR (Results)** - 0 Kayıt
Gerçek sonuçlar için hazır.

**Sonuçlarda saklanacak bilgiler:**
- Hangi maça ait
- Hangi market için
- Gerçek sonuç (tahminle karşılaştırma için)

---

### 8. **MODEL VERSİYONLARI (Model Versions)** - 0 Kayıt
Farklı model versiyonlarını takip etmek için.

**Model versiyonlarında saklanacak bilgiler:**
- Versiyon numarası/adı
- Açıklama
- Aktif mi pasif mi

---

### 9. **AÇIKLAMALAR (Explanations)** - 0 Kayıt
LLM (GPT/Grok) tarafından üretilen açıklamalar için.

**Açıklamalarda saklanacak bilgiler:**
- Hangi maça ait
- Hangi market için
- LLM çıktısı (metin açıklama)
- Özet istatistikler (JSON)

---

### 10. **DENEYLER (Experiments)** - 0 Kayıt
Model deneylerini takip etmek için.

**Deneylerde saklanacak bilgiler:**
- Deney ID'si
- Konfigürasyon (JSON)
- Dönem başlangıç/bitiş tarihleri
- Metrikler (JSON)

---

## 📅 Veri Tarih Aralığı

**Maçlar:** 2021-08-13 ile 2024-06-02 arası
- Yaklaşık **3 sezon** verisi var
- 2021-2022, 2022-2023, 2023-2024 sezonları

---

## 🔍 API'den Çekilen Veriler

### API-FOOTBALL'dan Çekilenler:
1. **Lig Bilgileri**
   - Lig adı, ülke, kod

2. **Takım Bilgileri**
   - Takım adı, kod, hangi ligde

3. **Maç Fikstürleri**
   - Maç tarihleri
   - Ev sahibi/deplasman takımları
   - Skorlar (oynanmış maçlar için)

4. **Maç İstatistikleri** (henüz çekilmedi)
   - Gol sayıları
   - Pas yüzdeleri
   - Top kontrolü
   - Şut sayıları
   - vs.

5. **Bahis Piyasaları** (henüz çekilmedi)
   - Match Result (1-X-2)
   - BTTS (Both Teams To Score)
   - Over/Under
   - vs.

---

## 📊 Özet İstatistikler

| Tablo | Kayıt Sayısı | Durum |
|-------|-------------|-------|
| Ligler | 7 | ✅ Dolu |
| Takımlar | 156 | ✅ Dolu |
| Maçlar | 3,926 | ✅ Dolu |
| İstatistikler | 0 | ⏳ Hazır |
| Marketler | 0 | ⏳ Hazır |
| Tahminler | 0 | ⏳ Hazır |
| Sonuçlar | 0 | ⏳ Hazır |
| Model Versiyonları | 0 | ⏳ Hazır |
| Açıklamalar | 0 | ⏳ Hazır |
| Deneyler | 0 | ⏳ Hazır |

---

## 🎯 Sonraki Adımlar

1. **İstatistikleri çek** - Maç istatistiklerini API'den çek
2. **Marketleri tanımla** - Bahis piyasalarını oluştur
3. **Model eğit** - Maç verileriyle model eğit
4. **Tahmin yap** - Yeni maçlar için tahmin üret
5. **Açıklama üret** - LLM ile tahmin açıklamaları oluştur

---

## 💡 Notlar

- Veritabanı SQLite formatında (`football_brain.db`)
- Tüm veriler API-FOOTBALL'dan çekiliyor
- Sistem modüler yapıda, yeni veriler kolayca eklenebilir
- Model tahminleri ve açıklamalar henüz üretilmedi

---

**Son Güncelleme:** 2025-11-29






