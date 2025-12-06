# Odds Eşleştirme İyileştirmeleri

## 🎯 Hedef: %65 → %75-80 Eşleşme Oranı

### Yapılan İyileştirmeler:

#### 1. **Daha Esnek Fuzzy Matching**
- ✅ Minimum eşik: 0.75 → **0.65** (daha fazla eşleşme)
- ✅ Takım isimlerine **%70 ağırlık** (öncelik takım isimlerinde)
- ✅ Özel durumlar:
  - Takım isimleri %75+ benzer + tarih aynı → Kabul et
  - Takım isimleri %85+ benzer → Kabul et (diğer faktörler önemli değil)

#### 2. **Genişletilmiş Tarih Toleransı**
- ✅ ±1 gün → **±3 gün** (daha geniş aralık)
- ✅ Tarih skoru:
  - Aynı gün: 1.0
  - 1 gün fark: 0.9
  - 2 gün fark: 0.8
  - 3 gün fark: 0.7

#### 3. **Esnek Lig Eşleşmesi**
- ✅ Default lig skoru: 0.5 → **0.6**
- ✅ Minimum lig skoru: **0.5** (lig farkı çok kritik değil)
- ✅ Lig benzerliği %50+ → Kabul et

#### 4. **Gelişmiş Takım İsmi Normalizasyonu**
- ✅ FC, AFC, United, City gibi varyasyonlar
- ✅ Özel karakter temizleme
- ✅ Fuzzy matching için agresif normalizasyon
- ✅ Ters sıra kontrolü (home-away vs away-home)

#### 5. **Çoklu Alternatif Key Formatları**
- ✅ `league_date_home_away`
- ✅ `league_date_away_home` (ters sıra)
- ✅ `date_home_away` (lig olmadan)
- ✅ League ID bazlı formatlar

#### 6. **Optimizasyon**
- ✅ Tarih bazlı indexleme (±3 gün)
- ✅ Sadece ilgili tarihlerdeki odds'ları kontrol
- ✅ Candidate'ları unique yap

## 📊 Beklenen Sonuçlar

### Önceki Durum:
- 33 bin odds → 15 bin eşleşme (%45)

### Şu Anki Durum:
- 33 bin odds → ~21 bin eşleşme (%65) ✅

### Hedef:
- 33 bin odds → ~25-28 bin eşleşme (%75-80) 🎯

## 🔍 İyileştirme Stratejisi

### 1. Minimum Eşik Düşürüldü
```python
# Önceki: 0.75
# Şimdi: 0.65 (daha esnek)
# Özel durumlar için daha da esnek kurallar
```

### 2. Tarih Toleransı Artırıldı
```python
# Önceki: ±1 gün
# Şimdi: ±3 gün
```

### 3. Takım İsimlerine Öncelik
```python
# Önceki: Takım %60, Lig %20, Tarih %20
# Şimdi: Takım %70, Lig %15, Tarih %15
```

### 4. Esnek Lig Eşleşmesi
```python
# Önceki: Lig çok kritik
# Şimdi: Lig farkı çok önemli değil (minimum 0.5)
```

## ⚠️ Dikkat Edilmesi Gerekenler

1. **Çok Düşük Eşleşmeler**: 0.65 eşik ile bazı yanlış eşleşmeler olabilir
   - Çözüm: Güven skoruna göre filtreleme

2. **Geniş Tarih Aralığı**: ±3 gün ile farklı maçlar eşleşebilir
   - Çözüm: Takım ismi benzerliği kontrolü ile önlenir

3. **Lig Farkı**: Farklı liglerden maçlar eşleşebilir
   - Çözüm: Takım isimlerine ağırlık verildi, lig skoru minimum 0.5

## 🚀 Sonraki Adımlar

Eğer %65'ten daha fazla eşleşme istiyorsanız:

1. **Minimum eşiği daha da düşürün** (0.60)
2. **Tarih toleransını artırın** (±5 gün)
3. **Kısmi takım ismi eşleşmesi** ekleyin
4. **Manuel eşleştirme** için log dosyası oluşturun

## 📈 Test Sonuçları

Script çalıştığında göreceğiniz:
- ✅ Detaylı eşleşme istatistikleri
- ✅ Kullanılmayan odds sayısı
- ✅ Odds kullanım oranı
- ✅ Eşleşme güven skorları

**Şu anki durum: %65 eşleşme oranı - İyi! ✅**

