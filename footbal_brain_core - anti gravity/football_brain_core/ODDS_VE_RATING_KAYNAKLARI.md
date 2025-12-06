# 📊 ODDS VE OYUNCU RATING KAYNAKLARI - DETAYLI AÇIKLAMA

## 🎯 NOTEBOOK 1: ODDS VE RATING KAYNAKLARI

### 📍 **ODDS (BOOKKEEPER DATA) NEREDEN GELİYOR?**

#### **Kaynak:**
```python
database = path + 'database.sqlite'
conn = sqlite3.connect(database)
match_data = pd.read_sql("SELECT * FROM Match;", conn)
```

**Kaggle "European Soccer Database" Dataset:**
- 📅 **Tarih Aralığı:** 2008-2016 sezonları
- 🏆 **Ligler:** Premier League, La Liga, Serie A, Bundesliga, Ligue 1
- 📊 **Toplam:** ~25,000 maç
- ⚠️ **GÜNCEL DEĞİL!** (2016'dan sonra güncellenmemiş)

#### **Bookkeeper'lar:**
```python
bk_cols = ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']
```

**Açıklamaları:**
- **B365:** Bet365 (büyük bahis şirketi)
- **BW:** Bet&Win
- **IW:** Interwetten
- **LB:** Ladbrokes
- **PS:** Pinnacle Sports
- **WH:** William Hill
- **SJ:** Sportingbet
- **VC:** VC Bet
- **GB:** Gamebookers
- **BS:** Betsson

#### **Nasıl Çekiliyor?**
```python
# Match tablosunda her maç için bookkeeper kolonları var:
# Örnek: B365H, B365D, B365A (Home, Draw, Away odds)
# Örnek: WHH, WHD, WHA (William Hill odds)
```

**Veri Formatı:**
- Odds decimal format'ta (örn: 2.50 = %40 probability)
- Her bookkeeper için 3 odds: Win, Draw, Defeat
- Maç öncesi sabit odds (maç sırasında değişmiyor)

#### **Güncellik Durumu:**
- ❌ **GÜNCEL DEĞİL!** (2016'dan sonra yok)
- ❌ **Statik veri** (Kaggle dataset'i artık güncellenmiyor)
- ⚠️ **Sadece geçmiş maçlar için kullanılabilir**

---

### 📍 **FIFA PLAYER RATING NEREDEN GELİYOR?**

#### **Kaynak:**
```python
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
```

**Kaggle "European Soccer Database" Dataset:**
- 📅 **Tarih Aralığı:** 2008-2016 sezonları
- 👤 **Oyuncu Sayısı:** ~11,000 oyuncu
- 📊 **Rating Güncellemeleri:** Her sezon başında (FIFA oyunu güncellemeleri)
- ⚠️ **GÜNCEL DEĞİL!** (2016'dan sonra yok)

#### **Nasıl Çekiliyor?**
```python
def get_fifa_stats(match, player_stats):
    # Her maç için 22 oyuncu ID'si var:
    # home_player_1, home_player_2, ..., home_player_11
    # away_player_1, away_player_2, ..., away_player_11
    
    # Her oyuncu için:
    # 1. Player ID'yi al
    # 2. Player_Attributes tablosunda o oyuncuyu bul
    # 3. Maç tarihinden ÖNCEKİ en son rating'i kullan
    # 4. overall_rating değerini al (0-100 arası)
```

**Veri Formatı:**
- `overall_rating`: 0-100 arası genel oyuncu rating'i
- `date`: Rating'in güncellendiği tarih
- Her oyuncu için birden fazla tarihli rating var (sezon güncellemeleri)

**Örnek:**
```
Oyuncu: Cristiano Ronaldo
- 2008-09-01: overall_rating = 87
- 2009-09-01: overall_rating = 89
- 2010-09-01: overall_rating = 91
```

**Maç Tarihi:** 2010-10-15
**Kullanılan Rating:** 91 (maç tarihinden önceki en son rating)

#### **Güncellik Durumu:**
- ❌ **GÜNCEL DEĞİL!** (2016'dan sonra yok)
- ❌ **Statik veri** (Kaggle dataset'i artık güncellenmiyor)
- ⚠️ **Sadece geçmiş maçlar için kullanılabilir**

---

## 🔄 GÜNCEL VERİ KAYNAKLARI

### **ODDS İÇİN GÜNCEL KAYNAKLAR:**

#### 1. **API-FOOTBALL** (Bizim kullandığımız)
- ✅ **Güncel:** Her gün güncellenir
- ✅ **Canlı odds:** Maç öncesi ve maç sırasında
- ✅ **Çoklu bookkeeper:** 10+ farklı bahis şirketi
- ⚠️ **Ücretli:** Pro plan gerekli (odds için)

#### 2. **The Odds API**
- ✅ **Güncel:** Real-time odds
- ✅ **Ücretsiz tier:** Sınırlı (500 request/ay)
- ✅ **Çoklu bookkeeper:** 10+ bahis şirketi
- 🔗 **Website:** https://the-odds-api.com/

#### 3. **Betfair API**
- ✅ **Güncel:** Real-time odds
- ✅ **Exchange odds:** Kullanıcılar arası bahis
- ⚠️ **Karmaşık:** API kurulumu zor

#### 4. **Web Scraping** (Yasal olmayabilir)
- ⚠️ **Riskli:** Bahis sitelerinin ToS'unu ihlal edebilir
- ⚠️ **Yavaş:** Rate limiting var
- ⚠️ **Kırılgan:** Site yapısı değişebilir

---

### **FIFA PLAYER RATING İÇİN GÜNCEL KAYNAKLAR:**

#### 1. **FIFA/EA Sports API** (Resmi)
- ❌ **Kapalı:** Public API yok
- ❌ **Sadece oyun içi:** FIFA oyunu için

#### 2. **Futhead / SoFIFA** (Web Scraping)
- ✅ **Güncel:** Her sezon güncellenir
- ✅ **Ücretsiz:** Web sitesinden çekilebilir
- ⚠️ **Scraping gerekli:** API yok

#### 3. **Transfermarkt**
- ✅ **Güncel:** Oyuncu değerleri ve rating'leri
- ✅ **Ücretsiz:** Web sitesinden çekilebilir
- ⚠️ **Scraping gerekli:** API yok

#### 4. **WhoScored / Opta**
- ✅ **Güncel:** Performans rating'leri
- ✅ **Profesyonel:** Spor analiz şirketleri
- ⚠️ **Ücretli:** API erişimi pahalı

#### 5. **API-FOOTBALL** (Bizim kullandığımız)
- ✅ **Güncel:** Oyuncu istatistikleri
- ✅ **Performans metrikleri:** Goals, assists, rating'ler
- ⚠️ **FIFA rating yok:** Sadece performans istatistikleri

---

## 📊 NOTEBOOK'LARDAKİ VERİLERİN DURUMU

### **ODDS:**
- ❌ **Güncel değil** (2016'dan sonra yok)
- ❌ **Statik veri** (Kaggle dataset'i artık güncellenmiyor)
- ✅ **Geçmiş maçlar için kullanılabilir** (2008-2016)
- ⚠️ **Yeni maçlar için kullanılamaz**

### **FIFA RATING:**
- ❌ **Güncel değil** (2016'dan sonra yok)
- ❌ **Statik veri** (Kaggle dataset'i artık güncellenmiyor)
- ✅ **Geçmiş maçlar için kullanılabilir** (2008-2016)
- ⚠️ **Yeni maçlar için kullanılamaz**

---

## 🎯 BİZİM SİSTEM İÇİN ÖNERİLER

### **ODDS İÇİN:**
1. **API-FOOTBALL** (Önerilen)
   - Pro plan al (odds için gerekli)
   - Her gün güncel odds çek
   - 10+ bookkeeper'dan odds al

2. **The Odds API** (Alternatif)
   - Ücretsiz tier ile başla
   - 500 request/ay limit var
   - Gerekirse ücretli plana geç

### **OYUNCU RATING İÇİN:**
1. **API-FOOTBALL Performans İstatistikleri** (Önerilen)
   - Goals, assists, rating'ler
   - Güncel performans metrikleri
   - FIFA rating yerine performans rating'i kullan

2. **SoFIFA Web Scraping** (Alternatif)
   - FIFA rating'leri çek
   - Her sezon güncelle
   - Legal risk var (ToS kontrol et)

3. **Transfermarkt** (Alternatif)
   - Oyuncu değerleri
   - Market value = güç göstergesi
   - Web scraping gerekli

---

## ✅ SONUÇ

**Notebook'lardaki veriler:**
- ❌ **Güncel değil** (2016'dan sonra yok)
- ✅ **Sadece geçmiş maçlar için kullanılabilir**
- ⚠️ **Yeni maçlar için kullanılamaz**

**Bizim sistem için:**
- ✅ **API-FOOTBALL** kullanıyoruz (güncel)
- ✅ **Odds çekebiliriz** (Pro plan ile)
- ✅ **Oyuncu istatistikleri çekebiliriz** (performans metrikleri)
- ⚠️ **FIFA rating yok** (ama performans rating'i var)

**Öneri:** API-FOOTBALL Pro plan ile hem odds hem de oyuncu performans istatistiklerini çekebiliriz! 🚀






