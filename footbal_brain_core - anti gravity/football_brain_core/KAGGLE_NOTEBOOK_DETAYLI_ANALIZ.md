# 📚 KAGGLE NOTEBOOK'LARI DETAYLI ANALİZ

## 🎯 NOTEBOOK 1: "Match Outcome Prediction Project" (saife245)

### 📋 GENEL AMAÇ
**Bookkeeper'lardan daha iyi tahmin yaparak pozitif ROI (Return on Investment) elde etmek**

---

### 🔧 ÖNEMLİ ÖZELLİKLER

#### 1. **FIFA Player Statistics Integration**
```python
def get_fifa_stats(match, player_stats):
    # Her maç için 22 oyuncunun (11 home + 11 away) FIFA rating'lerini çeker
    # Maç tarihinden önceki en son rating'leri kullanır
```
**Ne İşe Yarar:**
- Her oyuncunun `overall_rating` değerini alır
- Maç tarihinden önceki en güncel rating'i kullanır (temporal accuracy)
- 22 oyuncu × 1 feature = 22 feature
- Takım gücünü oyuncu bazında ölçer

**Bizim Sistemde:**
- ❌ Şu an yok
- ✅ Eklenebilir: `Player` tablosu + `PlayerAttributes` tablosu
- ✅ Her takımın ortalama FIFA rating'i feature olarak eklenebilir

---

#### 2. **Match Features (Takım Formu)**
```python
def get_match_features(match, matches, x = 10):
    # Son 10 maçtan:
    # - Goals scored/conceded
    # - Wins
    # - Head-to-head geçmişi
```
**Ne İşe Yarar:**
- `home_team_goals_difference`: Son 10 maçta gol farkı
- `away_team_goals_difference`: Son 10 maçta gol farkı
- `games_won_home_team`: Son 10 maçta kazanılan maç sayısı
- `games_won_away_team`: Son 10 maçta kazanılan maç sayısı
- `games_against_won`: İki takım arası son 3 maçta home takımın kazandığı
- `games_against_lost`: İki takım arası son 3 maçta away takımın kazandığı

**Bizim Sistemde:**
- ✅ Var: `build_team_features()` benzer şekilde çalışıyor
- ⚠️ Eksik: Head-to-head geçmişi detaylı değil
- ✅ Eklenebilir: `get_last_matches_against_eachother()` fonksiyonu

---

#### 3. **Bookkeeper Odds Integration**
```python
def get_bookkeeper_data(matches, bookkeepers, horizontal = True):
    # 10 farklı bookkeeper'dan odds çeker
    # Odds'ları probability'ye çevirir
```
**Ne İşe Yarar:**
- 10 farklı bookkeeper: `['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']`
- Her bookkeeper için 3 probability: Win, Draw, Defeat
- Toplam: 10 × 3 = 30 feature
- Bookkeeper'ların tahminlerini model'e öğretir

**Bizim Sistemde:**
- ❌ Şu an yok
- ✅ Eklenebilir: API-FOOTBALL'dan odds çekilebilir
- ✅ Bookkeeper probability'leri feature olarak eklenebilir

---

#### 4. **Dimensionality Reduction (PCA)**
```python
pca = PCA()
# Feature sayısını azaltır, noise'u temizler
```
**Ne İşe Yarar:**
- Çok fazla feature varsa (FIFA stats + bookkeeper = 50+ feature)
- PCA ile önemli feature'ları korur, gereksizleri atar
- Model eğitim süresini kısaltır
- Overfitting'i azaltır

**Bizim Sistemde:**
- ❌ Şu an yok
- ⚠️ Gerekli mi? Bizim feature sayımız az (20-30), belki gerekmez
- ✅ İleride feature sayısı artarsa kullanılabilir

---

#### 5. **Probability Calibration (Isotonic Regression)**
```python
clf = CalibratedClassifierCV(best_pipe, cv='prefit', method='isotonic')
# Model'in probability tahminlerini daha doğru hale getirir
```
**Ne İşe Yarar:**
- Model'in tahmin ettiği probability'ler gerçek probability'lere yakın olmayabilir
- Örnek: Model %70 diyor ama gerçekte %50 olabilir
- Isotonic regression ile probability'leri kalibre eder
- **Betting için kritik!** Çünkü doğru probability = doğru bahis kararı

**Bizim Sistemde:**
- ❌ Şu an yok
- ✅ Eklenebilir: Model'in probability output'larını kalibre edebiliriz
- ✅ Özellikle betting stratejisi için önemli

---

#### 6. **Model Comparison (5 Farklı Classifier)**
```python
clfs = [RandomForestClassifier, AdaBoostClassifier, GaussianNB, 
        KNeighborsClassifier, LogisticRegression]
# Her birini test eder, en iyisini seçer
```
**Ne İşe Yarar:**
- Random Forest: Ensemble method, güçlü
- AdaBoost: Boosting, zayıf modelleri birleştirir
- GaussianNB: Basit, hızlı
- KNN: Benzer maçları bulur
- Logistic Regression: Linear, interpretable
- **Sonuç:** GaussianNB + PCA en iyi performansı gösterdi (%55.38)

**Bizim Sistemde:**
- ✅ Var: Multi-task learning (PyTorch)
- ⚠️ Farklı: Bizim sistem daha gelişmiş (6 market aynı anda)
- ✅ Öğrenilebilir: Farklı model mimarileri deneyebiliriz

---

#### 7. **Betting Strategy Optimization**
```python
def find_good_bets(clf, dim_reduce, bk, bookkeepers, matches, fifa_data, 
                   percentile, prob_cap):
    # Model probability > Bookkeeper probability ise bahis yap
    # Minimum probability threshold var
```
**Ne İşe Yarar:**
- Model'in tahmin ettiği probability > Bookkeeper probability ise bahis yap
- `percentile`: En yüksek farklılık gösteren bahisleri seç
- `prob_cap`: Minimum probability threshold (örn: %50)
- **Sonuç:** Negatif ROI (-45.8%) - Model bookkeeper'dan kötü

**Bizim Sistemde:**
- ❌ Şu an yok
- ✅ Eklenebilir: Tahminlerimizi bookkeeper odds ile karşılaştırabiliriz
- ✅ ROI hesaplama sistemi eklenebilir

---

### 📊 NOTEBOOK 1 ÖZET

**Güçlü Yönler:**
- ✅ FIFA player stats (oyuncu bazlı güç ölçümü)
- ✅ Bookkeeper odds integration (piyasa tahminleri)
- ✅ Probability calibration (doğru probability'ler)
- ✅ Betting strategy (ROI odaklı)

**Zayıf Yönler:**
- ❌ Sadece 1-X-2 tahmini (bizim 6 market var)
- ❌ Basit feature engineering (bizim daha detaylı)
- ❌ Negatif ROI (model bookkeeper'dan kötü)

---

## 🎯 NOTEBOOK 2: "Predicting the Winning Football Team"

### 📋 GENEL AMAÇ
**Home team'in kazanıp kazanmayacağını tahmin etmek (binary classification: H vs NH)**

---

### 🔧 ÖNEMLİ ÖZELLİKLER

#### 1. **Matchweek-Based Cumulative Statistics**
```python
def get_goals_scored(playing_stat):
    # Her matchweek sonunda kümülatif gol sayısını hesaplar
    # Örnek: Matchweek 5'te takımın toplam attığı goller
```
**Ne İşe Yarar:**
- `HTGS` (Home Team Goals Scored): Maç haftasına kadar toplam atılan goller
- `ATGS` (Away Team Goals Scored): Maç haftasına kadar toplam atılan goller
- `HTGC` (Home Team Goals Conceded): Maç haftasına kadar toplam yenilen goller
- `ATGC` (Away Team Goals Conceded): Maç haftasına kadar toplam yenilen goller
- **Kritik:** Her maç için o anki sezon durumunu yansıtır

**Örnek:**
- Matchweek 1: HTGS = 0 (henüz gol yok)
- Matchweek 5: HTGS = 8 (5 haftada 8 gol atmış)
- Matchweek 10: HTGS = 15 (10 haftada 15 gol atmış)

**Bizim Sistemde:**
- ❌ Şu an yok
- ✅ Eklenebilir: `MatchRepository.get_cumulative_stats(team_id, match_date)`
- ✅ Sezon içi trend'i yakalar (bizim sistem sadece son N maça bakıyor)

---

#### 2. **Cumulative Points (Lig Pozisyonu)**
```python
def get_agg_points(playing_stat):
    # Her matchweek sonunda kümülatif puanları hesaplar
    # HTP = Home Team Points (maç haftasına kadar toplam puan)
    # ATP = Away Team Points (maç haftasına kadar toplam puan)
```
**Ne İşe Yarar:**
- `HTP`: Home takımın maç haftasına kadar toplam puanı
- `ATP`: Away takımın maç haftasına kadar toplam puanı
- Lig pozisyonunu yansıtır (daha fazla puan = daha iyi takım)
- **Normalizasyon:** `HTP / MatchWeek` ile hafta başına ortalama puan

**Bizim Sistemde:**
- ❌ Şu an yok
- ✅ Eklenebilir: `LeagueStanding` tablosu oluşturulabilir
- ✅ Her maç için o anki lig pozisyonu feature olarak eklenebilir

---

#### 3. **Form String Features (W/D/L)**
```python
def get_form(playing_stat, num):
    # Son N maçın sonuçlarını string olarak tutar
    # Örnek: "WWDLW" = Son 5 maç: Win, Win, Draw, Loss, Win
```
**Ne İşe Yarar:**
- `HM1`, `HM2`, `HM3`, `HM4`, `HM5`: Home takımın son 5 maçının sonuçları
- `AM1`, `AM2`, `AM3`, `AM4`, `AM5`: Away takımın son 5 maçının sonuçları
- **Dummy encoding:** Her sonuç (W/D/L) ayrı feature olur
- **Form points:** W=3, D=1, L=0 puan toplamı

**Örnek:**
- `HM1 = 'W'`, `HM2 = 'W'`, `HM3 = 'D'` → Son 3 maç: Win, Win, Draw
- `HTFormPts = 3 + 3 + 1 = 7` (son 3 maçtan 7 puan)

**Bizim Sistemde:**
- ⚠️ Kısmen var: `win_rate`, `draw_rate`, `loss_rate` (oran olarak)
- ❌ Eksik: Form string'leri (W/D/L sequence)
- ✅ Eklenebilir: Son N maçın sonuçlarını string olarak tutabiliriz

---

#### 4. **Win/Loss Streak Detection**
```python
def get_3game_ws(string):
    if string[-3:] == 'WWW':
        return 1  # Son 3 maç kazanmış
    else:
        return 0
```
**Ne İşe Yarar:**
- `HTWinStreak3`: Home takım son 3 maçı kazandı mı? (1/0)
- `HTWinStreak5`: Home takım son 5 maçı kazandı mı? (1/0)
- `HTLossStreak3`: Home takım son 3 maçı kaybetti mi? (1/0)
- `HTLossStreak5`: Home takım son 5 maçı kaybetti mi? (1/0)
- **Momentum feature:** Takımın form trend'ini yakalar

**Bizim Sistemde:**
- ❌ Şu an yok
- ✅ Eklenebilir: `FeatureBuilder`'a streak detection eklenebilir
- ✅ Momentum feature'ı model'e güç katabilir

---

#### 5. **Goal Difference Normalization by Matchweek**
```python
playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
playing_stat['HTGD'] = playing_stat['HTGD'] / playing_stat.MW
# Gol farkını matchweek'e böl (hafta başına ortalama)
```
**Ne İşe Yarar:**
- `HTGD`: Home takımın gol farkı (attığı - yediği)
- `ATGD`: Away takımın gol farkı
- **Normalizasyon:** Matchweek'e bölünür (erken sezonda yüksek değerleri dengeler)
- Örnek: Matchweek 5'te +10 gol farkı → 10/5 = 2.0 (hafta başına 2 gol farkı)

**Bizim Sistemde:**
- ⚠️ Kısmen var: `avg_goals_scored - avg_goals_conceded` (son N maç ortalaması)
- ❌ Eksik: Sezon başından itibaren kümülatif + normalizasyon
- ✅ Eklenebilir: Sezon bazlı kümülatif istatistikler

---

#### 6. **Feature Selection (Multicollinearity Removal)**
```python
# Correlation matrix çizilir
# Yüksek korelasyonlu feature'lar kaldırılır
dataset2 = dataset.drop(columns=['HTGS', 'ATGS', 'HTGC', 'ATGC', ...])
```
**Ne İşe Yarar:**
- İki feature çok yüksek korelasyonlu ise (örn: 0.95+), biri gereksiz
- Örnek: `HTGS` ve `HTGC` zaten `HTGD`'de var → `HTGS` ve `HTGC` kaldırılabilir
- **Overfitting'i azaltır**, model daha genelleştirilebilir olur

**Bizim Sistemde:**
- ❌ Şu an yok
- ✅ Eklenebilir: Feature correlation analizi yapılabilir
- ✅ Gereksiz feature'lar kaldırılabilir

---

#### 7. **Data Standardization**
```python
from sklearn.preprocessing import scale
X_all[['HTGD','ATGD','HTP','ATP']] = scale(X_all[['HTGD','ATGD','HTP','ATP']])
```
**Ne İşe Yarar:**
- Feature'ları mean=0, std=1'e normalize eder
- Farklı scale'deki feature'ları aynı seviyeye getirir
- Örnek: `HTP` (0-100 arası) ve `HTGD` (-50 to +50) → ikisi de -2 to +2 arası

**Bizim Sistemde:**
- ❌ Şu an yok
- ✅ Eklenebilir: Feature normalization eklenebilir
- ✅ Model performansını artırabilir

---

#### 8. **Model Comparison (4 Farklı Classifier)**
```python
# Logistic Regression: %64.65 accuracy
# SVM: %54 (kötü, sadece NH tahmin ediyor)
# Random Forest: %64.64 accuracy
# XGBoost: %65.65 accuracy (EN İYİ)
```
**Ne İşe Yarar:**
- XGBoost en iyi performansı gösterdi
- GridSearchCV ile hyperparameter tuning yapıldı
- **Sonuç:** %64.77 accuracy (test set)

**Bizim Sistemde:**
- ✅ Var: PyTorch Multi-task model
- ⚠️ Farklı: Bizim sistem 6 market için aynı anda tahmin yapıyor
- ✅ Öğrenilebilir: XGBoost'u baseline olarak kullanabiliriz

---

### 📊 NOTEBOOK 2 ÖZET

**Güçlü Yönler:**
- ✅ Matchweek-based cumulative stats (sezon içi trend)
- ✅ Form string features (W/D/L sequence)
- ✅ Win/loss streak detection (momentum)
- ✅ Goal difference normalization (scale düzeltme)
- ✅ Feature selection (multicollinearity removal)

**Zayıf Yönler:**
- ❌ Sadece binary classification (H vs NH)
- ❌ Draw'ı ignore ediyor (bizim sistem 1-X-2 tahmin ediyor)
- ❌ Basit model (XGBoost, bizim sistem daha gelişmiş)

---

## 🔄 İKİ NOTEBOOK KARŞILAŞTIRMASI

| Özellik | Notebook 1 | Notebook 2 | Bizim Sistem |
|---------|------------|-------------|--------------|
| **FIFA Stats** | ✅ Var | ❌ Yok | ❌ Yok (eklenebilir) |
| **Bookkeeper Odds** | ✅ Var | ❌ Yok | ❌ Yok (eklenebilir) |
| **Matchweek Stats** | ❌ Yok | ✅ Var | ❌ Yok (eklenebilir) |
| **Form Strings** | ❌ Yok | ✅ Var | ⚠️ Kısmen var |
| **Streak Detection** | ❌ Yok | ✅ Var | ❌ Yok (eklenebilir) |
| **Probability Calibration** | ✅ Var | ❌ Yok | ❌ Yok (eklenebilir) |
| **Multi-Market** | ❌ Yok (sadece 1-X-2) | ❌ Yok (sadece H/NH) | ✅ Var (6 market) |
| **Self-Learning** | ❌ Yok | ❌ Yok | ✅ Var |
| **Evolution Core** | ❌ Yok | ❌ Yok | ✅ Var |
| **Hype Features** | ❌ Yok | ❌ Yok | ✅ Var |

---

## 🎯 BİZİM SİSTEME EKLENEBİLECEKLER

### 1. **FIFA Player Stats** (Notebook 1'den)
- Her takımın ortalama FIFA rating'i
- En iyi 11'in ortalama rating'i
- Oyuncu bazlı güç ölçümü

### 2. **Bookkeeper Odds** (Notebook 1'den)
- API-FOOTBALL'dan odds çekme
- Bookkeeper probability'leri feature olarak ekleme
- Model tahminleri vs bookkeeper karşılaştırması

### 3. **Matchweek-Based Stats** (Notebook 2'den)
- Sezon başından itibaren kümülatif istatistikler
- Lig pozisyonu feature'ı
- Matchweek normalization

### 4. **Form Strings & Streaks** (Notebook 2'den)
- Son N maçın W/D/L sequence'i
- Win/loss streak detection
- Momentum feature'ları

### 5. **Probability Calibration** (Notebook 1'den)
- Isotonic regression ile probability kalibrasyonu
- Daha doğru probability tahminleri
- Betting stratejisi için kritik

---

## ✅ SONUÇ

**Her iki notebook da:**
- ✅ Yeterince açıklayıcı
- ✅ Kodlar kullanılabilir
- ✅ Feature engineering teknikleri öğrenilebilir
- ✅ Model yaklaşımları referans alınabilir

**Bizim sistem:**
- ✅ Daha gelişmiş (multi-task, self-learning, evolution)
- ⚠️ Eksik feature'lar var (FIFA stats, bookkeeper odds, matchweek stats)
- ✅ Bu notebook'lardan öğrenilebilir tekniklerle geliştirilebilir

**Öneri:** Bu notebook'lardaki feature engineering tekniklerini bizim sistemimize entegre edelim! 🚀






