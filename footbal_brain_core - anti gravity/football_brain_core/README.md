# Football Brain Core v1

Futbol takımlarının davranışlarını çok boyutlu olarak öğrenen, çoklu bahis marketlerinde olası senaryoları çıkaran ve bunları yüksek doğrulukla yorumlayan bir "zeka çekirdeği".

> 📘 **Documentation**: Visit the [**Code Wiki**](WIKI.md) for detailed Architecture, Concepts, and Component guides.

## Özellikler

- 🏆 **7 Lig Desteği**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Liga Portugal, Süper Lig
- 📊 **Çoklu Market Tahminleri**: Maç Sonucu, BTTS, Alt/Üst, Skor ve daha fazlası
- 🤖 **Multi-Task Neural Network**: Tüm marketler için aynı anda öğrenme
- 📈 **Deney Takibi**: Model versiyonları ve performans metrikleri
- 💬 **LLM Açıklamaları**: Tahminler için otomatik senaryo üretimi
- 📋 **Excel Raporları**: Günlük/haftalık öğrenme defteri çıktıları

## Hızlı Başlangıç

Detaylı kurulum için [SETUP.md](SETUP.md) dosyasına bakın.

### 1. Ortam Değişkenlerini Ayarlayın
```powershell
$env:API_FOOTBALL_KEY="your_key_here"
```

### 2. Veritabanını Initialize Edin
```bash
python -m football_brain_core.src.cli.main init-db
```

### 3. Tarihsel Veriyi Yükleyin
```bash
python -m football_brain_core.src.cli.main load-historical
```

### 4. Model Eğitin
```bash
python -m football_brain_core.src.cli.main train --train-seasons 2020 2021 2022 --val-seasons 2023
```

## Proje Yapısı

```
football_brain_core/
├── src/
│   ├── db/              # Veritabanı şeması ve işlemleri
│   ├── ingestion/       # API-FOOTBALL entegrasyonu
│   ├── features/        # Feature engineering
│   ├── models/          # ML model ve eğitim
│   ├── experiments/     # Deney yönetimi
│   ├── explanations/    # LLM entegrasyonu
│   ├── inference/       # Tahmin ve backtest
│   ├── reporting/       # Excel raporları
│   └── cli/             # Komut satırı arayüzü
└── reports/             # Excel çıktıları
```

## Gereksinimler

- Python 3.8+
- SQLite veya PostgreSQL
- API-FOOTBALL API key
- OpenAI/Grok API key (opsiyonel, LLM için)

## Lisans

Kişisel kullanım için geliştirilmiştir.







