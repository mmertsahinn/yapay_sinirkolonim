# Football Brain Core - Kurulum ve Başlangıç Rehberi

## 1. Ortam Değişkenlerini Ayarlama

Projeyi çalıştırmak için aşağıdaki ortam değişkenlerini ayarlamanız gerekiyor:

### Windows (PowerShell):
```powershell
$env:API_FOOTBALL_KEY="your_api_football_key_here"
$env:OPENAI_API_KEY="your_openai_key_here"  # Opsiyonel (LLM için)
$env:GROK_API_KEY="your_grok_key_here"  # Opsiyonel (LLM için)
$env:DATABASE_URL="sqlite:///./football_brain.db"  # Varsayılan, değiştirmeyebilirsiniz
```

### Windows (Kalıcı):
Sistem özelliklerinden ortam değişkenlerini ekleyin veya `.env` dosyası kullanın.

### Linux/Mac:
```bash
export API_FOOTBALL_KEY="your_api_football_key_here"
export OPENAI_API_KEY="your_openai_key_here"
export GROK_API_KEY="your_grok_key_here"
export DATABASE_URL="sqlite:///./football_brain.db"
```

## 2. API Key'leri Nereden Alınır?

- **API-FOOTBALL**: https://www.api-football.com/ adresinden ücretsiz hesap oluşturun
- **OpenAI**: https://platform.openai.com/ (LLM açıklamaları için)
- **Grok**: https://x.ai/ (Opsiyonel, alternatif LLM)

## 3. Başlangıç Adımları

### Adım 1: Veritabanını Initialize Et
```bash
python -m football_brain_core.src.cli.main init-db
```

### Adım 2: Tarihsel Veriyi Yükle
```bash
# Son 5 sezonu yükle (varsayılan)
python -m football_brain_core.src.cli.main load-historical

# Belirli sezonları yükle
python -m football_brain_core.src.cli.main load-historical --seasons 2020 2021 2022 2023 2024
```

**Not**: Bu işlem API limitlerine bağlı olarak uzun sürebilir. API-FOOTBALL'un ücretsiz planında günlük limit vardır.

### Adım 3: Model Eğitimi
```bash
# Eğitim ve validasyon sezonlarını belirt
python -m football_brain_core.src.cli.main train --train-seasons 2020 2021 2022 --val-seasons 2023
```

### Adım 4: Deney Çalıştırma (Opsiyonel)
```bash
python -m football_brain_core.src.cli.main experiment --train-seasons 2020 2021 2022 --val-seasons 2023
```

### Adım 5: Günlük Güncellemeler
```bash
# Yeni fikstürler ve sonuçları çek
python -m football_brain_core.src.cli.main daily-update
```

### Adım 6: Rapor Oluşturma
```bash
# Günlük rapor
python -m football_brain_core.src.cli.main report-daily

# Haftalık rapor
python -m football_brain_core.src.cli.main report-weekly
```

## 4. Önemli Notlar

1. **API Limitleri**: API-FOOTBALL'un ücretsiz planında günlük istek limiti vardır. Tarihsel veri yükleme sırasında rate limiting otomatik yapılıyor.

2. **Veritabanı**: Varsayılan olarak SQLite kullanılıyor. Daha büyük veri setleri için PostgreSQL'e geçebilirsiniz:
   ```bash
   export DATABASE_URL="postgresql://user:password@localhost/football_brain"
   ```

3. **Model Eğitimi**: İlk eğitim uzun sürebilir (saatler). GPU varsa otomatik kullanılır.

4. **Excel Raporları**: Raporlar `./reports/` klasörüne kaydedilir.

## 5. Sorun Giderme

- **API Key Hatası**: Ortam değişkenlerinin doğru ayarlandığından emin olun
- **Veritabanı Hatası**: `init-db` komutunu çalıştırdığınızdan emin olun
- **Import Hatası**: Proje kök dizininden çalıştırdığınızdan emin olun

## 6. Sonraki Adımlar

1. Model performansını değerlendirin
2. Farklı feature setleri deneyin
3. Model hiperparametrelerini optimize edin
4. LLM açıklamalarını test edin







