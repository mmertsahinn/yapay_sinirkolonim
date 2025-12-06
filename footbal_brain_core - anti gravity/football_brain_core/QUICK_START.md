# Hızlı Başlangıç - Adım Adım

## 1. API Key'i Ayarla

PowerShell'de çalıştır:
```powershell
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"
```

## 2. Veritabanını Oluştur

```bash
python -m football_brain_core.src.cli.main init-db
```

Bu komut:
- SQLite veritabanı dosyası oluşturur (`football_brain.db`)
- Tüm tabloları (leagues, teams, matches, vb.) oluşturur

## 3. API'den Veri Yükle

### Seçenek A: Tarihsel Veri (Önerilen - İlk Kurulum)
```bash
python -m football_brain_core.src.cli.main load-historical
```

Bu komut:
- Son 5 sezonun tüm maçlarını çeker
- Ligleri, takımları, maçları veritabanına yazar
- ⚠️ Uzun sürebilir (API limitlerine bağlı)

### Seçenek B: Sadece Bugünün Fikstürleri (Hızlı Test)
```bash
python -m football_brain_core.src.cli.main daily-update
```

Bu komut:
- Bugün ve önümüzdeki 7 günün fikstürlerini çeker
- Son 7 günün maç sonuçlarını günceller
- Daha hızlı (sadece yakın tarihler)

## 4. Kontrol Et

Veritabanında veri olup olmadığını kontrol et:
```bash
# SQLite ile kontrol (opsiyonel)
sqlite3 football_brain.db "SELECT COUNT(*) FROM matches;"
```

## Sonraki Adımlar

Veri yüklendikten sonra:
1. Model eğitimi yapabilirsin
2. Tahminler üretebilirsin
3. Raporlar oluşturabilirsin







