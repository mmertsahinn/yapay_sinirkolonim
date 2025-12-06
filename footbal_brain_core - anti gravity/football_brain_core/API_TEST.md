# 🔑 API Key'leri Ayarlama ve Test Etme

## Yöntem 1: PowerShell Script (Kolay)

```powershell
.\API_KEYS_SETUP.ps1
```

Sonra test et:
```powershell
python test_apis.py
```

---

## Yöntem 2: Manuel Ayarlama

PowerShell'de (her yeni terminal için tekrar yapman gerekir):

```powershell
# API-FOOTBALL
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"

# OpenRouter (GPT ve Grok için)
$env:OPENROUTER_API_KEY="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"
```

**Kontrol et:**
```powershell
echo $env:API_FOOTBALL_KEY
echo $env:OPENROUTER_API_KEY
```

---

## Yöntem 3: Kalıcı Ayarlama (Sistem Ortam Değişkenleri)

1. Windows tuşu + "Ortam değişkenleri" ara
2. "Ortam değişkenlerini düzenle" seç
3. "Yeni" butonuna tıkla
4. Şunları ekle:
   - Değişken adı: `API_FOOTBALL_KEY`
   - Değişken değeri: `647f5de88a29d150a9d4e2c0c7b636fb`
5. Tekrar "Yeni" → `OPENROUTER_API_KEY` → `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`
6. Tamam → Tamam
7. PowerShell'i yeniden başlat

---

## Test Et

```powershell
python test_apis.py
```

**Beklenen çıktı:**
```
API Test Baslatiyor...
============================================================

API Key Durumu:
   API_FOOTBALL_KEY: [OK] Ayarlandi
   OPENROUTER_API_KEY: [OK] Ayarlandi

1. API-FOOTBALL Testi
============================================================
[OK] API-FOOTBALL CALISIYOR! X fikstur bulundu.

2. OpenRouter - GPT Testi
============================================================
[OK] GPT CALISIYOR!
Cevap: ...

3. OpenRouter - Grok Testi
============================================================
[OK] Grok CALISIYOR!
Cevap: ...

4. Model Karsilastirmasi
============================================================
En hizli: GPT (X.XX saniye daha hizli)

TEST OZETI
============================================================
[OK] API-FOOTBALL: Calisiyor
[OK] OpenRouter GPT: Calisiyor
[OK] OpenRouter Grok: Calisiyor
```

---

## Sorun Giderme

### "API_FOOTBALL_KEY ayarlanmamış"
**Çözüm:** PowerShell'de key'i ayarla:
```powershell
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"
```

### "OPENROUTER_API_KEY ayarlanmamış"
**Çözüm:** PowerShell'de key'i ayarla:
```powershell
$env:OPENROUTER_API_KEY="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"
```

### Key'ler kayboluyor
**Çözüm:** Her yeni PowerShell penceresi için tekrar ayarlaman gerekir. Kalıcı yapmak için Yöntem 3'ü kullan.

---

## Not

- Key'ler config.py'de de varsayılan olarak var, ama ortam değişkeni öncelikli
- Güvenlik için key'leri kod içinde hardcode etme
- Key'ler sadece bu proje için kullanılmalı







