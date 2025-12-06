# 🚀 API Limit Yükseltme Rehberi

## API-FOOTBALL Planları

### Free Tier (Şu anki)
- **Limit:** 100 requests/day
- **Fiyat:** Ücretsiz
- **Süre:** 5 sezon için ~2-3 saat

### Basic Plan
- **Limit:** 300 requests/day
- **Fiyat:** ~$10/ay
- **Süre:** 5 sezon için ~1 saat
- **Link:** https://dashboard.api-football.com/pricing

### Pro Plan
- **Limit:** 1000 requests/day
- **Fiyat:** ~$30/ay
- **Süre:** 5 sezon için ~20-30 dakika
- **Link:** https://dashboard.api-football.com/pricing

### Enterprise Plan
- **Limit:** Unlimited
- **Fiyat:** Özel fiyatlandırma
- **Süre:** Çok hızlı

---

## Plan Yükseltme Adımları

1. **API-FOOTBALL Dashboard'a git:**
   https://dashboard.api-football.com/

2. **Yeni plan seç:**
   - Basic veya Pro planı seç
   - Ödeme yap

3. **Yeni API Key al:**
   - Dashboard'da yeni key oluştur
   - Eski key'i değiştir

4. **Yeni key'i ayarla:**
   ```powershell
   $env:API_FOOTBALL_KEY="YENI_KEY_BURAYA"
   ```

5. **Kod otomatik algılar:**
   - Kod, yeni plan limitini otomatik algılar
   - Rate limit delay'i otomatik ayarlanır
   - Daha hızlı yükleme başlar

---

## Mevcut Optimizasyonlar

Kod zaten optimize edildi:

✅ **Dinamik Rate Limiting:**
- API response header'larından limit bilgisi alınır
- Plan tipine göre otomatik delay ayarlanır
- Pro plan: 0.05s delay
- Basic plan: 0.08s delay
- Free plan: 0.1s delay

✅ **Rate Limit Takibi:**
- Günlük request sayısı takip edilir
- Limit aşıldığında uyarı verilir

---

## Hızlı Test

Plan tipini kontrol et:
```powershell
python -c "from football_brain_core.src.ingestion.api_client import APIFootballClient; import os; client = APIFootballClient(api_key=os.getenv('API_FOOTBALL_KEY')); print(f'Daily limit: {client.daily_limit}')"
```

---

## Not

- Free tier ile de çalışır, sadece daha yavaş
- Plan yükseltme isteğe bağlı
- Kod her iki durumda da çalışır







