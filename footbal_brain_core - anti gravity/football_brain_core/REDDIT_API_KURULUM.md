# 🔴 Reddit API Client ID Kurulumu

Reddit API kullanmak için client_id ve client_secret almanız gerekiyor. İşte adım adım rehber:

## 📋 ADIM 1: Reddit Hesabı Oluştur/Giriş Yap

1. https://www.reddit.com adresine git
2. Hesabın varsa giriş yap, yoksa ücretsiz hesap oluştur

## 📋 ADIM 2: Reddit App Oluştur

1. **Preferences** sayfasına git:
   - Sağ üst köşedeki profil ikonuna tıkla
   - "User Settings" → "Safety & Privacy" → En altta "apps" linkine tıkla
   - VEYA direkt: https://www.reddit.com/prefs/apps

2. **"create another app"** veya **"create app"** butonuna tıkla

3. **App bilgilerini doldur:**
   - **Name**: `football_brain_core` (veya istediğin isim)
   - **App type**: **"script"** seç (en basit)
   - **Description**: `Football match hype analyzer` (opsiyonel)
   - **About URL**: Boş bırakabilirsin
   - **Redirect URI**: `http://localhost:8080` (zorunlu, script için)

4. **"create app"** butonuna tıkla

## 📋 ADIM 3: Client ID ve Secret'i Al

App oluşturulduktan sonra şunları göreceksin:

```
under the name "football_brain_core"
client_id: xxxxxxxxxxxxxx
secret: xxxxxxxxxxxxxx
```

- **client_id**: App'in altındaki küçük metin (örn: `abc123def456`)
- **secret**: "secret" yazan yerdeki uzun metin (örn: `xyz789uvw012_secret_key`)

## 📋 ADIM 4: Environment Variable Olarak Ayarla

### Windows PowerShell:
```powershell
$env:REDDIT_CLIENT_ID="abc123def456"
$env:REDDIT_CLIENT_SECRET="xyz789uvw012_secret_key"
```

### Kalıcı yapmak için:
1. Sistem Özellikleri → Ortam Değişkenleri
2. "Yeni" → Kullanıcı değişkeni
3. `REDDIT_CLIENT_ID` = `abc123def456`
4. `REDDIT_CLIENT_SECRET` = `xyz789uvw012_secret_key`

## 📋 ADIM 5: Kodu Güncelle

`alternative_hype_scraper.py` dosyasını güncelle:

```python
import os

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),  # Environment variable'dan al
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),  # Environment variable'dan al
    user_agent="football_brain_core/1.0"
)
```

## ⚠️ ÖNEMLİ NOTLAR

1. **Client ID olmadan da çalışır** - Sadece rate limit daha düşük olur
2. **Rate Limit**: 
   - Client ID olmadan: ~60 request/dakika
   - Client ID ile: ~600 request/dakika
3. **Güvenlik**: Client secret'i asla paylaşma veya GitHub'a yükleme!

## ✅ Test Et

```python
import praw
import os

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="football_brain_core/1.0"
)

# Test
subreddit = reddit.subreddit("soccer")
print(f"Subreddit: {subreddit.display_name}")
print("✅ Reddit API çalışıyor!")
```

## 🔗 Faydalı Linkler

- Reddit Apps: https://www.reddit.com/prefs/apps
- PRAW Dokümantasyon: https://praw.readthedocs.io/
- Reddit API Rate Limits: https://www.reddit.com/r/redditdev/wiki/api





