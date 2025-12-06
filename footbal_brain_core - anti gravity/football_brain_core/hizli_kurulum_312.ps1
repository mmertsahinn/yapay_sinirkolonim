# Python 3.12 Hızlı Kurulum Scripti
# PowerShell'de çalıştır: .\hizli_kurulum_312.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PYTHON 3.12 KURULUM VE YAPILANDIRMA" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Python 3.12 kontrolü
Write-Host "[1/5] Python 3.12 kontrol ediliyor..." -ForegroundColor Yellow
try {
    $python312 = py -3.12 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python 3.12 bulundu: $python312" -ForegroundColor Green
    } else {
        Write-Host "❌ Python 3.12 bulunamadı!" -ForegroundColor Red
        Write-Host "📥 Lütfen Python 3.12'yi kurun: https://www.python.org/downloads/release/python-3127/" -ForegroundColor Yellow
        Write-Host "⚠️  Kurulum sırasında 'Add Python to PATH' seçeneğini işaretleyin!" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Python 3.12 bulunamadı!" -ForegroundColor Red
    Write-Host "📥 Lütfen Python 3.12'yi kurun: https://www.python.org/downloads/release/python-3127/" -ForegroundColor Yellow
    exit 1
}

# 2. Virtual environment oluştur
Write-Host ""
Write-Host "[2/5] Virtual environment oluşturuluyor..." -ForegroundColor Yellow
if (Test-Path "venv312") {
    Write-Host "⚠️  venv312 zaten var, atlanıyor..." -ForegroundColor Yellow
} else {
    py -3.12 -m venv venv312
    Write-Host "✅ Virtual environment oluşturuldu: venv312" -ForegroundColor Green
}

# 3. Virtual environment'ı aktif et
Write-Host ""
Write-Host "[3/5] Virtual environment aktif ediliyor..." -ForegroundColor Yellow
& .\venv312\Scripts\Activate.ps1
Write-Host "✅ Virtual environment aktif" -ForegroundColor Green

# 4. snscrape yükle
Write-Host ""
Write-Host "[4/5] snscrape yükleniyor..." -ForegroundColor Yellow
pip install snscrape
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ snscrape yüklendi" -ForegroundColor Green
} else {
    Write-Host "❌ snscrape yüklenemedi!" -ForegroundColor Red
    exit 1
}

# 5. Test et
Write-Host ""
Write-Host "[5/5] snscrape test ediliyor..." -ForegroundColor Yellow
python -c "import snscrape.modules.twitter as sntwitter; print('✅ snscrape çalışıyor!')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ snscrape test başarılı!" -ForegroundColor Green
} else {
    Write-Host "❌ snscrape test başarısız!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ KURULUM TAMAMLANDI!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Şimdi hype çekmeyi başlatabilirsin:" -ForegroundColor Yellow
Write-Host "   python tum_maclar_hype_cek.py" -ForegroundColor White
Write-Host ""






