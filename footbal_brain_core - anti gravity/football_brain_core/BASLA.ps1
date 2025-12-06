# API Key'leri Ayarla ve Test Et
# Bu dosyayı çalıştırmak için: .\BASLA.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "API Key'leri Ayarlaniyor..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# API-FOOTBALL Key (Yeni key - ÇALIŞAN)
$env:API_FOOTBALL_KEY="5abc4531c6a98fedb6a657d7f439d1c0"
Write-Host "[OK] API_FOOTBALL_KEY ayarlandi (Yeni key - CALISIYOR)" -ForegroundColor Green

# OpenRouter Key
$env:OPENROUTER_API_KEY="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"
Write-Host "[OK] OPENROUTER_API_KEY ayarlandi" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "API'ler Test Ediliyor..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test et
python test_apis.py

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Test Tamamlandi!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nSonraki adimlar:" -ForegroundColor Yellow
Write-Host "   1. Veritabani olustur: python init_db.py" -ForegroundColor Cyan
Write-Host "   2. Veri yukle: python load_data.py --seasons 2021 2022 2023" -ForegroundColor Cyan
Write-Host "   3. Model egit: python quick_test.py" -ForegroundColor Cyan

