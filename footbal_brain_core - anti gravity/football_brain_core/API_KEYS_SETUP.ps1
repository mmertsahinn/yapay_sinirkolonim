# API Key'leri Ayarla - PowerShell Script
# Bu dosyayı çalıştırmak için: .\API_KEYS_SETUP.ps1

Write-Host "API Key'leri ayarlaniyor..." -ForegroundColor Green

# API-FOOTBALL Key
$env:API_FOOTBALL_KEY="647f5de88a29d150a9d4e2c0c7b636fb"
Write-Host "[OK] API_FOOTBALL_KEY ayarlandi" -ForegroundColor Green

# OpenRouter Key (GPT ve Grok için)
$env:OPENROUTER_API_KEY="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"
Write-Host "[OK] OPENROUTER_API_KEY ayarlandi" -ForegroundColor Green

Write-Host "`nAPI Key'ler hazir! Test etmek icin:" -ForegroundColor Yellow
Write-Host "   python test_apis.py" -ForegroundColor Cyan






