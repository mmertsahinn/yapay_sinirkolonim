@echo off
echo ============================================================
echo MODEL EGITIM KONTROL EKRANI
echo ============================================================
echo.

:loop
cls
echo ============================================================
echo ZAMAN: %date% %time%
echo ============================================================
echo.

echo [1] MODEL DOSYALARI:
echo -------------------
if exist football_prediction_ensemble.joblib (
    echo [OK] football_prediction_ensemble.joblib - BULUNDU
    dir /B football_prediction_ensemble.joblib | findstr /R ".*"
) else (
    echo [BEKLENIYOR] football_prediction_ensemble.joblib - Henuz olusturulmadi
)

if exist label_encoder.joblib (
    echo [OK] label_encoder.joblib - BULUNDU
) else (
    echo [BEKLENIYOR] label_encoder.joblib - Henuz olusturulmadi
)

if exist home_goals_model.joblib (
    echo [OK] home_goals_model.joblib - BULUNDU
) else (
    echo [BEKLENIYOR] home_goals_model.joblib - Henuz olusturulmadi
)

if exist away_goals_model.joblib (
    echo [OK] away_goals_model.joblib - BULUNDU
) else (
    echo [BEKLENIYOR] away_goals_model.joblib - Henuz olusturulmadi
)

echo.
echo [2] GRAFIK DOSYALARI:
echo --------------------
if exist confusion_matrix.png (
    echo [OK] confusion_matrix.png - BULUNDU
) else (
    echo [BEKLENIYOR] confusion_matrix.png - Henuz olusturulmadi
)

if exist feature_importance.png (
    echo [OK] feature_importance.png - BULUNDU
) else (
    echo [BEKLENIYOR] feature_importance.png - Henuz olusturulmadi
)

echo.
echo [3] PYTHON ISLEMLERI:
echo --------------------
tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq train*" 2>nul | find /I "python.exe" >nul
if %errorlevel% equ 0 (
    echo [CALISYOR] Python egitim scripti aktif
) else (
    echo [DURDURULDU] Python egitim scripti bulunamadi
)

echo.
echo ============================================================
echo 10 saniye sonra yenilenecek... (Durdurmak icin Ctrl+C)
echo ============================================================

timeout /t 10 /nobreak >nul
goto loop





