# Odds Verilerini JSON Export'a Entegre Etme Kılavuzu

Bu script, odds klasöründeki tüm CSV dosyalarını okuyup `football_brain_export.json` dosyasına entegre eder.

## Kullanım

```bash
cd football_brain_core/odds
python integrate_odds_to_export.py
```

## Ne Yapar?

1. **CSV Dosyalarını Okur:**
   - Tüm lig klasörlerindeki CSV dosyalarını tarar
   - England, Spain, Italy, Germany, France, Portugal, Turkey liglerini işler

2. **Odds Verilerini Çevirir:**
   - CSV formatındaki odds verilerini JSON formatına çevirir
   - Tüm bahis şirketlerinin oranlarını toplar
   - Market ortalama ve maksimum değerlerini ekler

3. **Export Dosyasına Entegre Eder:**
   - Mevcut `football_brain_export.json` dosyasını yükler
   - Yeni odds verilerini ekler
   - Mevcut odds varsa günceller (birleştirir)
   - Yedek dosya oluşturur

## Desteklenen Odds Tipleri

### Match Result (1-X-2) Odds:
- Bet365 (B365H, B365D, B365A)
- Betfair (BFH, BFD, BFA)
- Betfred (BFDH, BFDD, BFDA)
- BetMGM (BMGMH, BMGMD, BMGMA)
- Betvictor (BVH, BVD, BVA)
- Coral (CLH, CLD, CLA)
- Ladbrokes (LBH, LBD, LBA)
- Pinnacle (PSH, PSD, PSA)
- William Hill (WHH, WHD, WHA)
- Market averages (MaxH, MaxD, MaxA, AvgH, AvgD, AvgA)

### Closing Odds:
- Bet365 Closing (B365CH, B365CD, B365CA)
- Betfred Closing
- BetMGM Closing
- ve diğerleri...

### Over/Under 2.5 Goals:
- Bet365 (B365>2.5, B365<2.5)
- Pinnacle (P>2.5, P<2.5)
- Market averages (Max>2.5, Max<2.5, Avg>2.5, Avg<2.5)

### Asian Handicap:
- Handicap size (AHh)
- Bet365 Asian Handicap (B365AHH, B365AHA)
- Pinnacle Asian Handicap (PAHH, PAHA)
- ve diğerleri...

## JSON Çıktı Formatı

Her odds kaydı şu formatta:

```json
{
  "match_id": "England_2025-08-15_Liverpool_Bournemouth",
  "league": "England",
  "division": "E0",
  "date": "2025-08-15T20:00:00",
  "home_team": "Liverpool",
  "away_team": "Bournemouth",
  "home_goals": 4,
  "away_goals": 2,
  "odds": {
    "b365_h": 1.3,
    "b365_d": 6.0,
    "b365_a": 8.5,
    "max_h": 1.34,
    "avg_h": 1.31,
    "b365_over_25": 1.36,
    "b365_under_25": 3.2,
    "all_odds": {
      "BWH": 1.32,
      "BWD": 5.5,
      ...
    }
  }
}
```

## Özellikler

✅ **Otomatik Yedekleme:** İşlem öncesi export dosyasının yedeği alınır  
✅ **Birleştirme:** Mevcut odds varsa güncellenir, yeni alanlar eklenir  
✅ **Hata Toleransı:** Bozuk CSV dosyaları atlanır, işlem devam eder  
✅ **İlerleme Göstergesi:** Hangi dosyaların işlendiği gösterilir  
✅ **Toplam İstatistik:** Kaç odds eklendiği/güncellendiği gösterilir  

## Çıktı Örneği

```
============================================================
ODDS VERILERI JSON EXPORT'A ENTEGRE EDILIYOR
============================================================

1. CSV dosyalari okunuyor...
Bulunan lig klasorleri: 7

England isleniyor...
  E0.csv: 380 maç
  E1.csv: 552 maç
  ...

Spain isleniyor...
  SP1.csv: 380 maç
  ...

Toplam 8500 maç odds verisi bulundu

2. Export dosyasina entegre ediliyor...
Export dosyasi yukleniyor: .../football_brain_export.json

Yeni odds eklendi: 8500
Guncellenen odds: 0
Toplam odds sayisi: 8500

Yedek olusturuldu: .../football_brain_export_backup.json

Export dosyasi guncelleniyor...
✅ Tamamlandi! Export dosyasi guncellendi
```

## Notlar

- Script çalışmadan önce `football_brain_export.json` dosyasının mevcut olması önerilir
- Büyük CSV dosyaları işlenirken zaman alabilir
- Tüm odds verileri `data.match_odds` array'ine eklenir
- Match ID formatı: `{league}_{date}_{home_team}_{away_team}`

## Sorun Giderme

**Hata: Export dosyası bulunamadı**
- Script yeni bir export dosyası oluşturur
- Veya önce export dosyasını oluşturun

**Hata: CSV okuma hatası**
- CSV dosyasının UTF-8 encoding'de olduğundan emin olun
- Hatalı satırlar atlanır, işlem devam eder

**Odds verisi eksik**
- CSV'deki kolon isimlerini kontrol edin
- Mapping tablosuna yeni kolon eklenebilir

