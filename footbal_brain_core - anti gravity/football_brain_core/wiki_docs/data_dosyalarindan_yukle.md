# Module: data_dosyalarindan_yukle.py

Data dosyalarından (.txt) veritabanına veri yükle
IDEMPOTENT: Var olan maçlara dokunmaz, sadece yeni maçları ekler
Sadece 2020-2025 arası maçları yükler

## Classes

### DataFileParser
football-data.co.uk formatındaki dosyaları parse eder

#### Methods
- **__init__**(self)

- **parse_date**(self, date_str, season_start_year)
  - Tarih string'ini parse et (örn: 'Fri Aug/13', 'Sat Nov/1')
season_start_year: Sezon başlangıç yılı (örn: 2025-26 sezonu için 2025)

- **parse_file**(self, file_path, league_name, season)
  - Dosyayı parse et ve veritabanına yükle
IDEMPOTENT: Var olan maçlara dokunmaz

Returns: (yeni_eklenen, zaten_var, tarih_disinda, toplam_okunan)

## Functions

### load_from_data_files()
football_data klasöründen TÜM dosyaları otomatik bulup yükle - IDEMPOTENT

