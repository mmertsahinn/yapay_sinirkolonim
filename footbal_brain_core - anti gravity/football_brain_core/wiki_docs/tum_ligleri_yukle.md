# Module: tum_ligleri_yukle.py

TÜM LİGLERİ YÜKLE - Tüm master klasörlerinden
england-master, espana-master, europe-master, italy-master

## Classes

### DataFileParser
football-data.co.uk formatındaki dosyaları parse eder

#### Methods
- **__init__**(self)

- **parse_date**(self, date_str, year)
  - Tarih string'ini parse et (örn: 'Fri Aug/13')

- **parse_file**(self, file_path, league_name, season)
  - Dosyayı parse et ve veritabanına yükle

## Functions

### yukle_england_master()
England-master klasöründen tüm ligleri yükle

### yukle_espana_master()
Espana-master klasöründen tüm ligleri yükle

### yukle_italy_master()
Italy-master klasöründen tüm ligleri yükle

### yukle_europe_master()
Europe-master klasöründen tüm ligleri yükle

### tum_ligleri_yukle()
Tüm master klasörlerinden tüm ligleri yükle

