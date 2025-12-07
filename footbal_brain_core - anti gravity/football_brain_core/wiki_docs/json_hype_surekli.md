# Module: json_hype_surekli.py

JSON HYPE FILLER - HIZLI KAYIT (INSTANT SAVE)
- Listenin EN SONUNDAN başlar.
- Her 5 maçta bir diske kaydeder (Anlık değişim görünür).
- Eksik verileri (null) doldurur.
- 10 Thread ile hızlı çalışır.

## Functions

### get_json_path()
Dosyanın tam yolunu bulur

### save_to_disk(full_data)
Veriyi diske yazar ve bilgi verir

### check_internet_connection()
### scrape_match_data(match)
Tek bir maç için veriyi çeker.

### main()
