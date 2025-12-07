# Module: json_hype_detayli.py

## Functions

### signal_handler(sig, frame)
Terminal kapatılırken çalışan handler

### get_scraper()
Global scraper nesnesini döndür (thread-safe)

### get_json_path()
JSON dosyasının yolunu döndürür.

### save_to_disk(full_data)
Veriyi diske yazar.

### scrape_match_data(match)
Tek bir maç için hype verisini detaylı şekilde çeker (olmuyorsa atlar, tekrar denemez).

### parse_arguments()
### main()
