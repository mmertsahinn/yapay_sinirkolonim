# Module: hype_surekli_calis.py

HYPE SÜREKLI ÇALIŞAN SİSTEM - ASLA DURMAZ
- Her hata verdiğinde bildirim verir
- Hataları otomatik çözer
- Sürekli kontrol eder ve düzeltir

## Functions

### ensure_hype_columns()
Hype kolonlarının var olduğundan emin ol

### get_hype_status(session)
Hype durumunu kontrol eder

### validate_hype_data(match)
Hype verilerinin geçerli olup olmadığını kontrol eder

### fetch_hype_for_match_safe(match, scraper, session)
Bir maç için hype verilerini çeker - GÜVENLİ VERSİYON
Returns: (success, error_message)

### fetch_hype_for_match_safe_cached(match, scraper, session, league, home_team, away_team)
Bir maç için hype verilerini çeker - CACHE'Lİ VERSİYON (daha hızlı)
Returns: (success, error_message)

### continuous_hype_fetch()
SÜREKLI ÇALIŞAN HYPE ÇEKME SİSTEMİ - MAXIMUM GÜÇ KULLANIMI

