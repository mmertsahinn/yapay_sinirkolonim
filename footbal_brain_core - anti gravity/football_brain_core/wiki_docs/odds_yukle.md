# Module: odds_yukle.py

Odds CSV dosyalarını okuyup database'e yükler.
Her lig için odds klasöründeki CSV dosyalarını okur ve MatchOdds tablosuna ekler.

## Functions

### parse_date(date_str, time_str)
DD/MM/YYYY formatındaki tarihi parse eder

### safe_float(value)
String değeri float'a çevirir, hata durumunda None döner

### parse_odds_row(row, match_date)
CSV satırından odds bilgilerini çıkarır

### clean_team_name(name)
Takım ismindeki skor pattern'ini ve gereksiz karakterleri temizler

### normalize_team_name_for_matching(name)
Takım ismini eşleştirme için normalize eder

### find_team_with_flexible_matching(session, team_name, league_id)
Esnek takım eşleştirmesi yapar - GELİŞTİRİLMİŞ VERSİYON

### find_match_in_db(session, home_team_name, away_team_name, match_date, league_code)
Database'de maçı bulur - ÖNCE TARİH, SONRA TAKIM İSİMLERİ

### load_odds_from_csv(csv_path, session)
CSV dosyasından odds'ları yükler

### load_all_odds()
Tüm odds klasörlerindeki CSV dosyalarını yükler

