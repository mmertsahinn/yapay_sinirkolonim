# Module: la_liga_odds_ozel.py

LA LIGA ÖZEL ODDS YÜKLEME
- Takım isimlerini karşılaştırır
- Eşleşmeyen takımları tespit eder
- Neden yüklenemediğini analiz eder
- Özel eşleştirme ile yükler

## Functions

### normalize_team_name(name)
Takım ismini normalize eder - ÇOK KAPSAMLI

### team_name_similarity(name1, name2)
İki takım ismi arasındaki benzerlik skoru

### parse_date_safe(date_str, time_str)
Tarih parse eder

### safe_float(value)
String değeri float'a çevirir

### safe_int(value)
String değeri int'e çevirir

### get_db_teams_for_la_liga(session)
La Liga'daki tüm takımları DB'den al

### get_csv_teams_from_la_liga_files()
La Liga CSV dosyalarındaki tüm takım isimlerini topla

### analyze_team_mismatches()
Takım isimlerini karşılaştır ve eşleşmeyenleri bul

### find_match_ultra_flexible(session, league_id, home_team_name, away_team_name, match_date, home_score, away_score)
DB'de maçı bulur - ULTRA ESNEK (La Liga için özel)
Returns: (Match, confidence, reason)

### load_la_liga_odds_special()
La Liga için özel odds yükleme

