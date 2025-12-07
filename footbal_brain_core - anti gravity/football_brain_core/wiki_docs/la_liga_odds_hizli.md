# Module: la_liga_odds_hizli.py

LA LIGA HIZLI ODDS YÜKLEME - Sadece yükleme, analiz yok

## Functions

### normalize_team_name(name)
Takım ismini normalize eder

### team_name_similarity(name1, name2)
İki takım ismi arasındaki benzerlik skoru

### parse_date_safe(date_str, time_str)
Tarih parse eder

### safe_float(value)
String değeri float'a çevirir

### safe_int(value)
String değeri int'e çevirir

### find_match_fast(session, league_id, home_team_name, away_team_name, match_date, home_score, away_score)
Hızlı maç bulma

### load_la_liga_odds_fast()
La Liga için hızlı odds yükleme

