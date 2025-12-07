# Module: la_liga_odds_skor_eslesme.py

LA LIGA ODDS YÜKLEME - SKOR VE TAKIM EŞLEŞMESİ ÖNCELİKLİ
Tarih toleransı çok geniş, skor ve takım eşleşmesi öncelikli

## Functions

### normalize_team_name(name)
Takım ismini normalize eder

### team_name_similarity(name1, name2)
İki takım ismi arasındaki benzerlik skoru

### parse_date_safe(date_str, time_str)
Tarih parse eder

### safe_float(value)
### safe_int(value)
### find_match_by_score_and_teams(session, league_id, home_team_name, away_team_name, home_score, away_score, match_date)
SKOR VE TAKIM EŞLEŞMESİ ÖNCELİKLİ
- Önce skor + takım eşleşmesi ara (tarih çok geniş)
- Sonra sadece takım eşleşmesi (skor yoksa)

### load_la_liga_odds_by_score()
La Liga için skor öncelikli odds yükleme

