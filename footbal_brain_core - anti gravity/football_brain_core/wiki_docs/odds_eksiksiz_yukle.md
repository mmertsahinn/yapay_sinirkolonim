# Module: odds_eksiksiz_yukle.py

ODDS EKSİKSİZ YÜKLEME SİSTEMİ
CSV dosyalarındaki TÜM maçları eksiksiz yükler
- Sürekli tekrar dener
- Eşleştirme algoritmasını sürekli iyileştirir
- Eksik 1 maç bile bırakmaz

## Functions

### normalize_team_name(name)
Takım ismini normalize eder - çok kapsamlı

### team_name_similarity(name1, name2)
İki takım ismi arasındaki benzerlik skoru (0-1)

### parse_date(date_str, time_str)
DD/MM/YYYY formatındaki tarihi parse eder

### safe_float(value)
String değeri float'a çevirir

### safe_int(value)
String değeri int'e çevirir

### find_match_in_db_advanced(session, league_id, home_team_name, away_team_name, match_date, home_score, away_score)
DB'de maçı bulur - ÇOK GELİŞMİŞ EŞLEŞTİRME
Returns: (Match, confidence_score)

### load_odds_from_csv(csv_file, session)
CSV dosyasından odds yükler - GELİŞTİRİLMİŞ

### continuous_odds_loading()
SÜREKLI ÇALIŞAN ODDS YÜKLEME - Eksik kalmayana kadar

