# Module: odds_surekli_calis.py

ODDS SÜREKLI ÇALIŞAN SİSTEM - ASLA DURMAZ
- Her hata verdiğinde bildirim verir
- Satır işleme hatalarını yakalar ve çözer
- Sürekli kontrol eder ve düzeltir

## Functions

### normalize_team_name(name)
Takım ismini normalize eder

### team_name_similarity(name1, name2)
İki takım ismi arasındaki benzerlik skoru

### parse_date_safe(date_str, time_str)
Tarih parse eder - GÜVENLİ VERSİYON

### safe_float(value)
String değeri float'a çevirir - GÜVENLİ

### safe_int(value)
String değeri int'e çevirir - GÜVENLİ

### find_match_in_db_advanced(session, league_id, home_team_name, away_team_name, match_date, home_score, away_score)
DB'de maçı bulur - ÇOK GELİŞTİRİLMİŞ (30 gün tolerans)

### process_csv_row_safe(row, csv_file, row_num, session)
CSV satırını işler - GÜVENLİ VERSİYON
Returns: {'success': bool, 'match_id': int, 'odds_added': bool, 'error': str}

### continuous_odds_loading()
SÜREKLI ÇALIŞAN ODDS YÜKLEME - ASLA DURMAZ

