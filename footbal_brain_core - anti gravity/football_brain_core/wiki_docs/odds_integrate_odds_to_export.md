# Module: odds\integrate_odds_to_export.py

Odds CSV dosyalarını okuyup football_brain_export.json dosyasına entegre eden script.
Tüm liglerden odds verilerini toplayıp JSON formatına çevirir ve mevcut export dosyasına ekler.

## Functions

### parse_date(date_str, time_str)
CSV'deki tarih formatını parse et

### normalize_team_name(name)
Takım ismini normalize et

### convert_odds_value(value)
Odds değerini float'a çevir

### read_all_csv_files(odds_dir)
Tüm CSV dosyalarını oku ve birleştir

### process_match_row(row, league_name, division)
Bir maç satırını işle ve JSON formatına çevir

### normalize_team_name_for_match(team_name)
Takım ismini match_id formatına uygun hale getir - GELİŞTİRİLMİŞ

### normalize_team_name_fuzzy(team_name)
Fuzzy matching için daha agresif normalizasyon

### team_name_similarity(name1, name2)
İki takım ismi arasındaki benzerlik skoru (0-1)

### create_match_lookup_key(match)
Match objesinden lookup key oluştur

### create_odds_lookup_key(odds_item)
Odds item'ından lookup key oluştur - matches ile eşleştirmek için

### integrate_to_export(odds_data, export_path)
Odds verilerini export JSON'a entegre et - HER MAÇ İÇİNE DİREKT EKLENİR

### main()
Ana fonksiyon

