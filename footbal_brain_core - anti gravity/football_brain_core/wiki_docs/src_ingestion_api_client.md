# Module: src\ingestion\api_client.py

## Classes

### APIFootballClient
#### Methods
- **__init__**(self, api_key, base_url, on_limit_warning)

- **_make_request**(self, endpoint, params)

- **_check_and_notify_limit**(self, remaining)
  - Limit durumunu kontrol et ve gerekirse bildirim gönder

- **_notify_limit_exceeded**(self)
  - Limit dolduğunda bildirim gönder

- **_notify_limit_low**(self, remaining)
  - Limit azaldığında uyarı gönder

- **_send_windows_notification**(self, title, message)
  - Windows toast bildirimi gönder

- **get_leagues**(self, country, season, league_id)

- **get_teams**(self, league_id, season, team_id)

- **get_fixtures**(self, league_id, season, date_from, date_to, team_id, fixture_id, last)

- **get_fixture_events**(self, fixture_id)

- **get_fixture_statistics**(self, fixture_id)

- **get_standings**(self, league_id, season)

- **get_odds**(self, league_id, season, date_from, date_to, fixture_id, bookmaker)

- **get_league_ids_for_season**(self, season)

