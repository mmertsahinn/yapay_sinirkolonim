# Module: checkpoint_loader.py

Checkpoint sistemi - Son çekilen maçların kaydı
Eksik maç kontrolü için kullanılır

## Classes

### CheckpointManager
Maç çekme ilerlemesini takip eder

#### Methods
- **__init__**(self, checkpoint_file)

- **save_checkpoint**(self, league_name, season, last_match_date, total_matches_loaded, api_requests_used, api_requests_remaining)
  - Checkpoint kaydet

- **load_all_checkpoints**(self)
  - Tüm checkpoint'leri yükle

- **get_checkpoint**(self, league_name, season)
  - Belirli bir lig/sezon için checkpoint al

- **mark_completed**(self, league_name, season)
  - Lig/sezon tamamlandı olarak işaretle

- **print_summary**(self)
  - Checkpoint özetini yazdır

