# Module: src\db\repositories.py

## Classes

### LeagueRepository
#### Methods
- **get_or_create**(session, name, country, code)

- **get_by_id**(session, league_id)

- **get_by_name**(session, name)

- **get_all**(session)

### TeamRepository
#### Methods
- **get_or_create**(session, name, league_id, code)

- **get_by_id**(session, team_id)

- **get_by_league**(session, league_id)

- **get_all**(session)

- **get_by_name**(session, name)

- **get_by_name_and_league**(session, name, league_id)
  - Takım ismini ve lig ID'sini kullanarak takımı bulur (isim eşleştirmesi esnek)

### MatchRepository
#### Methods
- **get_or_create**(session, match_id, league_id, home_team_id, away_team_id, match_date, home_score, away_score, status)

- **get_by_id**(session, match_id)

- **get_by_date_range**(session, date_from, date_to)

- **get_by_league_and_season**(session, league_id, season)

- **get_team_matches**(session, team_id, limit)

### StatRepository
#### Methods
- **create**(session, match_id, team_id, stat_type, stat_value)

- **get_by_match**(session, match_id)

### MarketRepository
#### Methods
- **get_or_create**(session, name, description)

- **get_all**(session)

### PredictionRepository
#### Methods
- **create**(session, match_id, market_id, predicted_outcome, model_version_id, p_hat)

- **get_by_match**(session, match_id)

- **get_by_model_version**(session, model_version_id)

### ResultRepository
#### Methods
- **get_or_create**(session, match_id, market_id, actual_outcome)

- **get_by_match**(session, match_id)

### ModelVersionRepository
#### Methods
- **create**(session, version, description)

- **get_active**(session)

- **get_by_version**(session, version)

- **deactivate_all**(session)

### ExperimentRepository
#### Methods
- **create**(session, experiment_id, config, period_start, period_end, metrics)

- **get_by_id**(session, experiment_id)

- **get_all**(session)

### ExplanationRepository
#### Methods
- **create**(session, match_id, market_id, llm_output, summary_stats)

- **get_by_match**(session, match_id)

### ErrorCaseRepository
PRD: Error Inbox - Hatalı tahminlerin saklandığı repository

#### Methods
- **create**(session, match_id, market_id, predicted_outcome, actual_outcome, model_version_id, llm_comment, user_note)

- **get_unresolved**(session)

- **get_by_cluster**(session, cluster_id)

### ErrorClusterRepository
PRD: Hata Cluster'ları - Benzer hataların gruplandığı repository

#### Methods
- **create**(session, cluster_name, error_summary, league_id, market_id, feature_vector)

- **get_unresolved**(session)

- **update_resolution**(session, cluster_id, resolution_level, root_cause)

### HumanFeedbackRepository
PRD: Kullanıcı geri bildirimleri repository

#### Methods
- **create**(session, error_cluster_id, question)

- **answer**(session, feedback_id, user_answer, suggested_features, action_taken)

### EvolutionPlanRepository
PRD: Evrim planları repository

#### Methods
- **create**(session, error_cluster_id, plan_type, description, suggested_changes)

- **get_pending**(session)

