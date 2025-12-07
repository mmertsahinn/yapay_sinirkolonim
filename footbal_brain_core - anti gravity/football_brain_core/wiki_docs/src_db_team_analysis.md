# Module: src\db\team_analysis.py

Takım Analizi ve İlişki Takibi
- Her takımın kendi pattern'lerini öğrenir
- Takım çiftlerinin ikili ilişkilerini analiz eder
- Her hatadan sonra bu bilgileri günceller

## Classes

### TeamAnalyzer
Takım bazlı analiz ve öğrenme.
Her takımın pattern'lerini ve takım çiftlerinin ilişkilerini öğrenir.

#### Methods
- **analyze_team_patterns**(self, team_id, matches, market_type)
  - Bir takımın belirli bir market için pattern'lerini analiz eder.

- **analyze_team_pair_relationship**(self, team_a_id, team_b_id, matches, market_type)
  - İki takım arasındaki ikili ilişkiyi analiz eder.
Hangi takım hangi takıma karşı nasıl performans gösteriyor.

- **get_all_team_patterns**(self, season, market_types)
  - Tüm takımların pattern'lerini analiz eder.

- **get_all_team_relationships**(self, season, market_types)
  - Tüm takım çiftlerinin ilişkilerini analiz eder.

