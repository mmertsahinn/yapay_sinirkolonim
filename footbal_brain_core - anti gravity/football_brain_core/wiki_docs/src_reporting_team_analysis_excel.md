# Module: src\reporting\team_analysis_excel.py

Takım Analizi Excel Export
- Her takımın pattern'leri
- Takım çiftlerinin ilişkileri
- Her iterasyondan sonra güncellenmiş analizler

## Classes

### TeamAnalysisExcelExporter
Takım analizlerini Excel'e export eder.
Her iterasyondan sonra güncellenmiş takım pattern'leri ve ilişkileri.

#### Methods
- **__init__**(self, config)

- **export_team_patterns**(self, team_patterns, iteration, season)
  - Takım pattern'lerini Excel'e export et.

- **export_team_relationships**(self, relationships, iteration, season)
  - Takım ilişkilerini Excel'e export et.

- **export_comprehensive_analysis**(self, team_patterns, relationships, iteration, season, error_summary)
  - Kapsamlı analiz Excel'i: Pattern'ler + İlişkiler + Hata analizi

- **_format_team_excel**(self, file_path)
  - Takım Excel'ini formatla

- **_format_relationship_excel**(self, file_path)
  - İlişki Excel'ini formatla

- **_format_comprehensive_excel**(self, file_path)
  - Kapsamlı Excel'i formatla

