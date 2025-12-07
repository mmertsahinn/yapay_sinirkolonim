# Module: src\reporting\backtest_excel.py

Backtest sonuçlarını Excel'e export et
PRD'ye uygun: Test sonuçları, doğruluk, hata analizleri

## Classes

### BacktestExcelExporter
Backtest sonuçlarını ve model performansını Excel'e export eder.
PRD'ye uygun: Her maç için tahmin, gerçek, doğruluk, LLM açıklamaları.

#### Methods
- **__init__**(self, config)

- **export_backtest_results**(self, backtest_results, date_from, date_to, model_version_id)
  - Backtest sonuçlarını Excel'e export et.

Args:
    backtest_results: Backtester'dan gelen sonuçlar
    date_from: Başlangıç tarihi
    date_to: Bitiş tarihi
    model_version_id: Model versiyonu ID

- **export_model_performance**(self, evaluation_results, model_version)
  - Model performans metriklerini Excel'e export et.

- **_format_backtest_excel**(self, file_path)
  - Backtest Excel'ini formatla

- **_format_performance_excel**(self, file_path)
  - Performans Excel'ini formatla

