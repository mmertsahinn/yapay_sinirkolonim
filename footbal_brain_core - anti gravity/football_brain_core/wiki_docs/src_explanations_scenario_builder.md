# Module: src\explanations\scenario_builder.py

## Classes

### ScenarioBuilder
#### Methods
- **__init__**(self, llm_client, use_both_models)
  - Args:
    llm_client: Özel LLM client (None ise varsayılan kullanılır)
    use_both_models: True ise hem GPT hem Grok kullanılır ve karşılaştırılır

- **build_summary_stats**(self, match, session)

- **generate_explanation**(self, match, predicted_outcomes, market_types)
  - Her iki modeli de çalıştırır ve sonuçları karşılaştırır

Returns:
    Dict[MarketType, Dict] - Her market için:
        - "gpt_explanation": GPT modelinin açıklaması
        - "grok_explanation": Grok modelinin açıklaması
        - "gpt_time": GPT'nin süresi (saniye)
        - "grok_time": Grok'un süresi (saniye)
        - "best_model": En hızlı model ("gpt" veya "grok")
        - "explanation": Kullanılacak açıklama (en hızlı olan)

- **save_explanations**(self, match, explanations, summary_stats)

