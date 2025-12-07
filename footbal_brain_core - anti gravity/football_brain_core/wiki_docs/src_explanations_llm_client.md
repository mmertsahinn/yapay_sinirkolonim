# Module: src\explanations\llm_client.py

## Classes

### LLMClient
#### Methods
- **__init__**(self, provider, api_key, base_url)

- **generate_explanation**(self, match_context, predicted_outcomes, summary_stats)

- **_build_prompt**(self, match_context, predicted_outcomes, summary_stats)

- **_call_openrouter**(self, prompt)
  - OpenRouter API çağrısı - PRD'de belirtilen openai/gpt-oss-20b:free modeli

- **_call_openai**(self, prompt)

- **_call_grok**(self, prompt)

- **_fallback_explanation**(self, match_context, predicted_outcomes)

