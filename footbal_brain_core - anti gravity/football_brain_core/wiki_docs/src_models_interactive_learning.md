# Module: src\models\interactive_learning.py

Etkileşimli Öğrenme ve Evrim Mekanizması
- Model yanlış tahmin yaptığında nedenini düşünür
- LLM ile yorum yapar
- Mantıklı sebep bulamazsa kullanıcıya sorar
- Kullanıcıyla karşılıklı fikir alışverişi yapar
- Sürekli birlikte evrilir

## Classes

### InteractiveLearning
Model ile kullanıcı arasında etkileşimli öğrenme mekanizması.
Hataları analiz eder, LLM ile yorum yapar, gerekirse kullanıcıya sorar.

#### Methods
- **__init__**(self, llm_client)

- **analyze_mistake_and_think**(self, match_id, market_type, predicted, actual, predicted_proba, match_context, summary_stats)
  - Yanlış tahminin nedenini düşünür ve yorum yapar.
Mantıklı sebep bulamazsa kullanıcıya sorar.

- **_ask_llm_why_wrong**(self, match_context, predicted, actual, summary_stats, error_analysis)
  - LLM'e neden yanlış olduğunu sorar

- **_evaluate_reasoning_quality**(self, llm_reasoning, error_analysis)
  - LLM'in cevabının mantıklı olup olmadığını değerlendirir

- **_generate_user_question**(self, match_context, predicted, actual, error_analysis)
  - Kullanıcıya sorulacak soruyu oluşturur

- **_extract_learning_points**(self, llm_reasoning, error_analysis)
  - LLM'in analizinden öğrenme noktalarını çıkarır

- **process_user_feedback**(self, match_id, market_type, user_feedback, error_analysis)
  - Kullanıcıdan gelen geri bildirimi işler ve öğrenir.

- **_extract_learning_from_feedback**(self, user_feedback, error_analysis)
  - Kullanıcı geri bildiriminden öğrenme noktalarını çıkarır

- **apply_learned_knowledge**(self, match_context, current_features)
  - Öğrenilen bilgileri mevcut tahminlere uygular.

- **get_learning_summary**(self)
  - Öğrenme özetini döndürür

