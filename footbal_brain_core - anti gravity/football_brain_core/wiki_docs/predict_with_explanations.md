# Module: predict_with_explanations.py

Tahmin yap ve LLM ile açıklama üret - PRD'ye uygun tam workflow

## Functions

### predict_and_explain(model_path, days_ahead, market_types)
Gelecek maçlar için tahmin yap ve LLM ile açıklama üret

Args:
    model_path: Eğitilmiş model dosyası yolu (None ise aktif model kullanılır)
    days_ahead: Kaç gün ileriye tahmin yapılacak
    market_types: Hangi marketler için tahmin yapılacak

