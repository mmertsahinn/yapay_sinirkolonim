# 🚀 HIZLI BAŞLANGIÇ

## Sistem Sıfırlandı!

### ✅ Korunan Veriler:
- `en_iyi_loralar/⭐_AKTIF_EN_IYILER/` (50 LoRA)
- `en_iyi_loralar/top_lora_list.txt`
- `lora_wallets/` (7005 wallet - İSİM_ID.txt)

### 🗑️ Temizlenenler:
- Loglar (evolution_logs/)
- State dosyaları (.pt, .joblib)
- Hibernated LoRA'lar
- Mucizeler

---

## 🧟 Diriltme (Top 50'den):

```bash
# Tüm Top 50'yi dirilt
python emergency_resurrect_all.py --target 0

# Veya spawn ile 250'ye tamamla
python spawn_diverse_population.py
```

---

## 🎯 Sistem Başlat:

```bash
# 500 maç çalıştır
python run_evolutionary_learning.py 2025_temmuz_sonrasi_TAKVIM.csv 2025_temmuz_sonrasi_SONUCLAR.csv --max-matches 500
```

---

## 🌊 Yeni Özellikler:

1. **Input Dim: 78** (60 base + 3 proba + 15 tarihsel)
2. **Ortak hafızadan dinamik veri** (gol, form, hype, H2H)
3. **Takım uzmanlıkları** (Top 5 per team)
4. **Genel uzmanlıklar** (Top 10 global)
5. **Çoklu uzmanlık = Ölümsüzlük** (5+ = %90)
6. **Spesifik scoreboard** (Manchester skorları sadece Manchester maçlarında)


