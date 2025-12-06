from lora_system.deep_evaluator import DeepEvaluator

def test_scenario(correct, total, conf, name):
    score = DeepEvaluator.calculate_bayesian_score(correct, total, conf)
    print(f"🧪 {name:<30} | {correct}/{total} ({correct/total*100:.0f}%) | Conf: {conf/total:.2f} | SCORE: {score:.4f}")

print("🧠 DEEP MATH EVALUATOR TEST")
print("===========================")

# Senaryo 1: Şanslı Çaylak vs İstikrarlı Uzman
test_scenario(1, 1, 0.9, "Şanslı Çaylak (1/1)")
test_scenario(9, 10, 9.0, "İstikrarlı Uzman (9/10)")

# Senaryo 2: Yüksek Güven vs Düşük Güven
test_scenario(5, 10, 9.0, "Emin Ama Orta (5/10, High Conf)")
test_scenario(5, 10, 5.0, "Emin Değil (5/10, Low Conf)")

# Senaryo 3: Mükemmeliyet
test_scenario(50, 50, 45.0, "Efsane (50/50)")

# Senaryo 4: Trend Bonusu
history_bad = [False, False, False, True, True] # Sonradan açılan
history_flat = [True, False, True, False, True] # Dalgalı
print(f"\n📈 Trend Bonusu (Sonradan Açılan): {DeepEvaluator.calculate_trend_bonus(history_bad):.4f}")
print(f"📉 Trend Bonusu (Dalgalı): {DeepEvaluator.calculate_trend_bonus(history_flat):.4f}")
