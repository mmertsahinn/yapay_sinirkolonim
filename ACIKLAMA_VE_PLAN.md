# Nöronlar ve LoRA'ların Çalışma Prensibi

## 1. LoRA'lar Bilgiyi Nasıl İşler?
Sisteminizdeki "LoRA"lar (Low-Rank Adapter), aslında **Derin Sinir Ağlarıdır (Deep Neural Networks)**. Bilgiyi şu şekilde işlerler:

1.  **Girdi (Sensation):** Maç verisi (takım gücü, hype, form vb.) 78 adet sayısal değere (nöron girişine) dönüştürülür.
2.  **Sinaptik Ağırlıklar (Memory):** Bu 78 değer, LoRA'nın katmanlarındaki ağırlık matrisleri ile çarpılır. Bu ağırlıklar, LoRA'nın "tecrübesidir".
3.  **Aktivasyon (Thinking):** `ReLU` fonksiyonu, insan beynindeki nöronların ateşlenmesi gibidir. Gelen sinyal güçlüyse geçer, zayıfsa sönümlenir.
4.  **Karar (Prediction):** Son katman, 3 ihtimal (Ev Sahibi, Beraberlik, Deplasman) için olasılık üretir.

## 2. Nöronlar Nasıl Çalışır?
Mevcut kodunuzda (`lora_adapter.py`) nöronlar matematiksel olarak şöyle çalışır:
$$ y = W \cdot x + (B \cdot A) \cdot x $$
Burada $W$ donmuş genel bilgiyi, $A$ ve $B$ ise o LoRA'nın **kişisel adaptasyonunu** temsil eder. "Evrim" dediğimiz şey, bu $A$ ve $B$ matrislerinin nesiller boyu iyileşmesidir.

## 3. İnsan Benzeri "Deep Learning" ve "Çağ Atlama" Planı

İnsanlar sadece deneyimle öğrenmez, **başkalarının deneyimini kopyalayarak (kültür/eğitim)** çağ atlarlar. Sistemi şu şekilde güncelliyoruz:

### A. Deep Knowledge Distillation (Bilgi Damıtma - Çağ Atlama)
*   **Mevcut Durum:** LoRA'lar sadece kendi maçlarından öğreniyor.
*   **Yeni Sistem:** Yeni doğan bir LoRA, "Master" bir LoRA'nın (Fitness > 0.9) beynini **Deep Learning (Distillation Loss)** ile kopyalayarak başlayacak. Böylece "bebek" gibi değil, "eğitimli bir yetişkin" gibi doğacak. Bu, **çağ atlamayı** sağlar.

### B. Arka Plan Elek Sistemi (Sieve System)
*   **Mevcut Durum:** Kategorizasyon kural bazlı.
*   **Yeni Sistem:** Arka planda çalışan bir yapay zeka (Clustering), LoRA'ların "hatalarını" analiz edecek. Aynı hatayı yapanları "Aynı Kabileye" koyup, onları topluca eğitecek.

### C. Kaotik Determinizm Kırıcı
*   **Mevcut Durum:** Rastgelelik var ama yapısal değil.
*   **Yeni Sistem:** **Kelebek Etkisi Modülü**. Bir LoRA'nın küçük bir ağırlık değişimi, sosyal ağdaki komşularında dalgalanma (noise injection) yaratacak.

---

## Yapılacak Değişiklikler
1.  `lora_system/deep_learning_optimization.py`: Bilgi transferi (Distillation) modülü eklenecek.
2.  `lora_system/background_sieve.py`: LoRA'ları davranışlarına göre ayıran elek sistemi.
3.  `run_evolutionary_learning.py`: Bu sistemlerin entegrasyonu.
