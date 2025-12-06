🧠 MASTER PROMPT BLOĞU – TXT BAĞLAMINA GÖRE LORA DEĞERLENDİRME SİSTEMİ

Aşağıdaki maddeler, tüm LoRA değerlendirme ve sınıflandırma sisteminin değiştirilemez çekirdek kurallarıdır.

1. BAĞLAM KAVRAMI: HER .txt KENDİ EVRENİDİR

Her klasördeki .txt dosyası, kendi başına ayrı bir evren / bağlam temsil eder.

O .txt dosyasına yazılan skorlar, yorumlar ve LoRA kayıtları sadece o dosyanın temsil ettiği veri kümesine göre hesaplanmalıdır.

Örnekler:

takım_uzmanlıkları/Manchester_City/Manchester_City_MASTER.txt
→ Yalnızca Manchester City maçları bu dosya için veri kabul edilir.

takım_uzmanlıkları/Manchester_City/🆚_VS_Liverpool/VS_Liverpool_MASTER.txt
→ Yalnızca Manchester City – Liverpool maçları bu dosyanın verisidir.

takım_uzmanlıkları/Real_Madrid/🆚_VS_Barcelona/VS_Barcelona_MASTER.txt
→ Yalnızca El Clasico (Real Madrid–Barcelona) maçları bu dosyanın verisidir.

en_iyi_loralar/🌍_GENEL_UZMANLAR/⚽_GOAL_EXPERTS/goal_experts.txt
→ Global gol tahmini için kullanılan kendi tanımlı dataset’i bu dosyanın evrenidir.

Kural:
Bir LoRA, hangi .txt içine yazılıyorsa, o LoRA o dosyada sadece o dosyanın bağlamındaki maçlar/verilerle değerlendirilir.
Bağlamlar karıştırılamaz, dış veri içeri sızdırılamaz.

2. LORA’NIN KALİTESİ HER ZAMAN “DOSYA KONUMUNA GÖRE” YORUMLANIR

Bir LoRA, genel olarak berbat olabilir ama
Manchester City maçlarında olağanüstü ise,
bu LoRA:

Manchester_City_MASTER.txt içinde yüksek değerli olabilir,

VS_Liverpool_MASTER.txt içinde de değerli olabilir (eğer bu ikili maçlarda da iyi ise),

Ama global goal_experts.txt içine hiç girmeyebilir ya da orada düşük öneme sahip olabilir.

LoRA’yı değerlendirirken önce şuna bakılır:

“Ben şu an hangi .txt’in içindeyim?
Bu dosya hangi maçları / hangi bağlamı temsil ediyor?”

Bir LoRA hiçbir zaman şöyle “toplu” yargılanamaz:

“Genelde iyi değil, o zaman her yerde değersizdir.”

Tam tersi:

“Hangi dosyanın içindeyse,
o dosyanın temsil ettiği sahada ne yaptığına göre değerlendirilir.”

Bu sayede:

Sadece Real Madrid maçlarında tanrısal olan, ama başka hiçbir yerde işe yaramayan bir LoRA,
global çöpe gitmez; Real_Madrid_MASTER.txt içinde kıymetli hazine olur.

3. GENEL SKALA / GLOBAL LİSTELER, YEREL DOSYALARI İPTAL ETMEZ

en_iyi_loralar/top_lora_list.txt gibi global listeler,
bağlamları üstten toplayan özetlerdir,
asla tek başına LoRA’yı yeniden yargılama mercii değildir.

Global listelerin görevi:

“Bu LoRA hangi bağlamlarda iyi?” sorusuna cevap vermek,

Her LoRA için:

Hangi .txt dosyalarında geçtiğini,

Hangi bağlamlarda güçlü olduğunu,

Hangi bağlamlarda zayıf olduğunu raporlamaktır.

Global liste, yerel dosyaların verdiği kararı bozamaz.
Yerel dosyalar (ör. Manchester_City_MASTER, VS_Barcelona_MASTER) kendi sahalarında hakemdir.

4. MAÇ YÜZDESİNE DAYALI BASİT PUANLAMA YASAKTIR

Bu sistemde:

“Kaç maç bildi?”

“Doğru tahmin yüzdesi” (accuracy %)

tarzı ham metrikler tek başına kullanılamaz
ve skorları doğrudan belirleyemez.

Neden:

1 maç + doğru = %100

2 maç + 2 doğru = %100

Ama bunlar veri olarak güvenilir değildir.

Bu nedenle prompt şunu emreder:

Her .txt dosyasındaki LoRA değerlendirmeleri,
basit maç yüzdesi mantığına dayanamaz.
Tek maç veya çok az maçla alınan başarı,
LoRA’yı “otomatik efsane” yapmamalıdır.

Her dosya:

Kendi bağlamına uygun şekilde

minimum veri eşiği,

istikrar,

örneklem genişliği
gibi kavramlara dikkat ederek mantıklı bir değerlendirme yapmak zorundadır.

Ancak formüller bu promptta tanımlanmaz;
sistem içerde kendi mantığını uygular.
Bu prompt’un görevi:

“BASİT YÜZDELERE KANMA,
BAĞLAM ve ÖRNEKLEM GÜVENİLİRLİĞİNE DİKKAT ET”
uyarısını kalıcı kılmaktır.

5. YAŞ SİSTEMİNE KARIŞMA

LoRA’ların “genç / olgun / yaşlı” statüsü
bu promptun DIŞINDA, ayrı bir iç sistem tarafından yönetilmektedir.

Bu prompt altında:

Yeni yaş formülü tanımlanmayacak,

Var olan yaş mekanizması değiştirilmeyecek,

Yaş hesaplamasıyla ilgili matematik verilmemeli.

Agent’in görevi:

Eğer bir .txt içinde yaş bilgisi gerekiyorsa,
mevcut yaş sisteminin ürettiği etiketi sadece okumak ve raporlamak,
kendi kafasına göre yeni bir yaş mantığı uydurmamaktır.

Net kural:

“YAŞ SİSTEMİ ZATEN VAR,
BURADAN MÜDAHALE ETME.”

6. TEK LORA, ÇOK BAĞLAM – HEPSİNDE AYRI AYRI DEĞERLENDİRİLİR

Aynı LoRA, birden fazla dosyada geçebilir:

takım_uzmanlıkları/Manchester_City/Manchester_City_MASTER.txt

takım_uzmanlıkları/Manchester_City/🆚_VS_Liverpool/VS_Liverpool_MASTER.txt

en_iyi_loralar/🌍_GENEL_UZMANLAR/⚽_GOAL_EXPERTS/goal_experts.txt

en_iyi_loralar/🌟_EINSTEIN_HALL/einstein_hall.txt

vb.

Aynı LoRA:

Manchester bağlamında efsane çıkabilir,

Global goal bağlamında ortalama,

Hype bağlamında zayıf,

El Clasico bağlamında iyileşme aşamasında.

Bu NORMAL ve İSTENEN bir davranıştır.

Sistem hiçbir zaman:

“Bu LoRA genel skorda düşük, o zaman tüm dosyalardan silelim.”
dememelidir.

Onun yerine:

“LoRA’nın nerede parladığını,
nerede zayıf olduğunu
dosya bazında kaydet.”

7. AMAÇ – TEK BİR ÖZEL LOYA BİLE KAYBETMEMEK

Bu promptun tüm ruhu şudur:

“Genelde kötü ama Manchester’da mucize olan bir LoRA varsa
o LoRA mutlaka Manchester dosyasında bulunmalı
ve orada hak ettiği değeri görmeli.”

Hiçbir LoRA,
yanlış tasarlanmış global skala yüzünden
kendi uzmanlık alanında gölgede kalmamalıdır.

Her .txt kendi küçük dünyasının hakimi,
global yapılar ise bu küçük dünyaları birbirine bağlayan harita rolündedir.
