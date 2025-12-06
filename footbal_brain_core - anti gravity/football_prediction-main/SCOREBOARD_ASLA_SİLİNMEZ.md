# ⚠️ KRİTİK KURAL: SCOREBOARD ASLA SİLİNMEZ!

## 🚫 YASAK İŞLEMLER

**ASLA YAPILMAYACAKLAR:**

1. ❌ `shutil.rmtree()` ile scoreboard klasörlerini silmek
2. ❌ `.pt` dosyalarını silmek (sadece güncellenir!)
3. ❌ "Cleanup" scripti çalıştırmak
4. ❌ Manuel dosya silme

---

## ✅ DOĞRU MANTIK

**Scoreboard mantığı:**

- LoRA'lar **asla silinmez**, sadece **yeniden sıralanır**
- Düşük sıralara düşebilir (#1 → #45)
- Ama dosya **hep orada kalır**!
- Yeni LoRA gelirse **dosya sayısı artar**
- Kimse çıkmaz, sadece **eklemeler** olur!

---

## 📂 KORUNAN KLASÖRLER

```
en_iyi_loralar/
├── ⭐_AKTIF_EN_IYILER/       ← ASLA SİLİNMEZ!
├── 🏆_MUCIZELER/              ← ASLA SİLİNMEZ!
├── 🌟_EINSTEIN_HALL/          ← ASLA SİLİNMEZ!
├── 🏛️_NEWTON_HALL/            ← ASLA SİLİNMEZ!
└── 🧬_DARWIN_HALL/            ← ASLA SİLİNMEZ!
```

---

## 🔄 GÜNCELLEME MANTIĞI

```python
# ❌ YANLIŞ:
os.remove(old_file)
torch.save(new_lora, new_file)

# ✅ DOĞRU:
torch.save(lora, f"{lora.id}.pt")  # ID değişmez, dosya sadece güncellenir!
```

---

## 💾 DOSYA İSİMLERİ

**ID bazlı sistem:**

```
abc123.pt  ← ID değişmez, dosya HEP BU!
```

**Rank değişikliği:**

Metadata içinde saklanır:
```python
{
    'rank': 25,        # Yeni sıra
    'old_rank': 12,    # Eski sıra
    'rank_change': -13 # Düşüş!
}
```

Dosya adı **değişmez!** Sadece içindeki `rank` metadatası güncellenir!

---

## ⚡ ÖZET

**KURAL:** Scoreboard = **Sonsuza kadar büyüyen tarihsel kayıt!**

- ✅ Eklemeler olur
- ✅ Sıralama değişir
- ✅ Metadata güncellenir
- ❌ Silme YOK!
- ❌ Azaltma YOK!

**Sonuç:** Dosya sayısı monoton artar! 📈

---

**SİLİNEN DOSYALAR GÖZÜKÜYORSA:**

1. Backup'tan geri yükle
2. Emergency resurrection çalıştır
3. `cleanup_*.py` dosyalarını sil!


