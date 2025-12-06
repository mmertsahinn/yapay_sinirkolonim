"""
KRİTİK VERİLERİ YEDEKLE!

En önemli:
1. collective_memory (ortak hafıza)
2. lora_population_state.pt (tüm state)
3. meta_lora_state.pt
4. replay_buffer.joblib
"""
import shutil
import os
from datetime import datetime

# Backup klasörü
backup_dir = f"KRITIK_YEDEK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(backup_dir, exist_ok=True)

print(f"📂 BACKUP KLASÖRÜ: {backup_dir}")
print(f"{'='*80}")

# 1. En kritik: State (ortak hafıza dahil!)
if os.path.exists('lora_population_state.pt'):
    shutil.copy2('lora_population_state.pt', os.path.join(backup_dir, 'lora_population_state.pt'))
    print(f"✅ lora_population_state.pt (ORTAK HAFIZA DAHİL!)")

# 2. Meta-LoRA
if os.path.exists('meta_lora_state.pt'):
    shutil.copy2('meta_lora_state.pt', os.path.join(backup_dir, 'meta_lora_state.pt'))
    print(f"✅ meta_lora_state.pt")

# 3. Replay Buffer
if os.path.exists('replay_buffer.joblib'):
    shutil.copy2('replay_buffer.joblib', os.path.join(backup_dir, 'replay_buffer.joblib'))
    print(f"✅ replay_buffer.joblib")

# 4. En iyi LoRA'lar klasörü
if os.path.exists('en_iyi_loralar'):
    shutil.copytree('en_iyi_loralar', os.path.join(backup_dir, 'en_iyi_loralar'), dirs_exist_ok=True)
    print(f"✅ en_iyi_loralar/ (Top LoRA'lar + Mucizeler)")

# 5. Wallet'lar (OPTIONAL - çok büyük!)
wallet_backup = input("\n📔 Wallet'ları da yedekle? (200+ dosya, büyük!) (y/n): ")
if wallet_backup.lower() == 'y':
    if os.path.exists('lora_wallets'):
        shutil.copytree('lora_wallets', os.path.join(backup_dir, 'lora_wallets'), dirs_exist_ok=True)
        print(f"✅ lora_wallets/ (200+ wallet)")

# 6. Evolution logs
if os.path.exists('evolution_logs'):
    shutil.copytree('evolution_logs', os.path.join(backup_dir, 'evolution_logs'), dirs_exist_ok=True)
    print(f"✅ evolution_logs/ (Excel, summary)")

print(f"\n{'='*80}")
print(f"✅ BACKUP TAMAMLANDI!")
print(f"📂 Klasör: {backup_dir}")
print(f"\n💾 EN ÖNEMLİ DOSYA:")
print(f"   → {os.path.join(backup_dir, 'lora_population_state.pt')}")
print(f"   → İçinde: ORTAK HAFIZA (500+ maç bilgisi)")
print(f"{'='*80}")



