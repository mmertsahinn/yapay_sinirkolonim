"""
🏗️ TAKIM KLASÖRLERINI OTOMATIK OLUŞTUR
========================================

CSV'deki maçlardan tüm takımları al ve klasör yapısını oluştur:
- Her takım için ana klasör
- Her takım için uzmanlık alt klasörleri (Win, Goal, Hype)
- Her takımın rakipleri için VS klasörleri
- Her klasörde formüllü TXT dosyaları

Sistem çalıştıkça bu klasörler dolacak!
"""

import os
import pandas as pd
from datetime import datetime
from collections import defaultdict


def get_all_teams_from_csv(csv_path: str = "2025_temmuz_sonrasi_TAKVIM.csv"):
    """CSV'den tüm takımları çıkar"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except:
        df = pd.read_csv(csv_path, encoding='latin-1')
    
    teams = set()
    team_opponents = defaultdict(set)  # Her takımın rakipleri
    
    for _, row in df.iterrows():
        home = row.get('home_team', '')
        away = row.get('away_team', '')
        
        if pd.notna(home) and home:
            teams.add(home)
            if pd.notna(away) and away:
                team_opponents[home].add(away)
        
        if pd.notna(away) and away:
            teams.add(away)
            if pd.notna(home) and home:
                team_opponents[away].add(home)
    
    return teams, team_opponents


def safe_team_name(team_name: str) -> str:
    """Dosya sistemi için güvenli takım ismi"""
    return team_name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('.', '_').replace(':', '').replace('*', '').replace('?', '').replace('"', '').replace('<', '').replace('>', '').replace('|', '')


def create_formul_txt(file_path: str, team_name: str, spec_type: str, opponent: str = None):
    """Formül açıklamalı TXT dosyası oluştur"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        
        if spec_type == 'WIN':
            f.write(f"🎯 {team_name.upper()} - WIN EXPERTS TOP 5\n")
        elif spec_type == 'GOAL':
            f.write(f"⚽ {team_name.upper()} - GOAL EXPERTS TOP 5\n")
        elif spec_type == 'HYPE':
            f.write(f"🔥 {team_name.upper()} - HYPE EXPERTS TOP 5\n")
        elif spec_type == 'VS':
            f.write(f"🆚 {team_name.upper()} VS {opponent.upper()} - EXPERTS TOP 5\n")
        
        f.write("=" * 100 + "\n")
        f.write(f"Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Maç: 0 (Henüz veri yok)\n")
        f.write("=" * 100 + "\n\n")
        
        # FORMÜL AÇIKLAMASI
        f.write("📐 UZMANLIK SKORU FORMÜLÜ:\n")
        f.write("=" * 100 + "\n")
        
        if spec_type in ['WIN', 'HYPE', 'VS']:
            f.write("SKOR = Accuracy (30%) + Age (20%) + Consistency (15%) +\n")
            f.write("       Peak (15%) + Momentum (10%) + Match Count (10%)\n\n")
            f.write("• Accuracy: Doğru tahmin yüzdesi (SADECE bu takımın maçlarında!)\n")
            f.write("• Age: LoRA'nın deneyimi (yaş normalizasyonu)\n")
            f.write("• Consistency: Son 20 maçtaki istikrar (SADECE bu takımda!)\n")
            f.write("• Peak: En iyi 10 maçlık dönem başarısı (SADECE bu takımda!)\n")
            f.write("• Momentum: İlk yarı vs İkinci yarı trend (SADECE bu takımda!)\n")
            
            if spec_type == 'VS':
                f.write("• Match Count: Bu eşleşme için tahmin sayısı bonusu (Min: 5 maç)\n\n")
            else:
                f.write("• Match Count: Bu takım için tahmin sayısı bonusu (Min: 20 maç)\n\n")
        
        elif spec_type == 'GOAL':
            f.write("SKOR = Accuracy (30%) + Age (20%) + Consistency (15%) +\n")
            f.write("       Peak (15%) + Momentum (10%) + Match Count (10%)\n\n")
            f.write("• Accuracy: MAE (Mean Absolute Error) bazlı (SADECE bu takımın gollerinde!)\n")
            f.write("  - MAE 0.0 → 1.0 skor (mükemmel!)\n")
            f.write("  - MAE 3.0 → 0.0 skor (kötü!)\n")
            f.write("• Age: LoRA'nın deneyimi (yaş normalizasyonu)\n")
            f.write("• Consistency: Son 20 maçtaki gol tahmin istikrarı (SADECE bu takımda!)\n")
            f.write("• Peak: En iyi 10 maçlık dönem gol tahmin başarısı (SADECE bu takımda!)\n")
            f.write("• Momentum: İlk yarı vs İkinci yarı gol tahmin trendi (SADECE bu takımda!)\n")
            f.write("• Match Count: Bu takım için gol tahmin sayısı bonusu (Min: 20 maç)\n\n")
        
        f.write("🎯 ÖNEMLİ: Tüm metrikler SADECE bu takımın maçlarına bakıyor!\n")
        if spec_type == 'VS':
            f.write(f"   {team_name} vs {opponent} maçları → Sadece bu eşleşme sayılır!\n")
        else:
            f.write(f"   {team_name} uzmanı → Sadece {team_name} maçları sayılır!\n")
        f.write("=" * 100 + "\n\n")
        
        # MİNİMUM KOŞULLAR
        f.write("📊 MİNİMUM KOŞULLAR:\n")
        f.write("-" * 100 + "\n")
        if spec_type == 'VS':
            f.write("  • Minimum 5 maç gerekli (az eşleşme olur)\n")
        else:
            f.write("  • Minimum 20 maç gerekli\n")
        f.write("  • Top 5 uzman seçilir (eşik yok, sadece en iyiler!)\n")
        f.write("  • Sistem her 50 maçta güncellenir\n\n")
        
        # BOŞ UZMAN LİSTESİ
        f.write("🏆 UZMANLAR:\n")
        f.write("=" * 100 + "\n")
        f.write("Henüz uzman yok. Sistem çalıştıkça bu liste dolacak!\n")
        f.write("(İlk 50-100 maçtan sonra uzmanlar ortaya çıkacak)\n\n")
        f.write("=" * 100 + "\n")


def initialize_all_team_folders(base_dir: str = "en_iyi_loralar/takım_uzmanlıkları"):
    """
    Tüm takım klasörlerini ve alt klasörleri oluştur
    """
    print(f"🏗️ Takım klasörleri oluşturuluyor...")
    print(f"   Hedef: {base_dir}")
    
    # CSV'den takımları al
    teams, team_opponents = get_all_teams_from_csv()
    
    print(f"\n📊 Bulunan takımlar: {len(teams)}")
    print(f"   Örnek: {list(teams)[:5]}")
    
    # Ana klasörü oluştur
    os.makedirs(base_dir, exist_ok=True)
    
    created_folders = 0
    created_txts = 0
    
    # Her takım için
    for team in sorted(teams):
        safe_name = safe_team_name(team)
        team_dir = os.path.join(base_dir, safe_name)
        os.makedirs(team_dir, exist_ok=True)
        created_folders += 1
        
        # 1) WIN EXPERTS
        win_dir = os.path.join(team_dir, "🎯_WIN_EXPERTS")
        os.makedirs(win_dir, exist_ok=True)
        win_txt = os.path.join(win_dir, "🎯_win_experts_top5.txt")
        create_formul_txt(win_txt, team, 'WIN')
        created_folders += 1
        created_txts += 1
        
        # 2) GOAL EXPERTS
        goal_dir = os.path.join(team_dir, "⚽_GOAL_EXPERTS")
        os.makedirs(goal_dir, exist_ok=True)
        goal_txt = os.path.join(goal_dir, "⚽_goal_experts_top5.txt")
        create_formul_txt(goal_txt, team, 'GOAL')
        created_folders += 1
        created_txts += 1
        
        # 3) HYPE EXPERTS
        hype_dir = os.path.join(team_dir, "🔥_HYPE_EXPERTS")
        os.makedirs(hype_dir, exist_ok=True)
        hype_txt = os.path.join(hype_dir, "🔥_hype_experts_top5.txt")
        create_formul_txt(hype_txt, team, 'HYPE')
        created_folders += 1
        created_txts += 1
        
        # 4) VS EXPERTS (Her rakip için)
        opponents = team_opponents.get(team, set())
        for opponent in sorted(opponents):
            safe_opponent = safe_team_name(opponent)
            vs_dir = os.path.join(team_dir, f"🆚_VS_{safe_opponent}")
            os.makedirs(vs_dir, exist_ok=True)
            vs_txt = os.path.join(vs_dir, f"🆚_vs_{safe_opponent.lower()}_top5.txt")
            create_formul_txt(vs_txt, team, 'VS', opponent)
            created_folders += 1
            created_txts += 1
        
        # 5) MASTER TXT (Takım özeti)
        master_txt = os.path.join(team_dir, f"{safe_name}_MASTER.txt")
        with open(master_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"🏆 {team.upper()} - UZMANLIK MASTER RAPORU\n")
            f.write("=" * 100 + "\n")
            f.write(f"Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Maç: 0\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("📐 TAKIM UZMANLIK SİSTEMİ:\n")
            f.write("=" * 100 + "\n")
            f.write("Bu takım için 4 tip uzmanlık kategorisi var:\n\n")
            f.write("1. 🎯 WIN EXPERTS: Bu takımın kazanacağını en iyi tahmin edenler\n")
            f.write("2. ⚽ GOAL EXPERTS: Bu takımın atacağı golleri en iyi tahmin edenler\n")
            f.write("3. 🔥 HYPE EXPERTS: Bu takımın hype'ını en iyi değerlendirenler\n")
            f.write(f"4. 🆚 VS EXPERTS: Bu takımın belirli rakiplerle maçlarını en iyi tahmin edenler ({len(opponents)} rakip)\n\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("📊 DURUM:\n")
            f.write("-" * 100 + "\n")
            f.write("Henüz veri yok. Sistem çalıştıkça bu klasörler dolacak!\n")
            f.write("İlk 50-100 maçtan sonra uzmanlar ortaya çıkacak.\n\n")
        
        created_txts += 1
    
    print(f"\n✅ Tamamlandı!")
    print(f"   📁 Oluşturulan klasör: {created_folders}")
    print(f"   📄 Oluşturulan TXT: {created_txts}")
    print(f"   🏆 Toplam takım: {len(teams)}")
    
    # Özet rapor
    summary_file = os.path.join(base_dir, "📊_KLASOR_YAPISI.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("📊 TAKIM UZMANLIK KLASÖR YAPISI\n")
        f.write("=" * 100 + "\n")
        f.write(f"Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Toplam Takım: {len(teams)}\n")
        f.write(f"Toplam Klasör: {created_folders}\n")
        f.write(f"Toplam TXT: {created_txts}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("TAKIM LİSTESİ:\n")
        f.write("-" * 100 + "\n")
        for i, team in enumerate(sorted(teams), 1):
            opponents = team_opponents.get(team, set())
            f.write(f"{i:3d}. {team:40s} → {len(opponents):2d} rakip\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("\nKLASÖR YAPISI (Her takım için):\n")
        f.write("-" * 100 + "\n")
        f.write("takım_adı/\n")
        f.write("  ├── takım_adı_MASTER.txt\n")
        f.write("  ├── 🎯_WIN_EXPERTS/\n")
        f.write("  │   └── 🎯_win_experts_top5.txt\n")
        f.write("  ├── ⚽_GOAL_EXPERTS/\n")
        f.write("  │   └── ⚽_goal_experts_top5.txt\n")
        f.write("  ├── 🔥_HYPE_EXPERTS/\n")
        f.write("  │   └── 🔥_hype_experts_top5.txt\n")
        f.write("  └── 🆚_VS_rakip1/\n")
        f.write("      └── 🆚_vs_rakip1_top5.txt\n")
        f.write("\n" + "=" * 100 + "\n")
    
    print(f"\n📄 Özet rapor: {summary_file}")


if __name__ == "__main__":
    print("=" * 100)
    print("🏗️ TAKIM KLASÖRLERINI OLUŞTUR")
    print("=" * 100)
    print()
    
    initialize_all_team_folders()
    
    print("\n" + "=" * 100)
    print("✅ TÜM KLASÖRLER HAZIR!")
    print("=" * 100)
    print()
    print("Şimdi run_evolutionary_learning.py çalıştırabilirsin.")
    print("Sistem her 50 maçta bu klasörleri dolduracak!")

