"""
EN İYİ LoRA'LAR EXPORTERı
============================

ÖNEMLİ: Bu modül "MASTER_CONTEXT_RULES.md" kurallarına sıkı sıkıya bağlıdır.
Her klasör kendi bağlamında değerlendirilir.

Her çalıştırmada:
1. Mucize LoRA'ları kopyala
2. Aktif en iyi N LoRA'yı kaydet
3. Okunabilir liste oluştur

Klasör: en_iyi_loralar/
"""

import os
import torch
import shutil
from datetime import datetime
from typing import List, Dict
from .top_score_calculator import TopScoreCalculator
from .advanced_score_calculator import AdvancedScoreCalculator


class TopLoRAExporter:
    """
    ⭐ EN İYİ LoRA'LAR EXPORTERı (LIVE SYNC VERSION)
    ================================================
    
    Her çalıştırmada:
    1. Hedef klasörleri TEMİZLER (Live Sync!)
    2. Kriterlere uyan LoRA'ları kopyalar (.pt + .txt)
    3. Her klasöre _SIRALAMA_LISTESI.txt oluşturur
    
    Klasör: best_loras/
    """
    
    def __init__(self, export_dir: str = "best_loras"):
        self.export_dir = export_dir
        
        # Alt Klasörler
        self.dirs = {
            'global': os.path.join(export_dir, "🏆_GLOBAL_TOP50"),
            'young': os.path.join(export_dir, "👶_GENC_YETENEKLER"),
            'veteran': os.path.join(export_dir, "👴_EFSON_VETERANLAR"),
            'einstein': os.path.join(export_dir, "🧠_EINSTEIN"),
            'teams': os.path.join(export_dir, "⚽_TAKIMLAR"),
            'h2h': os.path.join(export_dir, "⚔️_H2H_RIVALS")  # 🆕 H2H KLASÖRÜ
        }
        
        # Klasörleri oluştur
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
        print(f"⭐ Top LoRA Exporter (Live Sync) başlatıldı: {export_dir}")
    
    def export_all(self, population: List, miracle_system, match_count: int, 
                   all_loras_ever: Dict = None, top_n: int = 50, collective_memory=None):
        """
        Tüm kategorileri export et (CANLI SENKRONİZASYON!)
        """
        print(f"\n{'⭐'*40}")
        print(f"CANLI LoRA SENKRONİZASYONU BAŞLIYOR...")
        print(f"{'⭐'*40}")
        
        if all_loras_ever is None:
            all_loras_ever = {}
            
        # Tüm LoRA'ları listeye çevir (Kolay işlem için)
        all_loras_list = []
        for lid, info in all_loras_ever.items():
            lora = info['lora']
            # Skorları önceden hesapla
            adv_score = AdvancedScoreCalculator.calculate_advanced_score(lora, match_count)
            all_loras_list.append({
                'lora': lora,
                'info': info,
                'adv_score': adv_score,
                'age': info.get('age', 0)
            })
            
        # 1. GLOBAL TOP 50 (Genel En İyiler)
        self._sync_category(
            category_key='global',
            loras=all_loras_list,
            sort_key=lambda x: x['adv_score'],
            filter_func=lambda x: True, # Hepsi aday
            top_n=top_n,
            match_count=match_count,
            title="TÜM ZAMANLARIN EN İYİLERİ"
        )
        
        # 2. GENÇ YETENEKLER (Yaş < 50, Potansiyel > 0.6)
        self._sync_category(
            category_key='young',
            loras=all_loras_list,
            sort_key=lambda x: x['adv_score'], # Şimdilik skora göre, ilerde potansiyele göre olabilir
            filter_func=lambda x: x['age'] < 50 and getattr(x['lora'], '_lazarus_lambda', 0) > 0.6,
            top_n=20,
            match_count=match_count,
            title="GENÇ YETENEKLER (<50 Maç)"
        )
        
        # 3. EFSANE VETERANLAR (Yaş > 100, Skor > 0.7)
        self._sync_category(
            category_key='veteran',
            loras=all_loras_list,
            sort_key=lambda x: x['adv_score'],
            filter_func=lambda x: x['age'] > 100 and x['adv_score'] > 0.6,
            top_n=20,
            match_count=match_count,
            title="EFSANE VETERANLAR (>100 Maç)"
        )
        
        # 4. EINSTEIN (Zeka Küpleri)
        try:
            from lora_system.folder_specific_scorer import folder_specific_scorer
            einstein_sort_key = lambda x: folder_specific_scorer.calculate_score_for_folder(x['lora'], "EINSTEIN")
        except ImportError:
            # Fallback: Advanced score kullan
            einstein_sort_key = lambda x: x['adv_score']
        
        self._sync_category(
            category_key='einstein',
            loras=all_loras_list,
            sort_key=einstein_sort_key,
            filter_func=lambda x: True,
            top_n=15,
            match_count=match_count,
            title="EINSTEIN (Yüksek Potansiyel)"
        )
        
        # 5. TAKIM UZMANLARI (Özel Klasörler!)
        self._sync_teams(all_loras_list, match_count, collective_memory)
        
        # 6. H2H RIVALS (İkili İlişkiler!)
        if collective_memory:
            self._sync_h2h(all_loras_list, match_count, collective_memory)
        
        print(f"\n✅ CANLI SENKRONİZASYON TAMAMLANDI!")
        print(f"   📂 Klasörler güncellendi: {self.export_dir}")
        print(f"{'⭐'*40}\n")

    def _sync_h2h(self, loras: List[Dict], match_count: int, collective_memory):
        """
        H2H Klasörlerini Yönet
        """
        try:
            from lora_system.folder_specific_scorer import folder_specific_scorer
        except ImportError:
            # Fallback: H2H özelliği devre dışı
            print("   ⚠️ folder_specific_scorer modülü bulunamadı, H2H klasörleri atlanıyor")
            return
        
        # 1. Önemli H2H çiftlerini bul (En az 3 maç yapılmış)
        pairs = set()
        for match_data in collective_memory.memory.values():
            info = match_data['match_info']
            # Alfabetik sıra ile tuple yap (TeamA, TeamB)
            pair = tuple(sorted([info['home'], info['away']]))
            pairs.add(pair)
            
        base_h2h_dir = self.dirs['h2h']
        
        for team1, team2 in pairs:
            # Bu çift için kaç maç var?
            # (Basitçe kontrol et, folder_specific_scorer zaten 3 maç altını eliyor)
            
            # Klasör adı: TeamA_vs_TeamB
            folder_name = f"{team1}_vs_{team2}".replace(" ", "_")
            h2h_dir = os.path.join(base_h2h_dir, folder_name)
            
            # Adayları puanla
            candidates = []
            for item in loras:
                score = folder_specific_scorer.calculate_h2h_score(item['lora'], team1, team2, collective_memory)
                if score > 0.6: # Sadece başarılı olanlar!
                    candidates.append((item, score))
            
            if not candidates:
                continue # Kimse başarılı değilse klasör açma
                
            # Klasör oluştur ve temizle
            os.makedirs(h2h_dir, exist_ok=True)
            self._clear_folder(h2h_dir)
            
            # Sırala ve Kaydet (Top 5)
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:5]
            
            for rank, (item, score) in enumerate(top_candidates, 1):
                # Detayları al
                details = folder_specific_scorer.get_h2h_details(item['lora'], team1, team2, collective_memory)
                
                self._save_lora_pair(
                    h2h_dir, 
                    item['lora'], 
                    rank, 
                    item['info'], 
                    match_count, 
                    f"H2H_{folder_name}",
                    extra_details=details # 🆕 Ekstra detaylar!
                )
                
            # Liste oluştur
            self._create_ranking_list(h2h_dir, [c[0] for c in top_candidates], f"{team1} vs {team2} UZMANLARI", match_count)

    def _sync_category(self, category_key: str, loras: List[Dict], sort_key, filter_func, top_n: int, match_count: int, title: str):
        """
        Bir kategori için klasörü temizle ve yeniden doldur
        """
        target_dir = self.dirs[category_key]
        
        # 1. TEMİZLİK (Live Sync!)
        self._clear_folder(target_dir)
        
        # 2. FİLTRELE VE SIRALA
        candidates = [l for l in loras if filter_func(l)]
        sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
        top_list = sorted_candidates[:top_n]
        
        # 3. KAYDET (.pt + .txt)
        for rank, item in enumerate(top_list, 1):
            self._save_lora_pair(target_dir, item['lora'], rank, item['info'], match_count, category_key)
            
        # 4. LİSTE OLUŞTUR
        self._create_ranking_list(target_dir, top_list, title, match_count)
        
        print(f"   ✅ {title}: {len(top_list)} dosya senkronize edildi.")

    def _sync_teams(self, loras: List[Dict], match_count: int, collective_memory=None):
        """
        Takım klasörlerini yönet (DEEP SCAN VERSION)
        
        Eski mantık: Sadece 'specialization' etiketi olanları kontrol et.
        YENİ MANTIK: Tüm LoRA'ları, tarihteki TÜM takımlar için tara!
        
        Neden?
        "General" etiketli bir LoRA, Manchester maçlarında %100 yapıyor olabilir.
        Onu kaçırmamak için herkesi her takım için puanlıyoruz.
        """
        if not collective_memory:
            return

        # 1. Tarihteki tüm takımları bul
        all_teams = set()
        for match_data in collective_memory.memory.values():
            info = match_data['match_info']
            all_teams.add(info['home'])
            all_teams.add(info['away'])
            
        base_team_dir = self.dirs['teams']
        try:
            from lora_system.folder_specific_scorer import folder_specific_scorer
            use_folder_scorer = True
        except ImportError:
            # Fallback: Advanced score kullan
            use_folder_scorer = False
            print("   ⚠️ folder_specific_scorer modülü bulunamadı, takım skorları için advanced score kullanılıyor")
        
        print(f"   🔍 DEEP SCAN: {len(all_teams)} takım için {len(loras)} LoRA taranıyor...")
        
        for team in all_teams:
            # Klasör adı (Boşlukları _ yap)
            safe_team_name = team.replace(" ", "_")
            team_dir = os.path.join(base_team_dir, safe_team_name)
            
            # Adayları Puanla (HERKES ADAYDIR!)
            scored_experts = []
            for expert in loras:
                # Orijinal objeyi bozma
                expert_copy = expert.copy()
                
                # Bu LoRA'nın bu takımdaki skorunu hesapla
                if use_folder_scorer:
                    local_score = folder_specific_scorer.calculate_score_for_folder(
                        expert['lora'], team, match_count, collective_memory
                    )
                else:
                    # Fallback: Advanced score kullan (specialization kontrolü ile)
                    specialization = getattr(expert['lora'], 'specialization', None)
                    if specialization and team.lower() in specialization.lower():
                        local_score = expert['adv_score'] * 1.2  # Takım uzmanıysa bonus
                    else:
                        local_score = expert['adv_score'] * 0.8  # Değilse ceza
                
                # Sadece kayda değer olanları al (Eşik: 0.4)
                # Çöp LoRA'larla listeyi doldurmayalım.
                if local_score > 0.4:
                    expert_copy['local_score'] = local_score
                    scored_experts.append(expert_copy)
            
            # Eğer hiç uzman yoksa klasör açma
            if not scored_experts:
                continue
                
            # Klasörü oluştur ve temizle
            os.makedirs(team_dir, exist_ok=True)
            self._clear_folder(team_dir)
            
            # Sırala (Local Score'a göre)
            sorted_experts = sorted(
                scored_experts, 
                key=lambda x: x['local_score'],
                reverse=True
            )
            
            # Top 10 Kaydet
            top_experts = sorted_experts[:10]
            for rank, item in enumerate(top_experts, 1):
                # .pt kaydederken orijinal objeyi kullan
                self._save_lora_pair(team_dir, item['lora'], rank, item['info'], match_count, f"Team_{team}")
                
            # Liste (Local Score ile!)
            self._create_ranking_list(team_dir, top_experts, f"{team} UZMANLARI", match_count)

    def _clear_folder(self, folder_path: str):
        """Klasörün içini tamamen boşalt"""
        if not os.path.exists(folder_path):
            return
            
        for f in os.listdir(folder_path):
            f_path = os.path.join(folder_path, f)
            try:
                if os.path.isfile(f_path):
                    os.remove(f_path)
            except Exception as e:
                print(f"⚠️ Silme hatası: {f} - {e}")

    def _save_lora_pair(self, folder: str, lora, rank: int, info: Dict, match_count: int, category: str):
        """
        .pt ve .txt çiftini kaydet
        Dosya adı: LoRA_Name_ID (SABİT!)
        """
        # Dosya adı (Rütbe YOK, sadece kimlik!)
        safe_name = lora.name.replace(" ", "_")
        base_filename = f"{safe_name}_{lora.id}"
        
        pt_path = os.path.join(folder, f"{base_filename}.pt")
        txt_path = os.path.join(folder, f"{base_filename}.txt")
        
        # 1. .pt Kaydet
        torch.save({
            'lora_params': lora.get_all_lora_params(),
            'metadata': {
                'id': lora.id,
                'name': lora.name,
                'rank': rank, # Rütbe metadata içinde!
                'category': category,
                'exported_at': datetime.now().isoformat()
            }
        }, pt_path)
        
        # 2. .txt Kaydet (Detaylı Bilgi Kartı)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"👤 KİMLİK KARTI: {lora.name}\n")
            f.write(f"{'='*40}\n")
            f.write(f"🏆 ŞU ANKİ RÜTBE: #{rank}\n")
            f.write(f"📂 KATEGORİ: {category}\n")
            f.write(f"{'-'*40}\n")
            
            # Durum
            status = "YAŞIYOR 🟢" if info['alive'] else f"ÖLDÜ 💀 (Maç #{info.get('death_match')})"
            f.write(f"Durum: {status}\n")
            f.write(f"Yaş: {info.get('age', 0)} maç\n")
            f.write(f"Nesil: {lora.generation}\n")
            f.write(f"Uzmanlık: {getattr(lora, 'specialization', 'Genel')}\n")
            
            # Skorlar
            adv_score = AdvancedScoreCalculator.calculate_advanced_score(lora, match_count)
            f.write(f"\n📊 PERFORMANS:\n")
            f.write(f"   • Advanced Score: {adv_score:.3f}\n")
            f.write(f"   • Fitness: {info['final_fitness']:.3f}\n")
            
            # Fizik
            lazarus = getattr(lora, '_lazarus_lambda', 0.5)
            f.write(f"\n🧬 FİZİK MOTORU:\n")
            f.write(f"   • Lazarus Potansiyeli: {lazarus:.3f}\n")
            f.write(f"   • Langevin Sıcaklığı: {getattr(lora, '_langevin_temp', 0.0):.4f}\n")
            
            # Mizaç
            f.write(f"\n🧠 MİZAÇ:\n")
            for k, v in lora.temperament.items():
                f.write(f"   • {k}: {v:.2f}\n")

    def _create_ranking_list(self, folder: str, loras: List[Dict], title: str, match_count: int):
        """
        Klasör için ana sıralama listesi oluştur
        """
        list_path = os.path.join(folder, "_SIRALAMA_LISTESI.txt")
        
        with open(list_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"📜 {title}\n")
            f.write(f"📅 Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Maç #{match_count})\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"{'NO':<4} | {'DURUM':<8} | {'İSİM':<30} | {'SKOR':<8} | {'YAŞ':<5} | {'UZMANLIK'}\n")
            f.write(f"{'-'*80}\n")
            
            for i, item in enumerate(loras, 1):
                lora = item['lora']
                info = item['info']
                # Varsa yerel skoru kullan, yoksa genel skoru
                score = item.get('local_score', item['adv_score'])
                
                status = "🟢" if info['alive'] else "💀"
                age = info.get('age', 0)
                spec = getattr(lora, 'specialization', '-')
                
                f.write(f"#{i:<3} | {status:<8} | {lora.name:<30} | {score:<8.3f} | {age:<5} | {spec}\n")



# Global instance
top_lora_exporter = TopLoRAExporter()

