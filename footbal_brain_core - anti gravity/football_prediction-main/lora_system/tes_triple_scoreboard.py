"""
🔬 TES TRIPLE SCOREBOARD (3 Ayrı Hall of Fame!)
================================================

Einstein Hall: Sürpriz uzmanları!
Newton Hall: İstikrar uzmanları!
Darwin Hall: Popülasyon liderleri!

Her biri ayrı dosya, ayrı kullanım!
"""

import os
import torch
from typing import List, Dict
from datetime import datetime


class TESTripleScoreboard:
    """
    3 ayrı scoreboard sistemi
    """
    
    def __init__(self, export_dir: str = "en_iyi_loralar"):
        self.export_dir = export_dir
        
        # 6 AYRI HALL!
        self.einstein_dir = os.path.join(export_dir, "🌟_EINSTEIN_HALL")
        self.newton_dir = os.path.join(export_dir, "🏛️_NEWTON_HALL")
        self.darwin_dir = os.path.join(export_dir, "🧬_DARWIN_HALL")
        self.potansiyel_dir = os.path.join(export_dir, "🌱_POTANSIYEL_HALL")
        self.hybrid_dir = os.path.join(export_dir, "🌈_HYBRID_HALL")  # 🆕 5. HALL! (0.30+)
        self.strong_hybrid_dir = os.path.join(export_dir, "🌟_STRONG_HYBRID_HALL")  # 🆕 6. HALL! (0.50+)
        self.perfect_hybrid_dir = os.path.join(export_dir, "💎_PERFECT_HYBRID_HALL")  # 🆕 7. HALL! (0.75+)
        
        # Klasörleri oluştur
        for directory in [self.einstein_dir, self.newton_dir, self.darwin_dir, 
                         self.potansiyel_dir, self.hybrid_dir, 
                         self.strong_hybrid_dir, self.perfect_hybrid_dir]:
            os.makedirs(directory, exist_ok=True)
        
        print("🔬 TES Scoreboard başlatıldı (7 Hall: Einstein, Newton, Darwin, Potansiyel, Hybrid, Strong Hybrid, Perfect Hybrid!)")
    
    def export_all_types(self, population: List, all_loras_ever: Dict, match_count: int, top_n: int = 15):
        """
        Tüm tipleri export et!
        
        Args:
            population: Yaşayan LoRA'lar
            all_loras_ever: Tüm zamanlar (ölüler dahil!)
            match_count: Maç sayısı
            top_n: Her tipten kaç tane? (15)
        """
        from lora_system.tes_scoreboard import tes_scoreboard
        
        # Tüm LoRA'ları topla (yaşayan + ölü)
        all_loras = []
        
        # Yaşayanlar
        for lora in population:
            tes_data = tes_scoreboard.calculate_tes_score(lora, population, collective_memory=None)
            all_loras.append({
                'lora': lora,
                'alive': True,
                'tes': tes_data,
                'life_energy': getattr(lora, 'life_energy', 1.0)
            })
        
        # Ölüler (TES hesaplamayı dene)
        for lora_id, lora_info in all_loras_ever.items():
            if not lora_info.get('alive', False):
                lora_obj = lora_info.get('lora')
                if lora_obj and hasattr(lora_obj, 'fitness_history'):
                    try:
                        tes_data = tes_scoreboard.calculate_tes_score(lora_obj, population, collective_memory=None)
                        all_loras.append({
                            'lora': lora_obj,
                            'alive': False,
                            'tes': tes_data,
                            'life_energy': 0.0
                        })
                    except Exception as e:
                        print(f"⚠️ {lora_obj.name} ({lora_id}) için TES hesaplanamadı: {e}")
        
        # TİPLERE GÖRE AYIR
        # Not: HYBRID LoRA'lar birden fazla listede olabilir!
        einstein_loras = [l for l in all_loras if 'EINSTEIN' in l['tes']['lora_type'] or 
                         'HYBRID(E' in l['tes']['lora_type'] or 
                         'PERFECT HYBRID' in l['tes']['lora_type']]
        
        newton_loras = [l for l in all_loras if 'NEWTON' in l['tes']['lora_type'] or 
                       'HYBRID(N' in l['tes']['lora_type'] or 
                       'HYBRID(E-N)' in l['tes']['lora_type'] or
                       'PERFECT HYBRID' in l['tes']['lora_type']]
        
        darwin_loras = [l for l in all_loras if 'DARWIN' in l['tes']['lora_type'] or 
                       'HYBRID(D' in l['tes']['lora_type'] or 
                       'HYBRID(E-D)' in l['tes']['lora_type'] or
                       'HYBRID(N-D)' in l['tes']['lora_type'] or
                       'PERFECT HYBRID' in l['tes']['lora_type']]
        
        hybrid_loras = [l for l in all_loras if 'HYBRID' in l['tes']['lora_type']]
        
        # DEBUG: Tip dağılımını yazdır
        print(f"\n📊 TES TİP DAĞILIMI:")
        print(f"   Einstein: {len(einstein_loras)} LoRA")
        print(f"   Newton: {len(newton_loras)} LoRA")
        print(f"   Darwin: {len(darwin_loras)} LoRA")
        print(f"   Hybrid: {len(hybrid_loras)} LoRA")
        
        # Eğer hala boşsa, tüm LoRA'ların tiplerini yazdır (ilk 5)
        if len(einstein_loras) == 0 and len(newton_loras) == 0:
            print(f"\n⚠️ UYARI: Einstein ve Newton boş! İlk 5 LoRA tipi:")
            for i, l in enumerate(all_loras[:5]):
                lora_type = l['tes'].get('lora_type', 'UNKNOWN')
                tes_total = l['tes'].get('total_tes', 0)
                darwin = l['tes'].get('darwin', 0)
                einstein = l['tes'].get('einstein', 0)
                newton = l['tes'].get('newton', 0)
                print(f"     {i+1}. {l['lora'].name}: {lora_type} (TES:{tes_total:.3f}, D:{darwin:.2f}, E:{einstein:.2f}, N:{newton:.2f})")
        
        # 🌱 POTANSIYEL HALL (Genç + Art arda başarılı!)
        potansiyel_loras = []
        for l in all_loras:
            lora = l['lora']
            age = match_count - lora.birth_match
            fitness = lora.get_recent_fitness()
            
            # Streak hesapla
            if len(lora.fitness_history) > 5:
                recent = lora.fitness_history[-10:]  # Son 10 maç
                streak = 0
                for fit in reversed(recent):
                    if fit > 0.5:
                        streak += 1
                    else:
                        break
                
                # POTANSIYEL KRİTERLERİ: Genç + Art arda 5+ + Yüksek fitness
                if age <= 15 and streak >= 5 and fitness >= 0.90:
                    potansiyel_loras.append(l)
        
        # 🌈 HYBRID HİYERARŞİSİ (3 SEVİYE!)
        # Perfect Hybrid (0.75+) > Strong Hybrid (0.50+) > Hybrid (0.30+)
        perfect_hybrid_loras = [l for l in hybrid_loras if 'PERFECT HYBRID💎💎💎' in l['tes']['lora_type']]
        strong_hybrid_loras = [l for l in hybrid_loras if 'STRONG HYBRID🌟🌟' in l['tes']['lora_type']]
        normal_hybrid_loras = [l for l in hybrid_loras if 'HYBRID🌟' in l['tes']['lora_type'] and 'STRONG' not in l['tes']['lora_type'] and 'PERFECT' not in l['tes']['lora_type']]
        
        # Her tipi sırala ve export et
        self._export_type_hall(einstein_loras, 'EINSTEIN⭐', self.einstein_dir, top_n, match_count)
        self._export_type_hall(newton_loras, 'NEWTON🏛️', self.newton_dir, top_n, match_count)
        self._export_type_hall(darwin_loras, 'DARWIN🧬', self.darwin_dir, top_n, match_count)
        self._export_type_hall(potansiyel_loras, 'POTANSIYEL🌱', self.potansiyel_dir, top_n, match_count)
        self._export_type_hall(normal_hybrid_loras, 'HYBRID🌈', self.hybrid_dir, top_n, match_count)  # 5. HALL (0.30+)
        self._export_type_hall(strong_hybrid_loras, 'STRONG HYBRID🌟', self.strong_hybrid_dir, top_n, match_count)  # 🆕 6. HALL (0.50+)
        self._export_type_hall(perfect_hybrid_loras, 'PERFECT HYBRID💎', self.perfect_hybrid_dir, top_n, match_count)  # 🆕 7. HALL (0.75+)
        
        # Özet
        print(f"\n🌈 HYBRID HİYERARŞİSİ (3 SEVİYE!):")
        print(f"   💎 Perfect (0.75+ üçünde): {len(perfect_hybrid_loras)} LoRA")
        print(f"   🌟 Strong (0.50+ üçünde): {len(strong_hybrid_loras)} LoRA")
        print(f"   ⭐ Normal (0.30+ üçünde): {len(normal_hybrid_loras)} LoRA")
    
    def _export_type_hall(self, loras: List[Dict], type_name: str, export_dir: str, top_n: int, match_count: int):
        """
        Bir tip için Hall of Fame oluştur!
        """
        if len(loras) == 0:
            print(f"   {type_name}: Henüz yok")
            return
        
        # TES skoruna göre sırala
        loras.sort(key=lambda x: x['tes']['total_tes'], reverse=True)
        
        # İlk top_n'i al
        top_loras = loras[:top_n]
        
        # Dosyaları kaydet (.pt)
        for idx, lora_data in enumerate(top_loras, start=1):
            lora = lora_data['lora']
            
            # Dosya adı: İsim_ID.pt (wallet ile aynı format)
            filename = f"{lora.name}_{lora.id}.pt"
            filepath = os.path.join(export_dir, filename)
            
            # Kaydet
            torch.save({
                'lora_params': lora.get_all_lora_params(),
                'metadata': {
                    'id': lora.id,
                    'name': lora.name,
                    'tes_scores': lora_data['tes'],
                    'life_energy': lora_data['life_energy'],
                    'alive': lora_data['alive'],
                    'rank': idx,
                    'type': type_name,
                    'exported_at': match_count,
                    # Tüm detaylar!
                    'temperament': getattr(lora, 'temperament', {}),
                    'specialization': getattr(lora, 'specialization', None),
                    'emotional_archetype': getattr(lora, 'emotional_archetype', 'Dengeli'),
                    'physics_archetype': getattr(lora, 'physics_archetype', 'Standart'),
                    'particle_archetype': getattr(lora, '_particle_archetype', 'Unknown'),
                    'langevin_temp': getattr(lora, '_langevin_temp', 0.01),
                    'nose_hoover_xi': getattr(lora, '_nose_hoover_xi', 0.0),
                    'kinetic_energy': getattr(lora, '_kinetic_energy', 0.0),
                    'lazarus_lambda': getattr(lora, '_lazarus_lambda', 0.5),
                    'om_action': getattr(lora, '_om_action', 0.0),
                    'ghost_potential': getattr(lora, '_ghost_potential', 0.0),
                    'reputation_score': getattr(lora, '_reputation_score', 0.0),
                    'generation': lora.generation,
                    'birth_match': lora.birth_match,
                    'parents': getattr(lora, 'parents', []),
                    'offspring_count': getattr(lora, 'offspring_count', 0),
                    'fitness': lora.get_recent_fitness()
                }
            }, filepath)
        
        # 🔄 TXT DOSYASINI .PT DOSYALARINDAN OLUŞTUR!
        # (_create_txt_from_pt_files zaten TXT dosyasını oluşturuyor!)
        self._create_txt_from_pt_files(export_dir, type_name, match_count)
        
        print(f"   {type_name}: {len(top_loras)} LoRA export edildi")
    
    def _create_txt_from_pt_files(self, export_dir: str, type_name: str, match_count: int):
        """
        Klasördeki TÜM .pt dosyalarından txt listesi oluştur!
        Sadece yaşayanlar değil, HERKES!
        """
        # Klasördeki tüm .pt dosyalarını bul
        try:
            pt_files = [f for f in os.listdir(export_dir) if f.endswith('.pt')]
        except FileNotFoundError:
            print(f"⚠️ {export_dir} dizini bulunamadı, oluşturuluyor...")
            os.makedirs(export_dir, exist_ok=True)
            pt_files = []
        
        if len(pt_files) == 0:
            print(f"⚠️ {type_name}: Henüz .pt dosyası yok!")
            # Boş TXT yaz (kafası karışmasın)
            list_file = os.path.join(export_dir, f"{type_name}_hall.txt")
            with open(list_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"{type_name} HALL OF FAME\n")
                f.write("="*80 + "\n")
                f.write(f"Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Maç: {match_count}\n")
                f.write(f"Toplam {type_name}: 0\n")
                f.write("="*80 + "\n\n")
                f.write("⚠️ Henüz bu tipte LoRA bulunamadı!\n")
                f.write("   Sistem evrimleştikçe bu hall dolacak.\n")
                f.write("   (TES skorları arttıkça bu tip ortaya çıkacak)\n")
                f.write("="*80 + "\n")
            return
        
        # Tüm .pt dosyalarını oku ve metadata'larını topla
        all_loras_data = []
        
        for pt_file in pt_files:
            filepath = os.path.join(export_dir, pt_file)
            try:
                data = torch.load(filepath, map_location='cpu')
                metadata = data.get('metadata', {})
                
                # TES skoruna göre sıralama için
                tes_total = metadata.get('tes_scores', {}).get('total_tes', 0.0)
                
                all_loras_data.append({
                    'filename': pt_file,
                    'metadata': metadata,
                    'tes_total': tes_total
                })
            except Exception as e:
                print(f"⚠️ {pt_file} okunamadı: {e}")
        
        # TES skoruna göre sırala
        all_loras_data.sort(key=lambda x: x['tes_total'], reverse=True)
        
        # TXT dosyası oluştur
        list_file = os.path.join(export_dir, f"{type_name}_hall.txt")
        
        with open(list_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"{type_name} HALL OF FAME\n")
            f.write("="*80 + "\n")
            f.write(f"Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Maç: {match_count}\n")
            f.write(f"Toplam {type_name}: {len(all_loras_data)}\n")
            f.write("="*80 + "\n\n")
            
            # Kullanım önerisi
            if 'EINSTEIN' in type_name:
                f.write("💡 KULLANIM: Derbi, Hype, Sürpriz maçlarda bu LoRA'ları dinle!\n")
            elif 'NEWTON' in type_name:
                f.write("💡 KULLANIM: Rutin, güvenilir tahmin için bu LoRA'ları dinle!\n")
            elif 'DARWIN' in type_name:
                f.write("💡 KULLANIM: Genel strateji, popülasyon kararları için dinle!\n")
            
            f.write("\n" + "="*80 + "\n\n")
            
            # HER BİR LORA İÇİN DETAYLI BİLGİ
            for idx, lora_info in enumerate(all_loras_data, start=1):
                meta = lora_info['metadata']
                pt_filename = lora_info['filename']  # .pt dosya adı
                
                # Temel bilgiler
                lora_id = meta.get('id', 'Unknown')
                lora_name = meta.get('name', 'Unknown')
                alive = meta.get('alive', False)
                tes_scores = meta.get('tes_scores', {})
                energy = meta.get('life_energy', 0.0)
                
                # Fizik parametreleri
                langevin_temp = meta.get('langevin_temp', 0.01)
                nose_hoover_xi = meta.get('nose_hoover_xi', 0.0)
                kinetic_energy = meta.get('kinetic_energy', 0.0)
                lazarus_lambda = meta.get('lazarus_lambda', 0.5)
                om_action = meta.get('om_action', 0.0)
                ghost_potential = meta.get('ghost_potential', 0.0)
                
                # Arketipler
                particle_archetype = meta.get('particle_archetype', 'Unknown')
                emotional_archetype = meta.get('emotional_archetype', 'Dengeli')
                physics_archetype = meta.get('physics_archetype', 'Standart')
                
                # Uzmanlık ve itibar
                specialization = meta.get('specialization', 'Genel')
                reputation_score = meta.get('reputation_score', 0.0)
                
                # Nesil bilgileri
                generation = meta.get('generation', 0)
                birth_match = meta.get('birth_match', 0)
                age = match_count - birth_match
                fitness = meta.get('fitness', 0.0)
                parents_count = len(meta.get('parents', []))
                offspring_count = meta.get('offspring_count', 0)
                
                # Mizaç
                temperament = meta.get('temperament', {})
                top_traits = sorted(temperament.items(), key=lambda x: x[1], reverse=True)[:5]
                
                status = "⚡ YAŞIYOR" if alive else "💀 ÖLÜ"
                
                f.write(f"{'='*80}\n")
                f.write(f"#{idx:02d} | {lora_name} | TES:{tes_scores.get('total_tes', 0):.3f}\n")
                f.write(f"📁 Dosya: {pt_filename}\n")
                f.write(f"{'='*80}\n")
                
                # Temel Bilgiler
                f.write(f"📊 TEMEL BİLGİLER:\n")
                f.write(f"   ID: {lora_id}\n")
                f.write(f"   Durum: {status}\n")
                f.write(f"   Yaş: {age} maç\n")
                f.write(f"   Nesil: {generation}\n")
                f.write(f"   Fitness: {fitness:.3f}\n")
                f.write(f"   Ebeveynler: {parents_count} ebeveyn\n")
                f.write(f"   Çocuklar: {offspring_count} çocuk\n")
                f.write("\n")
                
                # TES Skorları
                f.write(f"🔬 TES SKORLARI:\n")
                f.write(f"   Toplam TES: {tes_scores.get('total_tes', 0):.3f}\n")
                f.write(f"   Darwin (Katkı): {tes_scores.get('darwin', 0):.3f}\n")
                f.write(f"   Einstein (Sürpriz): {tes_scores.get('einstein', 0):.3f}\n")
                f.write(f"   Newton (İstikrar): {tes_scores.get('newton', 0):.3f}\n")
                f.write(f"   Tip: {tes_scores.get('lora_type', 'Unknown')}\n")
                f.write("\n")
                
                # Enerji ve Fizik
                f.write(f"⚡ ENERJİ VE FİZİK:\n")
                f.write(f"   Life Energy: {energy:.3f}\n")
                f.write(f"   Langevin Sıcaklık (T): {langevin_temp:.4f}\n")
                f.write(f"   Nosé-Hoover Sürtünme (ξ): {nose_hoover_xi:.4f}\n")
                f.write(f"   Kinetik Enerji: {kinetic_energy:.4f}\n")
                f.write(f"   Lazarus Lambda (Λ): {lazarus_lambda:.3f}\n")
                f.write(f"   Onsager-Machlup (S_OM): {om_action:.3f}\n")
                f.write(f"   Ghost Potansiyel: {ghost_potential:.3f}\n")
                f.write("\n")
                
                # Arketipler
                f.write(f"🎭 ARKETİPLER:\n")
                f.write(f"   Parçacık Arketipi: {particle_archetype}\n")
                f.write(f"   Duygu Arketipi: {emotional_archetype}\n")
                f.write(f"   Fizik Arketipi: {physics_archetype}\n")
                f.write("\n")
                
                # Uzmanlık ve İtibar
                f.write(f"🎯 UZMANLIK VE İTİBAR:\n")
                f.write(f"   Uzmanlık: {specialization if specialization else 'Henüz yok'}\n")
                f.write(f"   İtibar Skoru: {reputation_score:.3f}\n")
                f.write("\n")
                
                # Mizaç (Top 5)
                if top_traits:
                    f.write(f"🎨 MİZAÇ (İlk 5 Özellik):\n")
                    for trait_name, trait_value in top_traits:
                        f.write(f"   {trait_name}: {trait_value:.2f}\n")
                    f.write("\n")
                
                # Dosya Yolu
                f.write(f"💾 DOSYA:\n")
                f.write(f"   {lora_info['filename']}\n")
                f.write("\n")
        
        print(f"   📝 {type_name} txt dosyası güncellendi: {len(all_loras_data)} LoRA")


# Global instance
tes_triple_scoreboard = TESTripleScoreboard()

