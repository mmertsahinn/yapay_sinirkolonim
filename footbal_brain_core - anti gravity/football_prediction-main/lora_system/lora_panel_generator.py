"""
📊 LoRA PANEL GENERATOR
=======================

Bu script, mevcut popülasyonun durumunu özetleyen şık bir Markdown paneli oluşturur.
Kullanıcı bu paneli açarak LoRA'ların durumunu anlık takip edebilir.
"""

import os
import pandas as pd
from datetime import datetime

class LoRAPanelGenerator:
    def __init__(self, log_dir="evolution_logs"):
        self.log_dir = log_dir
        self.panel_file = os.path.join(log_dir, "LORA_PANEL.md")
        
    def generate_panel(self, population, match_count, nature_thermostat=None):
        """
        Markdown paneli oluştur
        """
        if not population:
            return
            
        # En iyileri seç
        top_loras = sorted(population, key=lambda x: x.get_recent_fitness(), reverse=True)[:10]
        
        # Markdown içeriği
        md = f"# 🧬 LoRA EVRİM PANELİ (Maç #{match_count})\n"
        md += f"**Son Güncelleme:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 1. DOĞA DURUMU (Entropi & Sıcaklık)
        if nature_thermostat:
            temp = nature_thermostat.temperature
            status = "🔥 SICAK (Agresif)" if temp > 0.7 else ("❄️ SOĞUK (Pasif)" if temp < 0.3 else "🌿 ILIK (Dengeli)")
            md += "## 🌡️ Doğa Durumu\n"
            md += f"- **Sıcaklık:** {temp:.2f} ({status})\n"
            md += f"- **Hedef Entropi:** {nature_thermostat.target_entropy}\n"
            md += f"- **Zorluk Çarpanı:** x{nature_thermostat.get_difficulty_multiplier():.2f}\n\n"
            
        # 2. LİDER TABLOSU (Top 10)
        md += "## 🏆 Lider Tablosu (Top 10)\n"
        md += "| Sıra | İsim | ID | Fitness | Yaş | Gen | Mizaç | Etiketler |\n"
        md += "|---|---|---|---|---|---|---|---|\n"
        
        for i, lora in enumerate(top_loras, 1):
            fitness = lora.get_recent_fitness()
            age = match_count - lora.birth_match
            tags = lora.get_status_tags()
            tag_str = " ".join(tags) if tags else "-"
            
            # Mizaç özeti
            temp = lora.temperament
            if temp['independence'] > 0.7: mizaç = "Bağımsız"
            elif temp['social_intelligence'] > 0.7: mizaç = "Sosyal"
            elif temp['contrarian_score'] > 0.7: mizaç = "Karşıt"
            else: mizaç = "Dengeli"
            
            md += f"| {i} | **{lora.name}** | `{lora.id[:6]}` | **{fitness:.3f}** | {age} | {lora.generation} | {mizaç} | {tag_str} |\n"
            
        md += "\n"
        
        # 3. YÜKSELEN YILDIZLAR (High Lazarus)
        rising_stars = [l for l in population if getattr(l, '_lazarus_lambda', 0) > 0.7]
        rising_stars = sorted(rising_stars, key=lambda x: getattr(x, '_lazarus_lambda', 0), reverse=True)[:5]
        
        if rising_stars:
            md += "## 🌟 Yükselen Yıldızlar (Yüksek Potansiyel)\n"
            md += "| İsim | Lazarus Λ | Fitness | Etiketler |\n"
            md += "|---|---|---|---|\n"
            for lora in rising_stars:
                lazarus = getattr(lora, '_lazarus_lambda', 0)
                tags = lora.get_status_tags()
                tag_str = " ".join(tags)
                md += f"| {lora.name} | **{lazarus:.3f}** | {lora.get_recent_fitness():.3f} | {tag_str} |\n"
            md += "\n"
            
        # 4. TRAVMATİK VAKALAR (High Fear)
        traumatized = [l for l in population if l.temperament.get('fear', 0) > 0.7]
        if traumatized:
            md += "## 🚑 Travmatik Vakalar (Rehabilitasyon Gerekebilir)\n"
            md += "| İsim | Korku Seviyesi | Resilience | Durum |\n"
            md += "|---|---|---|---|\n"
            for lora in traumatized:
                fear = lora.temperament.get('fear', 0)
                res = lora.temperament.get('resilience', 0)
                md += f"| {lora.name} | 😨 {fear:.2f} | 🛡️ {res:.2f} | ⚠️ Riskli |\n"
        
        # Dosyayı yaz
        with open(self.panel_file, 'w', encoding='utf-8') as f:
            f.write(md)
            
        print(f"📊 Panel güncellendi: {self.panel_file}")
