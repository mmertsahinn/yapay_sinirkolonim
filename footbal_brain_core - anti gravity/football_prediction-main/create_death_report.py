"""
💀 ÖLÜM RAPORU OLUŞTURUCU
==========================

Tüm ölümleri detaylı Excel raporu olarak kaydeder.
"""

import pandas as pd
import json
import os
from datetime import datetime

print(f"{'='*80}")
print(f"💀 ÖLÜM RAPORU OLUŞTURULUYOR")
print(f"{'='*80}\n")

# Evolution events'ten ölümleri çek
all_deaths = []

if os.path.exists('evolution_logs/evolution_data.json'):
    with open('evolution_logs/evolution_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ölüm eventlerini filtrele
    deaths = [e for e in data['events'] if e['type'] == 'death']
    
    print(f"📊 Toplam ölüm eventi: {len(deaths)}")
    
    for death in deaths:
        # Detaylı kayıt
        record = {
            'Ölüm Maçı': death.get('match', 'N/A'),
            'Tarih': death.get('timestamp', 'N/A'),
            'LoRA İsmi': death.get('lora_name', 'N/A'),
            'LoRA ID': death.get('lora_id', 'N/A'),
            'Yaş (Maç)': death.get('age_in_matches', death.get('age', 'N/A')),
            'Yaş (Gün)': death.get('age_days', 'N/A'),
            'Final Fitness': death.get('final_fitness', 'N/A'),
            'Generasyon': death.get('generation', 'N/A'),
            'Ölüm Sebebi': death.get('death_detail', death.get('reason', 'Bilinmiyor')),
            'Dirilme Sayısı': death.get('resurrection_count', 0),
            'Şanslı Kurtuluş': death.get('lucky_survival_count', 0),
            'Şanslı Kurtuldu mu?': 'EVET' if death.get('lucky_survived', False) else 'HAYIR'
        }
        
        all_deaths.append(record)

# Excel'e kaydet
if all_deaths:
    df = pd.DataFrame(all_deaths)
    
    # Sırala (ölüm maçına göre)
    df = df.sort_values('Ölüm Maçı')
    
    # Excel dosyası
    excel_file = f"evolution_logs/OLUM_RAPORU_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Gelişmiş Excel yazımı
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Ana sheet
        df.to_excel(writer, sheet_name='Tüm Ölümler', index=False)
        
        # Ölüm sebepleri özeti
        if 'Ölüm Sebebi' in df.columns:
            reason_summary = df['Ölüm Sebebi'].value_counts().reset_index()
            reason_summary.columns = ['Ölüm Sebebi', 'Sayı']
            reason_summary['Yüzde'] = (reason_summary['Sayı'] / len(df) * 100).round(1)
            reason_summary.to_excel(writer, sheet_name='Sebep Özeti', index=False)
        
        # Maç bazlı özet (hangi maçta kaç LoRA öldü)
        if 'Ölüm Maçı' in df.columns:
            match_summary = df.groupby('Ölüm Maçı').size().reset_index()
            match_summary.columns = ['Maç', 'Ölüm Sayısı']
            match_summary = match_summary.sort_values('Ölüm Sayısı', ascending=False)
            match_summary.to_excel(writer, sheet_name='Maç Bazlı', index=False)
        
        # Generasyon bazlı
        if 'Generasyon' in df.columns:
            gen_summary = df.groupby('Generasyon').size().reset_index()
            gen_summary.columns = ['Generasyon', 'Ölüm Sayısı']
            gen_summary.to_excel(writer, sheet_name='Generasyon Bazlı', index=False)
    
    print(f"\n✅ EXCEL RAPORU OLUŞTURULDU!")
    print(f"{'─'*80}")
    print(f"📁 Dosya: {excel_file}")
    print(f"\n📊 İÇERİK:")
    print(f"   • Tüm Ölümler: {len(df)} kayıt")
    print(f"   • Sebep Özeti: {df['Ölüm Sebebi'].nunique()} farklı sebep")
    print(f"   • Maç Bazlı Analiz")
    print(f"   • Generasyon Bazlı Analiz")
    
    # En çok ölüm olan maçlar
    print(f"\n💀 EN ÇOK ÖLÜM OLAN MAÇLAR:")
    print(f"{'─'*80}")
    match_deaths = df['Ölüm Maçı'].value_counts().head(5)
    for match, count in match_deaths.items():
        print(f"   Maç #{match}: {count} LoRA öldü")
        # O maçtaki ölüm sebepleri
        match_reasons = df[df['Ölüm Maçı'] == match]['Ölüm Sebebi'].unique()
        for reason in match_reasons[:2]:
            print(f"      → {reason}")
    
    # En yaygın ölüm sebepleri
    print(f"\n💀 EN YAYGIN ÖLÜM SEBEPLERİ:")
    print(f"{'─'*80}")
    top_reasons = df['Ölüm Sebebi'].value_counts().head(5)
    for reason, count in top_reasons.items():
        pct = count / len(df) * 100
        print(f"   {count:3d}x ({pct:5.1f}%) - {reason}")
    
    # Yaş istatistikleri
    if 'Yaş (Maç)' in df.columns:
        avg_age = df['Yaş (Maç)'].mean()
        min_age = df['Yaş (Maç)'].min()
        max_age = df['Yaş (Maç)'].max()
        
        print(f"\n⏳ YAŞ İSTATİSTİKLERİ:")
        print(f"{'─'*80}")
        print(f"   Ortalama yaş: {avg_age:.1f} maç (~{avg_age/10:.1f} yaş)")
        print(f"   En genç ölüm: {min_age} maç")
        print(f"   En yaşlı ölüm: {max_age} maç")
    
    print(f"\n{'='*80}")
    print(f"✅ Rapor tamamlandı!")
    print(f"{'='*80}\n")
    
else:
    print(f"⚠️ evolution_data.json bulunamadı!")
    print(f"   Önce maç oynat, sonra rapor oluşturulur.")



