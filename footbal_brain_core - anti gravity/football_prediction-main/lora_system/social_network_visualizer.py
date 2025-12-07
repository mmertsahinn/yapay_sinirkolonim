"""
🕸️ SOSYAL AĞ GÖRSELLEŞTİRİCİSİ
================================

Bu modül, Akışkan Sosyal Ağ'ın (RAM'deki verinin)
görselleştirilebilir formatlara (CSV, GEXF) dökülmesini sağlar.

ÇIKTI DOSYALARI (evolution_logs/social_network/):
1. nodes.csv: LoRA'lar (ID, Name, Fitness, Generation, Rank)
2. edges.csv: Bağlar (Source, Target, Weight, Type)
3. network_snapshot_X.gexf: Gephi için zaman damgalı snapshot

KULLANIM:
    visualizer.export_snapshot(social_network, population, match_idx)
"""

import os
import csv
import networkx as nx
from typing import List, Dict
from datetime import datetime

class SocialNetworkVisualizer:
    def __init__(self, output_dir: str = "evolution_logs/social_network"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"🕸️ Social Network Visualizer hazır: {output_dir}")

    def export_snapshot(self, 
                        social_network, 
                        population: List, 
                        match_idx: int):
        """
        Gephi uyumlu GEXF ve basit CSV çıktıları üretir.
        """
        if not social_network.bonds:
            return

        # 1. Graph Oluştur (NetworkX)
        G = nx.Graph()

        # Nodes ekle
        for lora in population:
            # Gelinlik (Centrality) hesaplanabilir ama Gephi zaten yapar
            tes_type = getattr(lora, '_tes_scores', {}).get('lora_type', 'Unknown')
            specialization = getattr(lora, 'specialization', 'None')
            
            G.add_node(lora.id, 
                       label=lora.name,
                       fitness=lora.get_recent_fitness(),
                       generation=lora.generation,
                       specialization=specialization,
                       tes_type=tes_type)

        # Edges ekle
        edge_count = 0
        for (id1, id2), strength in social_network.bonds.items():
            if strength > 0.1: # Sadece anlamlı bağlar
                G.add_edge(id1, id2, weight=strength)
                edge_count += 1

        # 2. GEXF Kaydet (Gephi için en iyisi)
        gexf_path = os.path.join(self.output_dir, f"network_match_{match_idx}.gexf")
        nx.write_gexf(G, gexf_path)
        
        # 3. CSV Kaydet (Excel için)
        self._export_csvs(G, match_idx)

        print(f"   🕸️ Sosyal Ağ Snapshot (Maç #{match_idx}): {len(population)} Node, {edge_count} Edge")
        print(f"      📄 {gexf_path}")

    def _export_csvs(self, G, match_idx):
        # Nodes CSV
        nodes_path = os.path.join(self.output_dir, f"nodes_match_{match_idx}.csv")
        with open(nodes_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Id', 'Label', 'Fitness', 'Generation', 'Specialization', 'TesType']) 
            for node, attrs in G.nodes(data=True):
                writer.writerow([
                    node, 
                    attrs.get('label', ''),
                    f"{attrs.get('fitness', 0):.3f}",
                    attrs.get('generation', 0),
                    attrs.get('specialization', ''),
                    attrs.get('tes_type', '')
                ])
        
        # Edges CSV
        edges_path = os.path.join(self.output_dir, f"edges_match_{match_idx}.csv")
        with open(edges_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Source', 'Target', 'Weight', 'Type'])
            for u, v, attrs in G.edges(data=True):
                writer.writerow([u, v, f"{attrs.get('weight', 0):.3f}", 'Undirected'])

    def export_mentor_tree(self, social_network, population: List, match_idx: int):
        """Mentörlük ağacını raporlar (Append mode)"""
        mentorships = getattr(social_network, 'mentorships', {})
        if not mentorships:
            return

        report_path = os.path.join(self.output_dir, "mentorship_report.txt")
        
        # Name lookup
        id_to_name = {l.id: l.name for l in population}
        
        try:
            with open(report_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{'='*60}\n")
                f.write(f"🎓 MENTÖRLÜK RAPORU - MAÇ #{match_idx} - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"{'='*60}\n")
                
                # Group by mentor
                mentor_map = {} # mentor_id -> [apprentice_ids]
                for app_id, mentor_id in mentorships.items():
                    if mentor_id not in mentor_map:
                        mentor_map[mentor_id] = []
                    mentor_map[mentor_id].append(app_id)
                
                f.write(f"• Toplam Mentör: {len(mentor_map)}\n")
                f.write(f"• Toplam Çırak: {len(mentorships)}\n\n")
                
                f.write("🌳 MENTÖRLÜK AĞACI:\n")
                for mentor_id, apprentices in mentor_map.items():
                    mentor_name = id_to_name.get(mentor_id, f"Unknown({mentor_id[:8]})")
                    f.write(f"  👨‍🏫 {mentor_name}\n")
                    for app_id in apprentices:
                         app_name = id_to_name.get(app_id, f"Unknown({app_id[:8]})")
                         f.write(f"      └─ 👶 {app_name}\n")
                f.write("\n")
                
            print(f"   🎓 Mentörlük raporu güncellendi: {report_path}")
        except Exception as e:
            print(f"⚠️ Mentörlük raporu yazılamadı: {e}")
