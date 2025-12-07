# Module: src\reporting\learning_notebook_excel.py

PRD: Excel Öğrenme Defteri
Her maç + market için:
- Lig, tarih, takımlar, skor
- Market tipi ve seçilen outcome etiketi
- Doğru/yanlış bilgisi (renk/flags)
- Basit form özetleri (son 5 maç puan ort., gol farkı vb.)
- LLM senaryosu (kısa metin)

## Classes

### LearningNotebookExporter
PRD: Excel Öğrenme Defteri
Modelin öğrenme sürecini takip etmek için detaylı Excel raporu

#### Methods
- **__init__**(self, output_dir)

- **export_learning_notebook**(self, date_from, date_to, league_ids)
  - PRD formatında Excel Öğrenme Defteri oluştur

- **_get_form_summary**(self, team_id, match_date, session)
  - Takımın son 5 maç form özeti

- **_format_learning_notebook**(self, file_path)
  - Excel dosyasını formatla (renkler, fontlar)

