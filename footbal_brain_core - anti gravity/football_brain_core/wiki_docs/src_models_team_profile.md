# Module: src\models\team_profile.py

Takım Profili Sistemi
Her takımın en ince ayrıntısına kadar öğrenilmesi:
- Form döngüleri, güçlü/zayıf yönler
- Ev sahibi/deplasman pattern'leri
- Market bazlı davranışlar
- Zaman bazlı trendler

## Classes

### TeamProfile
Bir takımın detaylı profili - her şeyi ezberinde tutar.

#### Methods
- **__init__**(self, team_id)

- **build_comprehensive_profile**(self, matches, market_types)
  - Takımın kapsamlı profilini oluştur - en ince ayrıntısına kadar.

- **_build_market_profile**(self, matches, market_type, session)
  - Bir market için detaylı profil

- **_analyze_detailed_patterns**(self, matches, venue, session)
  - Çok detaylı pattern analizi

- **_analyze_form_cycles**(self, matches, session)
  - Form döngülerini analiz et

- **_identify_strengths_weaknesses**(self)
  - Güçlü ve zayıf yönleri belirle

- **_analyze_trends**(self, matches, session)
  - Zaman bazlı trendler

- **_team_won**(self, match)
  - Takım kazandı mı?

- **_calculate_detailed_stats**(self, matches, session)
  - En ince ayrıntılar

### TeamProfileManager
Tüm takım profillerini yönetir - her takım için detaylı profil tutar.

#### Methods
- **__init__**(self)

- **get_or_create_profile**(self, team_id)
  - Takım profili al veya oluştur

- **build_all_profiles**(self, season, market_types)
  - Tüm takımların profillerini oluştur

