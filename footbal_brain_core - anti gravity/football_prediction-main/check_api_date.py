import requests

headers = {'x-apisports-key': '647f5de88a29d150a9d4e2c0c7b636fb'}
url = 'https://v3.football.api-sports.io/fixtures'

# 2025-11-09 kontrol
params = {'date': '2025-11-09'}
r = requests.get(url, headers=headers, params=params)
data = r.json()

print(f"Tarih: 2025-11-09")
print(f"Status: {r.status_code}")
print(f"Maç sayısı: {len(data.get('response', []))}")

if len(data.get('response', [])) > 0:
    print("\nİlk 5 maç:")
    for f in data.get('response', [])[:5]:
        print(f"  {f['teams']['home']['name']} vs {f['teams']['away']['name']}")
else:
    print("\n❌ Hiç maç yok!")
    print("\nAPI yanıtı:")
    print(data)



