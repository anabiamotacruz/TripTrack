import requests
import time
import pandas as pd
from urllib.parse import quote

API_KEY = "API_KEY"
BASE_URL = "http://ws.audioscrobbler.com/2.0/"
CACHE = {}

def safe_request(url):
    if url in CACHE:
        return CACHE[url]
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                try:
                    data = resp.json()
                    CACHE[url] = data
                    return data
                except Exception:
                    print(f"⚠️ Erro ao decodificar JSON. Resposta: {resp.text[:100]}")
            else:
                print(f"⚠️ Resposta inválida ({resp.status_code}). Tentando novamente...")
        except Exception as e:
            print(f"⚠️ Erro de conexão: {e}")
        time.sleep(1.5)
    return {}

def get_top_artists(limit=100):
    url = f"{BASE_URL}?method=chart.gettopartists&api_key={API_KEY}&format=json&limit={limit}"
    resp = safe_request(url)
    if "artists" not in resp:
        return []
    return [a["name"] for a in resp["artists"]["artist"]]

def get_top_tracks(artist, limit=20):
    artist_q = quote(artist)
    url = f"{BASE_URL}?method=artist.gettoptracks&artist={artist_q}&api_key={API_KEY}&format=json&limit={limit}"
    resp = safe_request(url)
    if "toptracks" not in resp:
        return []
    return [
        {
            "name": t["name"],
            "url": t.get("url", "")
        } 
        for t in resp["toptracks"]["track"]
    ]

def get_similar_tracks(artist, track, limit=100):
    artist_q = quote(artist)
    track_q = quote(track)
    url = f"{BASE_URL}?method=track.getsimilar&artist={artist_q}&track={track_q}&api_key={API_KEY}&format=json&limit={limit}"
    resp = safe_request(url)
    if "similartracks" not in resp or not resp["similartracks"].get("track"):
        return []
    return [
        {
            "name": t["name"],
            "artist": t["artist"]["name"],
            "url": t.get("url", "")
        }
        for t in resp["similartracks"]["track"]
    ]

def get_track_tags(artist, track, limit=10):
    artist_q = quote(artist)
    track_q = quote(track)
    url = f"{BASE_URL}?method=track.gettoptags&artist={artist_q}&track={track_q}&api_key={API_KEY}&format=json&limit={limit}"
    resp = safe_request(url)
    if "toptags" not in resp or "tag" not in resp["toptags"]:
        return []
    return [tag["name"] for tag in resp["toptags"]["tag"][:limit]]

def main():
    data = []
    artists = get_top_artists()

    for artist in artists:
        print(f"\n🎤 Coletando faixas de {artist}...")
        tracks = get_top_tracks(artist, limit=5)
        for track_info in tracks:
            track = track_info["name"]
            track_url = track_info["url"]
            tags = get_track_tags(artist, track)
            data.append({
                "artist": artist,
                "track": track,
                "track_url": track_url,
                "related_artist": None,
                "related_track": None,
                "related_track_url": None,
                "tags": ", ".join(tags)
            })

            similar_tracks = get_similar_tracks(artist, track, limit=10)
            for s_info in similar_tracks:
                s_track = s_info["name"]
                s_artist = s_info["artist"]
                s_url = s_info["url"]
                s_tags = get_track_tags(s_artist, s_track)
                data.append({
                    "artist": artist,
                    "track": track,
                    "track_url": track_url,
                    "related_artist": s_artist,
                    "related_track": s_track,
                    "related_track_url": s_url,
                    "tags": ", ".join(s_tags)
                })
            time.sleep(0.5)

    df = pd.DataFrame(data)
    df.to_csv("back\data\unprocessed_musics.csv", index=False, encoding="utf-8")
    print("\n✅ Dataset salvo em unprocessed_musics.csv")

if __name__ == "__main__":
    main()
