import requests
import time
import pandas as pd
import networkx as nx
from collections import Counter

# -------------------------
# CONFIGURATION
# -------------------------
BASE_URL = "https://musicbrainz.org/ws/2"
HEADERS = {
    "User-Agent": "COMP4602-NetworkProject/1.1 ( student-project; your-email@example.com )"
}

TOP_K = 15
MIN_COLLAB_THRESHOLD = 2
ENABLE_TWO_HOP = True
MAX_GROUPS_TO_SCAN = 500  # Increased to find later-career collabs

# Caches to prevent redundant API calls
artist_cache = {} 
mbid_cache = {}

top_artists = [
    "Bruno Mars","Bad Bunny","The Weeknd","Rihanna","Taylor Swift",
    "Justin Bieber","Lady Gaga","Coldplay","Drake","Billie Eilish",
    "Ed Sheeran","Ariana Grande","J Balvin","David Guetta","Shakira",
    "Kendrick Lamar","Maroon 5","Eminem","Calvin Harris","SZA",
    "Kanye West","Pitbull","Dua Lipa","Lana Del Rey","Zara Larsson",
]

# -------------------------
# UTILS
# -------------------------
def safe_get(url, params):
    """Handles MusicBrainz rate limits (1 req/s) and 503 errors."""
    for attempt in range(3):
        time.sleep(1.1)  # Mechanical 1-second pause
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 503:
                print("  [Wait] Rate limited. Sleeping 5s...")
                time.sleep(5)
            else:
                break
        except Exception as e:
            print(f"  [Retry] Error: {e}")
            time.sleep(2)
    return None

# -------------------------
# STEP 2: Get MBID (Exact Search)
# -------------------------
def get_artist_mbid(name):
    if name in mbid_cache:
        return mbid_cache[name]
    
    url = f"{BASE_URL}/artist/"
    # Using strict artist name query to avoid 'Drake' matching 'Drake Bell' etc.
    params = {"query": f'artist:"{name}"', "fmt": "json", "limit": 1}
    
    data = safe_get(url, params)
    if data and data.get("artists"):
        mbid = data["artists"][0]["id"]
        mbid_cache[name] = mbid
        return mbid
    return None

# -------------------------
# STEP 3: Get Release Groups (Prioritized)
# -------------------------
def get_prioritized_release_groups(mbid):
    """Fetches and sorts Release Groups so Albums/EPs are processed first."""
    all_groups = []
    offset = 0
    limit = 100

    while len(all_groups) < MAX_GROUPS_TO_SCAN:
        url = f"{BASE_URL}/release-group"
        params = {"artist": mbid, "fmt": "json", "limit": limit, "offset": offset}
        data = safe_get(url, params)
        if not data or "release-groups" not in data:
            break

        page = data["release-groups"]
        all_groups.extend(page)
        if len(page) < limit:
            break
        offset += limit

    # Sort so high-value credits (Albums, EPs) come before Mixtapes/Live/Bootlegs
    priority = {"Album": 1, "EP": 2, "Single": 3}
    sorted_groups = sorted(
        all_groups, 
        key=lambda x: priority.get(x.get("primary-type"), 4)
    )
    return sorted_groups

# -------------------------
# STEP 4: Get Top-K collaborators
# -------------------------
def get_top_collaborators(artist_name):
    if artist_name in artist_cache:
        return artist_cache[artist_name]

    print(f"  Fetching data for: {artist_name}...")
    mbid = get_artist_mbid(artist_name)
    if not mbid:
        return []

    # Use the recording endpoint to catch 'Track-level' features
    url = f"{BASE_URL}/recording"
    params = {
        "artist": mbid,
        "fmt": "json",
        "limit": 100,
        "inc": "artist-credits" # This gets the 'feat.' artists on the tracks
    }
    
    data = safe_get(url, params)
    if not data or "recordings" not in data:
        return []

    collab_counter = Counter()
    
    # We use a set of titles to avoid counting the same song on 5 different albums
    seen_songs = set()

    for rec in data["recordings"]:
        title = rec.get("title", "").lower()
        if title in seen_songs:
            continue
        seen_songs.add(title)

        credits = rec.get("artist-credit", [])
        for c in credits:
            if isinstance(c, dict) and "artist" in c:
                name = c["artist"]["name"]
                if name.lower() != artist_name.lower():
                    collab_counter[name] += 1

    result = collab_counter.most_common(TOP_K)
    artist_cache[artist_name] = result
    
    if not result:
        print(f"    ! No collaborators found in top 100 recordings for {artist_name}")
    else:
        print(f"    + Found {len(result)} collaborators (Top: {result[0][0]})")
        
    return result

# -------------------------
# STEP 5: BUILD GRAPH
# -------------------------
G = nx.Graph()
processed = set()

for artist in top_artists:
    print(f"\n>>> Seed Artist: {artist}")
    top_collabs = get_top_collaborators(artist)
    processed.add(artist)

    for collab, weight in top_collabs:
        G.add_edge(artist, collab, weight=weight)

        # -------------------------
        # 2nd Hop with Caching
        # -------------------------
        if ENABLE_TWO_HOP and collab not in processed:
            # Skip 2nd hop if the collaborator is a main seed (will be handled later)
            if collab not in top_artists:
                print(f"    Scanning 2-hop: {collab}")
                second_collabs = get_top_collaborators(collab)
                processed.add(collab)

                for c2, w2 in second_collabs:
                    if w2 >= MIN_COLLAB_THRESHOLD:
                        G.add_edge(collab, c2, weight=w2)
    
    # Checkpoint after each seed artist
    nx.write_gml(G, "checkpoint_network.gml")

# -------------------------
# STEP 6: SAVE OUTPUT
# -------------------------
print(f"\nFinal Graph: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")

edges_df = nx.to_pandas_edgelist(G)
edges_df.to_csv("artist_collaborations.csv", index=False)
nx.write_gml(G, "artist_collaborations.gml")

print("Files saved: CSV and GML.")