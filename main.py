import requests
import time
import itertools
import pandas as pd
import networkx as nx
import json
import os
from collections import Counter, deque

BASE_URL = "https://musicbrainz.org/ws/2"

HEADERS = {
    "User-Agent": "COMP4602-NetworkProject/1.0 ( student-project )"
}


# RATE LIMITERr

LAST_REQUEST_TIME = 0

def rate_limited_request(url, params):
    global LAST_REQUEST_TIME

    elapsed = time.time() - LAST_REQUEST_TIME
    if elapsed < 1:
        time.sleep(1 - elapsed)

    r = requests.get(url, params=params, headers=HEADERS)
    LAST_REQUEST_TIME = time.time()
    return r



# CONFIG
MAX_DEPTH = 2
MAX_RECORDINGS_PER_ARTIST = 500
MIN_COLLABS_TO_EXPAND = 1
SAVE_EVERY = 10   # 🔥 autosave frequency


# FILE PaTHS

MBID_CACHE_FILE = "mbid_cache.json"
RECORDING_CACHE_FILE = "recording_cache.json"
STATE_FILE = "bfs_state.json"
EDGES_FILE = "edges.json"



# LOAD STATE (RESUME)


def load_json(file, default):
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return default

mbid_cache = load_json(MBID_CACHE_FILE, {})
recording_cache = load_json(RECORDING_CACHE_FILE, {})
edges = load_json(EDGES_FILE, [])

state = load_json(STATE_FILE, None)

if state:
    print("Resuming previous run...")
    visited_mbids = set(state["visited_mbids"])
    queue = deque(state["queue"])
else:
    print("Starting fresh...")

    visited_mbids = set()
    queue = deque()

    top_artists = [
        "Bruno Mars","Bad Bunny","The Weeknd","Rihanna","Taylor Swift",
        "Justin Bieber","Lady Gaga","Coldplay","Drake","Billie Eilish",
        "Ed Sheeran","Ariana Grande","J Balvin","David Guetta","Shakira",
        "Kendrick Lamar","Maroon 5","Eminem","Calvin Harris","SZA",
        "Ye","Pitbull","Dua Lipa","Lana Del Rey","Zara Larsson",
    ]

    for artist in top_artists:
        queue.append((artist, 0))



#SAVE STATE

def save_all():
    with open(MBID_CACHE_FILE, "w") as f:
        json.dump(mbid_cache, f)

    with open(RECORDING_CACHE_FILE, "w") as f:
        json.dump(recording_cache, f)

    with open(EDGES_FILE, "w") as f:
        json.dump(edges, f)

    with open(STATE_FILE, "w") as f:
        json.dump({
            "visited_mbids": list(visited_mbids),
            "queue": list(queue)
        }, f)

    print("Progress saved!")



# STEP 1: MBID (cached)


def get_artist_mbid(name):
    if name in mbid_cache:
        return mbid_cache[name]

    url = f"{BASE_URL}/artist/"
    params = {"query": name, "fmt": "json", "limit": 1}

    try:
        r = rate_limited_request(url, params)
        data = r.json()

        if data["artists"]:
            mbid = data["artists"][0]["id"]
            mbid_cache[name] = mbid
            return mbid
    except:
        pass

    return None



# STEP 2: RECORDINGS (cached)

def get_recordings(mbid):
    if mbid in recording_cache:
        return recording_cache[mbid]

    recordings = []
    limit = 100
    offset = 0

    while len(recordings) < MAX_RECORDINGS_PER_ARTIST:
        url = f"{BASE_URL}/recording"
        params = {
            "artist": mbid,
            "fmt": "json",
            "limit": limit,
            "offset": offset,
            "inc": "artist-credits"
        }

        try:
            r = rate_limited_request(url, params)
            data = r.json()
        except:
            break

        page_records = data.get("recordings", [])
        if not page_records:
            break

        recordings.extend(page_records)

        if len(page_records) < limit:
            break

        offset += limit

    recording_cache[mbid] = recordings[:MAX_RECORDINGS_PER_ARTIST]
    return recording_cache[mbid]



# BFS LOOP (RESUMABLE)

processed_count = 0

while queue:
    current_artist, depth = queue.popleft()

    if depth > MAX_DEPTH:
        continue

    mbid = get_artist_mbid(current_artist)
    if not mbid or mbid in visited_mbids:
        continue

    print(f"Processing: {current_artist} (depth {depth})")
    visited_mbids.add(mbid)

    recordings = get_recordings(mbid)

    collaborators = set()

    for rec in recordings:
        artists = [
            credit["artist"]["name"]
            for credit in rec.get("artist-credit", [])
            if "artist" in credit
        ]

        for pair in itertools.combinations(set(artists), 2):
            edges.append(pair)

        collaborators.update(artists)

    # prune weak nodes
    if len(collaborators) >= MIN_COLLABS_TO_EXPAND:
        for collab in collaborators:
            queue.append((collab, depth + 1))

    processed_count += 1

    # AUTO SAVE
    if processed_count % SAVE_EVERY == 0:
        save_all()



# FINAL SAVE

save_all()



# GRAPH BUILDING (SKIPS REFETCH)

print("Building graph...")

edge_counts = Counter(tuple(sorted(e)) for e in edges)

edges_df = pd.DataFrame(
    [(a, b, w) for (a, b), w in edge_counts.items()],
    columns=["artist1", "artist2", "weight"]
)

edges_df = edges_df[edges_df["weight"] >= 2]

G = nx.from_pandas_edgelist(
    edges_df,
    "artist1",
    "artist2",
    edge_attr="weight"
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())



# METRICS and SUCCESS SCORE

degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G, weight="weight")
eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=500)
betweenness = nx.betweenness_centrality(G, weight="weight")

def normalize(d):
    min_v, max_v = min(d.values()), max(d.values())
    if max_v - min_v == 0:
        return {k: 0 for k in d}
    return {k: (v - min_v) / (max_v - min_v) for k, v in d.items()}

deg = normalize(degree_centrality)
pr = normalize(pagerank)
eig = normalize(eigenvector)
bet = normalize(betweenness)

success_score = {
    node: 0.25 * pr[node] +
          0.25 * eig[node] +
          0.2 * deg[node] +
          0.3 * bet[node]
    for node in G.nodes()
}

success_df = pd.DataFrame({
    "artist": list(success_score.keys()),
    "success_score": list(success_score.values())
}).sort_values(by="success_score", ascending=False)

success_df.to_csv("artist_success_scores.csv", index=False)
nx.write_gml(G, "artist_collaboration_network.gml")

print("Done!")
print(success_df.head(10))