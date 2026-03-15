import requests
import time
import itertools
import pandas as pd
import networkx as nx

BASE_URL = "https://musicbrainz.org/ws/2"

HEADERS = {
    "User-Agent": "COMP4602-NetworkProject/1.0 ( student-project )"
}

# Step 1: top artists list
# Replace with your own top 100 list

top_artists = [
"Bruno Mars",
"Bad Bunny",
"The Weeknd",
"Rihanna",
"Taylor Swift",
"Justin Bieber",
"Lady Gaga",
"Coldplay",
"Drake",
"Billie Eilish",
"Ed Sheeran",
"Ariana Grande",
"J Balvin",
"David Guetta",
"Shakira",
"Kendrick Lamar",
"Maroon 5",
"Eminem",
"Calvin Harris",
"SZA",
"Kanye West",
"Pitbull",
"Dua Lipa",
"Lana Del Rey",
"Zara Larsson",
]

# you can extend this to 100 artists


# Step 2:gget MusicBrainz Artist ID

def get_artist_mbid(name):
    url = f"{BASE_URL}/artist/"
    params = {
        "query": name,
        "fmt": "json",
        "limit": 1
    }

    r = requests.get(url, params=params, headers=HEADERS)
    data = r.json()

    if data["artists"]:
        return data["artists"][0]["id"]

    return None


# Step 3: Get recordings

def get_recordings(mbid):
    recordings = []
    limit = 100
    offset = 0

    while True:
        url = f"{BASE_URL}/recording"
        params = {
            "artist": mbid,
            "fmt": "json",
            "limit": limit,
            "offset": offset,
            "inc": "artist-credits"
        }

        r = requests.get(url, params=params, headers=HEADERS)
        data = r.json()

        page_records = data.get("recordings", [])
        if not page_records:
            break

        recordings.extend(page_records)

        # If fewer results than limit, we reached the last page
        if len(page_records) < limit:
            break

        offset += limit
        time.sleep(1)  # respect MusicBrainz rate limits

    return recordings


# Step 4: Extract collaborations

edges = []

for artist in top_artists:

    print("Processing:", artist)

    mbid = get_artist_mbid(artist)

    if not mbid:
        print("Artist not found")
        continue

    recordings = get_recordings(mbid)

    for rec in recordings:

        artists = []

        for credit in rec["artist-credit"]:
            if "artist" in credit:
                artists.append(credit["artist"]["name"])

        # create pairwise collaborations
        for pair in itertools.combinations(set(artists), 2):
            edges.append(pair)

    time.sleep(1)  # avoid API rate limits


# Step 5: build edge list


edges_df = pd.DataFrame(edges, columns=["artist1", "artist2"])

edges_df.to_csv("collaborations.csv", index=False)

print("Saved edge list")


#Step 6: build NetworkX graph

G = nx.from_pandas_edgelist(edges_df, "artist1", "artist2")

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# Save graph

nx.write_gml(G, "artist_collaboration_network.gml")

print("Network saved")