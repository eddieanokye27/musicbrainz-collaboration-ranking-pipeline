import json
from collections import Counter, defaultdict
import pandas as pd

# Description: This script calculate the most common genre overlaps among collaborators for each artist, 
#              and also counts how many times each pair of genres co-occur among collaborators.
#
# example 1:   Dua-Lipa has 94 collaborators, among which 15 of them has "pop" as one of their genres which is 
#              the most common genre, so her dominant collaborator genre is "pop" with count 15.
# example 2:   Drake has 355 collaborators, among wich 12 of them has "Pop" and 12 of them has "Hip-Hop" while 
#              both of them are the most common genres, his dominant collaborator genre is "Pop" because it is alphabetically smaller than "Hip-Hop", 
#              and the count is 12.
#              Also Drake does not have "Pop" as one of his known genres, 
#              so this is an example where we can infer a new genre for an artist based on their collaborators.
#
# Problem:     1. Some artists have no dominant collaborator genre because all of their collaborators have no metadata, 
#                 so we cannot infer any dominant collaborator genre for them. For example, 88-Keys has 3 collaborations -- "Ye" "The Beatles" and "Ye" 
#                 but "Ye" and "The Beatles" have no metadata, so we cannot infer any new genre for "88-Keys" based on his collaborators.
#                 
#              2. Repeated collaborations -- some artists collaborate with the same artist multiple times, for example "Ye" and
#                 "88-Keys" have 2 collaborations, they are still counted as 2 collaborations and they contribute 2 counts to the genre overlaps, 
#                 we need to decided whether to keep the repeated collaborations or to deduplicate them, if we keep the duplicated collaborations,
#                 we also need to decide how to show them in our network.


# -----------------------------
# 1. Load files
# -----------------------------
collab_file = "collaborations.csv"
dataset = "top100_artists_dataset.json"

collab_df = pd.read_csv(collab_file)

with open(dataset, "r") as f:
    meta_json = json.load(f)

artists_meta = meta_json["artists"]


# -----------------------------
# 2. Build metadata lookup
# -----------------------------
# name -> metadata
meta_by_name = {}
for artist in artists_meta:
    meta_by_name[artist["name"]] = {
        "genres": artist.get("genre", []),
        "country": artist.get("country_of_origin"),
        "monthly_listeners_est": artist.get("monthly_listeners_est"),
        "peak_monthly_listeners_est": artist.get("peak_monthly_listeners_est"),
    }


# -----------------------------
# 3. Build undirected adjacency
# -----------------------------
adj = defaultdict(set)

for row in collab_df.itertuples(index=False):
    a = row.artist1
    b = row.artist2
    adj[a].add(b)
    adj[b].add(a)


# -----------------------------
# 4. Count collaborator-genre overlaps for each artist
# -----------------------------
artist_genre_overlap = {}

for artist, neighbors in adj.items():
    genre_counter = Counter()

    for neighbor in neighbors:
        if neighbor in meta_by_name:
            neighbor_genres = meta_by_name[neighbor]["genres"]
            genre_counter.update(neighbor_genres)

    artist_genre_overlap[artist] = genre_counter


# -----------------------------
# 5. Infer primary genre
# -----------------------------
# Rule:
# - if artist already has metadata, keep their known genres
# - also record dominant collaborator genre
# - if artist has no metadata, infer primary genre from collaborator overlaps
# - if no an artist has no dominant collaborator genre (ie. all of their collaborators have no metadata)
#   , then dominant collaborator genre is None

rows = []

for artist in sorted(adj.keys()):
    overlap_counter = artist_genre_overlap[artist]
    overlap_sorted = overlap_counter.most_common()

    known_genres = meta_by_name.get(artist, {}).get("genres", [])
    inferred_primary_genre = overlap_sorted[0][0] if overlap_sorted else None
    inferred_primary_count = overlap_sorted[0][1] if overlap_sorted else 0

    rows.append({
        "artist": artist,
        "num_collaborators": len(adj[artist]),
        "known_genres": ", ".join(known_genres) if known_genres else None,
        "dominant_collaborator_genre": inferred_primary_genre,
        "dominant_collaborator_genre_count": inferred_primary_count,
        "top_5_collaborator_genres": str(overlap_sorted[:5]),
        "has_metadata": artist in meta_by_name,
    })

artist_genre_df = pd.DataFrame(rows)


# -----------------------------
# 6. Build genre-to-genre collaboration matrix
# -----------------------------
# For each collaboration edge, if both artists have known genres,
# count all genre pair combinations between them.
genre_pair_counter = Counter()

for row in collab_df.itertuples(index=False):
    a = row.artist1
    b = row.artist2

    genres_a = meta_by_name.get(a, {}).get("genres", [])
    genres_b = meta_by_name.get(b, {}).get("genres", [])

    if not genres_a or not genres_b:
        continue

    for ga in genres_a:
        for gb in genres_b:
            pair = tuple(sorted([ga, gb]))
            genre_pair_counter[pair] += 1

genre_pair_rows = []
for (g1, g2), count in genre_pair_counter.items():
    genre_pair_rows.append({
        "genre_1": g1,
        "genre_2": g2,
        "collaboration_count": count
    })

genre_pairs_df = pd.DataFrame(genre_pair_rows).sort_values(
    by="collaboration_count", ascending=False
)


# -----------------------------
# 7. Save outputs
# -----------------------------
artist_genre_df.to_csv("artist_genre_overlap_results.csv", index=False)
genre_pairs_df.to_csv("genre_pair_collaboration_counts.csv", index=False)

print("Saved:")
print("- artist_genre_overlap_results.csv")
print("- genre_pair_collaboration_counts.csv")


# -----------------------------
# 8. Example checks
# -----------------------------
for target in ["The Weeknd", "Bruno Mars", "Daft Punk", "Philip Lawrence", "Drake"]:
    if target in artist_genre_df["artist"].values:
        row = artist_genre_df[artist_genre_df["artist"] == target].iloc[0]
        print("\nArtist:", target)
        print("Known genres:", row["known_genres"])
        print("Dominant collaborator genre:", row["dominant_collaborator_genre"])
        print("Top 5 collaborator genres:", row["top_5_collaborator_genres"])