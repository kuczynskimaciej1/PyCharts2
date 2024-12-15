from globals import pd, os, json, glob


def read_file(file_range_from, file_range_to) -> pd.DataFrame:
    json_path = os.path.join("..", "data", f"mpd.slice.{file_range_from}-{file_range_to}.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    playlists_df = pd.json_normalize(
        data, 
        record_path=["playlists", "tracks"],
        meta=[
            ["playlists", "name"],
            ["playlists", "collaborative"],
            ["playlists", "pid"],
            ["playlists", "num_tracks"],
            ["playlists", "num_albums"],
            ["playlists", "num_followers"]
        ]
    )

    print(f"Reading {file_range_from}-{file_range_to} complete.")
    return playlists_df



def write_track_uris_to_file(json_data, file_range_from, file_range_to) -> None:
    track_uris = []
    for track in json_data['track_uri']:
            track_uris.append(track)

    unique_uris = list(dict.fromkeys(uri.strip() for uri in track_uris))
    print(f"Removing duplicates in {file_range_from}-{file_range_to} complete.")

    with open(f"{file_range_from}-{file_range_to}_uris_nodupl.txt", "w") as file:
        for uri in unique_uris:
            uri = uri[14:]
            file.write(uri + "\n")
    print(f"Writing {file_range_from}-{file_range_to} complete.")



def extract_track_uris() -> None:
    for i in range(0, 1000000, 1000):
        data = read_file(i, i+999)
        write_track_uris_to_file(data, i, i+999)



def merge_txts() -> None:
    file_list = glob("*.txt")

    unique_lines = set()

    for f in file_list:
        with open(f, "r", encoding="utf-8") as infile:
            for line in infile:
                unique_lines.add(line.strip())

    with open("all_nodupl.txt", "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(sorted(unique_lines)))