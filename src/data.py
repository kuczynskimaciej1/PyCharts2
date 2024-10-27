from globals import pd, os, json

def fix_parentheses():
    read_json_path = os.path.join("..", "data", "mpd.slice.0-999.json")
    read_file = open(read_json_path, "r")
    content = read_file.read()
    read_file.close()

    content_v1 = content.replace("[", "{")
    content_v2 = content_v1.replace("]", "}")
    
    write_json_path = os.path.join("..", "data", "mpd.slice.0-999b.json")
    write_file = open(write_json_path, "w")
    write_file.write(content_v2)
    write_file.close()



def read_file():
    json_path = os.path.join("..", "data", "mpd.slice.0-999.json")
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

    print(playlists_df)