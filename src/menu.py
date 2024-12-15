import globals
import data
import spotify_api

def start():
    print("PyCharts v2.0 - the dynamic AI learning analysis software")
    print("Created by Maciej Kuczynski, 2024/2025")
    spotify_api.authenticate_spotify()
    # Example: Ensure token validity before making API calls
    #spotify_api.ensure_token_valid()
    # Example Spotify API call (get current user's playlists)
    #playlists = spotify_api.spotify.current_user_playlists()
    #for playlist in playlists['items']:
        #print(f"{playlist['name']} - {playlist['tracks']['total']} tracks")
    spotify_api.fetch_tracks_info()
    #while(True):
    #    menu()

def menu():
    pass