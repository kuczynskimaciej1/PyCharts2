from globals import webbrowser, sp_oauth, spotipy, spotify, token_info

def authenticate_spotify():
    global token_info, spotify

    # Get the authorization URL and open it in the browser
    auth_url = sp_oauth.get_authorize_url()
    print(f"Please navigate to the following URL to authorize: {auth_url}")
    webbrowser.open(auth_url)

    # Prompt the user to paste the redirect URL
    redirect_response = input("Paste the URL you were redirected to: ")

    # Extract token information from the response
    token_info = sp_oauth.get_access_token(sp_oauth.parse_response_code(redirect_response))

    # Initialize Spotify client
    spotify = spotipy.Spotify(auth=token_info['access_token'])
    print("Authentication successful!")

    # Display user information
    user_info = spotify.me()
    print(f"Logged in as: {user_info['display_name']} ({user_info['email']})")



def ensure_token_valid():
    global token_info, spotify

    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        spotify = spotipy.Spotify(auth=token_info['access_token'])



def fetch_tracks_info():
    track_ids = []
    with open("all_nodupl.txt", "r") as file:
        track_ids.append([file.readline() for _ in range(100)])

    ensure_token_valid()
    ids_param = ""
    for element in track_ids[0]:
        print(element)
        element = element.rstrip()
        ids_param = ids_param + element + ","

    ids_param = ids_param[:-1]
    print(ids_param)
    response = spotify.tracks(tracks=ids_param)

    return response