from globals import sp_oauth, spotipy, spotify, token_info

def authenticate_spotify() -> None:
    global token_info, spotify

    # Get the authorization URL and open it in the browser
    auth_url = sp_oauth.get_authorize_url()
    print(f"Please navigate to the following URL to authorize: {auth_url}")

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



def ensure_token_valid() -> None:
    global token_info, spotify

    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        spotify = spotipy.Spotify(auth=token_info['access_token'])