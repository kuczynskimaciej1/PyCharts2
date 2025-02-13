import spotipy
from spotipy.oauth2 import SpotifyOAuth
from secrets import token_hex

SPOTIPY_CLIENT_ID = "8e50c2065b334b158b7d20399aea45af"
SPOTIPY_CLIENT_SECRET = "60b71876911648ff8ebbdd4a67f21293"
SPOTIPY_REDIRECT_URI = "http://127.0.0.1:5000/after_login_setup"

token_info = None
user_info = None
username = None
admin = False

spotify = spotipy.Spotify(auth=None, 
                          client_credentials_manager=spotipy.oauth2.SpotifyClientCredentials(
                            SPOTIPY_CLIENT_ID, 
                            SPOTIPY_CLIENT_SECRET))

sp_oauth = SpotifyOAuth(client_id = SPOTIPY_CLIENT_ID, 
                        client_secret = SPOTIPY_CLIENT_SECRET, 
                        redirect_uri = SPOTIPY_REDIRECT_URI,
                        state = token_hex(16),
                        scope = "user-library-read,user-library-modify,user-follow-read,user-follow-modify,playlist-modify-private,playlist-modify-public,user-read-private,user-read-email,user-read-playback-position,user-top-read,user-read-recently-played,ugc-image-upload")


def authenticate_spotify() -> None:
    global token_info, user_info, username  # Użyj zmiennych globalnych

    # Pobierz URL autoryzacji i otwórz go w przeglądarce
    auth_url = sp_oauth.get_authorize_url()
    print(f"Please navigate to the following URL to authorize: {auth_url}")

    # Poproś użytkownika o wklejenie przekierowanego URL
    redirect_response = input("Paste the URL you were redirected to: ")

    # Pobierz informacje o tokenie z odpowiedzi
    token_info = sp_oauth.get_access_token(sp_oauth.parse_response_code(redirect_response))

    # Inicjalizuj klienta Spotify
    global spotify
    spotify = spotipy.Spotify(auth=token_info['access_token'])
    print("Authentication successful!")

    # Pobierz informacje o użytkowniku
    user_info = spotify.me()
    username = user_info['display_name']
    print(f"Logged in as: {user_info['display_name']} ({user_info['email']})")



def ensure_token_valid() -> None:
    global token_info, spotify  # Użyj zmiennych globalnych

    if token_info is None:
        print("Token is not available. Please authenticate first.")
        authenticate_spotify()
    elif sp_oauth.is_token_expired(token_info):
        print("Token expired. Refreshing...")
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        spotify = spotipy.Spotify(auth=token_info['access_token'])
        print("Token refreshed successfully.")