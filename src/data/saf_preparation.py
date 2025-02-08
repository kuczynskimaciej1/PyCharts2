import pandas as pd



def merge(set1: pd.DataFrame, set2: pd.DataFrame) -> pd.DataFrame:
    total = pd.DataFrame()
    total = pd.concat([set1, set2], ignore_index = True)
    return total



def delete_duplicates(set: pd.DataFrame) -> pd.DataFrame:
    print(set.shape[0])
    set.drop_duplicates(subset=['track_id'])
    print(set.shape[0])
    return set



november2018 = pd.read_csv("../../data/Spotify Audio Features/original/SpotifyAudioFeaturesNov2018.csv")
april2019 = pd.read_csv("../../data/Spotify Audio Features/original/SpotifyAudioFeaturesApril2019.csv")
total = merge(november2018, april2019)
total = delete_duplicates(total)
total.to_csv("../../data/Spotify Audio Features/saf_data.csv")