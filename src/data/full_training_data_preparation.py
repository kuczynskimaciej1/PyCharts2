import pandas as pd

def merge_sets():
    set_8m: pd.DataFrame = pd.read_csv('../../data/8+ M. Spotify Tracks, Genre, Audio Features/8m_data.csv')
    set_10m: pd.DataFrame = pd.read_csv('../../data/10+ M. Beatport Tracks/10m_data.csv')
    total: pd.DataFrame = pd.concat([set_8m, set_10m], ignore_index = True, axis=0)
    total.to_csv('../../data/full_training_data.csv')



def drop_duplicates_pandas() -> None:
    total: pd.DataFrame = pd.read_csv('../../data/full_training_data.csv')
    total = total.drop_duplicates(subset=['track_id'], ignore_index=True)
    total.to_csv('../../data/full_training_data.csv')



def count_unique_values() -> None:
    total: pd.DataFrame = pd.read_csv('../../data/full_training_data.csv')
    print("read")
    print(total.shape[0])
    print(total['track_id'].nunique())



def delete_cols() -> None:
    total: pd.DataFrame = pd.read_csv('../../data/full_training_data.csv')
    total = total.drop(columns=['Unnamed: 0.3','Unnamed: 0.2','Unnamed: 0.1','Unnamed: 0'])
    total.to_csv('../../data/full_training_data.csv')



def all() -> None:
    set_8m: pd.DataFrame = pd.read_csv('../../data/8+ M. Spotify Tracks, Genre, Audio Features/8m_data.csv')
    set_10m: pd.DataFrame = pd.read_csv('../../data/10+ M. Beatport Tracks/10m_data.csv')
    total: pd.DataFrame = pd.concat([set_8m, set_10m], ignore_index = True, axis=0)
    total = total.drop_duplicates(subset=['track_id'], ignore_index=True)
    print(total.shape[0])
    print(total['track_id'].nunique())
    print(total.columns)
    total.to_csv('../../data/full_training_data.csv')



def release_id_null_values() -> None:
    data = pd.read_csv("../../data/full_training_data.csv")
    print(data[['track_id', 'artist_id', 'release_id']].isnull().sum())
    data = data.dropna(subset=['release_id'])
    data.to_csv('../../data/full_training_data.csv')