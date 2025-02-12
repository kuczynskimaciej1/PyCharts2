import pandas as pd
from pyspark.sql import SparkSession

def merge_sets():
    set_8m = pd.read_csv('../../data/8+ M. Spotify Tracks, Genre, Audio Features/8m_data.csv')
    set_10m = pd.read_csv('../../data/10+ M. Beatport Tracks/10m_data.csv')
    print("loaded")
    total = pd.concat([set_8m, set_10m], ignore_index = True, axis=0)
    print("concat done")
    total.to_csv('../../data/full_training_data.csv')
    print("saved")

    print(total.columns)
    print(total.shape[0])

def drop_duplicates_pandas() -> None:
    total = pd.read_csv('../../data/full_training_data.csv')
    print("read")
    total.drop_duplicates(subset=['track_id'], ignore_index=True)
    print("dropped")
    total.to_csv('../../data/full_training_data.csv')
    print("saved")

def drop_duplicates_spark() -> None:
    # Inicjalizacja sesji Spark
    spark = SparkSession.builder \
        .appName("DropDuplicates") \
        .getOrCreate()

    # Wczytaj dane
    total = spark.read.csv('../../data/full_training_data.csv', header=True, inferSchema=True)
    print("Dane wczytane")

    # Usuń duplikaty na podstawie kolumny 'track_id'
    total = total.dropDuplicates(subset=['track_id'])
    print("Duplikaty usunięte")

    # Zapisz wynik
    total.write.csv('../../data/full_training_data_no_duplicates.csv', header=True, mode='overwrite')
    print("Dane zapisane")

    # Zatrzymaj sesję Spark
    spark.stop()

drop_duplicates_pandas()