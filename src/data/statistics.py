import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Wczytanie danych (zakładam, że dane są w pliku CSV)
df = pd.read_csv('../../data/full.csv')  # Zmień nazwę pliku jeśli potrzeba

# Usunięcie kolumny indeksującej (jeśli nie jest potrzebna)
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Podstawowe statystyki opisowe
print("Statystyki opisowe dla kolumn numerycznych:")
print(df.describe(include=[np.number]))

# Analiza dla kolumn kategorycznych/tekstowych
categorical_cols = ['track_id', 'track_name', 'artist_id', 'artist_name', 'release_id', 'release_name']
print("\nUnikalne wartości dla kolumn kategorycznych:")
for col in categorical_cols:
    if col in df.columns:
        print(f"{col}: {df[col].nunique()} unikalnych wartości")

# Generowanie histogramów dla kolumn numerycznych
numeric_cols = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(15, 20))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(6, 3, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Rozkład {col}')
    plt.xlabel(col)
    plt.ylabel('Liczba utworów')
    plt.grid(True, alpha=0.3)
    
plt.tight_layout()
plt.savefig('all_histograms.png', dpi=300)
plt.show()

# Oddzielne histogramy dla każdej kolumny (zapis do plików)
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Rozkład {col}')
    plt.xlabel(col)
    plt.ylabel('Liczba utworów')
    plt.grid(True, alpha=0.3)
    
    # Zapis do pliku
    filename = f'histogram_{col}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano histogram dla {col} do {filename}")

# Dodatkowa analiza dla kolumn kategorycznych
if 'explicit' in df.columns:
    plt.figure(figsize=(8, 5))
    df['explicit'].value_counts().plot(kind='bar')
    plt.title('Rozkład treści eksplicytnych')
    plt.xlabel('Czy utwór zawiera treści eksplicytne?')
    plt.ylabel('Liczba utworów')
    plt.savefig('explicit_distribution.png', dpi=300)
    plt.show()