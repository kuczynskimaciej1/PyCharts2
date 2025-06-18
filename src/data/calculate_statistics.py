import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import sys

# Konfiguracja zapisu outputu do pliku
original_stdout = sys.stdout
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f'spotify_analysis_report_{timestamp}.txt'

df = None

with open(output_filename, 'w') as f:
    sys.stdout = f
    
    print(f"Analiza danych Spotify - {timestamp}\n")
    print("="*80 + "\n")
    
    try:
        # Wczytanie danych
        print("Ładowanie danych...")
        df = pd.read_csv('../../data/full.csv')  # Zmień nazwę pliku jeśli potrzeba
        
        # Usunięcie kolumny indeksującej
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
            print("Usunięto kolumnę indeksującą 'Unnamed: 0'")
        
        # Podstawowe informacje o danych
        print("\n=== Podstawowe informacje o zbiorze danych ===")
        print(f"Liczba utworów: {len(df)}")
        print(f"Kolumny: {list(df.columns)}\n")
        
        # Statystyki opisowe
        print("\n=== Statystyki opisowe (kolumny numeryczne) ===")
        print(df.describe(include=[np.number]).to_string())
        
        # Analiza dla kolumn kategorycznych
        categorical_cols = ['track_id', 'track_name', 'artist_id', 'artist_name', 'release_id', 'release_name']
        print("\n=== Analiza kolumn kategorycznych ===")
        for col in categorical_cols:
            if col in df.columns:
                print(f"\n{col}:")
                print(f"  Unikalnych wartości: {df[col].nunique()}")
                if df[col].nunique() < 20:  # Wyświetl wartości tylko jeśli jest ich mało
                    print(f"  Przykładowe wartości: {df[col].unique()[:5]}")
        
        # Macierz korelacji
        print("\n=== Macierz korelacji ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        print(corr_matrix.to_string())
        
        # Zapis macierzy korelacji do pliku
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Macierz korelacji cech numerycznych')
        corr_filename = f'correlation_matrix_{timestamp}.png'
        plt.savefig(corr_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nZapisano macierz korelacji do {corr_filename}")
        
        # Generowanie histogramów (z obsługą różnych wersji Seaborn)
        print("\n=== Generowanie histogramów ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Wersja dla Seaborn >= 0.11 (histplot) lub starsza (distplot)
        def create_hist(col, data):
            plt.figure(figsize=(10, 6))
            try:
                # Nowsze wersje Seaborn
                sns.histplot(data=data, x=col, kde=True, bins=30)
            except AttributeError:
                # Starsze wersje Seaborn
                sns.distplot(data[col], kde=True, bins=30)
            plt.title(f'Rozkład {col}')
            plt.xlabel(col)
            plt.ylabel('Liczba utworów')
            plt.grid(True, alpha=0.3)
            filename = f'histogram_{col}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Zapisano histogram dla {col} do {filename}")
        
        for col in numeric_cols:
            create_hist(col, df)
        
        # Dodatkowa analiza dla kolumn kategorycznych
        if 'explicit' in df.columns:
            plt.figure(figsize=(8, 5))
            df['explicit'].value_counts().plot(kind='bar')
            plt.title('Rozkład treści eksplicytnych')
            plt.xlabel('Czy utwór zawiera treści eksplicytne?')
            plt.ylabel('Liczba utworów')
            explicit_filename = f'explicit_distribution_{timestamp}.png'
            plt.savefig(explicit_filename, dpi=300)
            plt.close()
            print(f"\nZapisano rozkład treści eksplicytnych do {explicit_filename}")
        
        print("\n=== Analiza zakończona pomyślnie ===")
        
    except Exception as e:
        print(f"\n!!! Wystąpił błąd podczas analizy: {str(e)}")
    
    finally:
        sys.stdout = original_stdout
        print(f"Pełny raport został zapisany do: {output_filename}")

# Wyświetlenie podsumowania w konsoli
print(f"Analiza zakończona. Wyniki zapisano w:")
print(f"- Raport tekstowy: {output_filename}")
print(f"- Macierz korelacji: correlation_matrix_{timestamp}.png")
print(f"- Histogramy: histogram_*_{timestamp}.png")
if 'explicit' in df.columns:
    print(f"- Rozkład treści eksplicytnych: explicit_distribution_{timestamp}.png")