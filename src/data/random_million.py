import csv
import random

def random_sample_large_csv(input_file, output_file, sample_size=1000000):
    # Najpierw zlicz wiersze (bez nagłówka)
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Pobierz nagłówki
        n_lines = sum(1 for _ in reader)
    
    if n_lines < sample_size:
        raise ValueError(f"Plik ma tylko {n_lines} wierszy.")
    
    # Znajdź indeks kolumny do usunięcia
    drop_col_index = headers.index('Unnamed: 0')
    
    # Wygeneruj unikalne losowe indeksy
    random_indices = set(random.sample(range(n_lines), sample_size))
    
    # Przeczytaj i zapisz wybrane wiersze
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Zapisz nagłówki (bez kolumny do usunięcia)
        headers = next(reader)
        writer.writerow([h for i, h in enumerate(headers) if i != drop_col_index])
        
        for i, row in enumerate(reader):
            if i in random_indices:
                # Zapisz wiersz bez kolumny do usunięcia
                writer.writerow([col for j, col in enumerate(row) if j != drop_col_index])
                
    print(f"Zapisano {sample_size} losowych wierszy do {output_file}")

# Użycie
random_sample_large_csv('../../data/full_training_data.csv', '../../data/million_data.csv')