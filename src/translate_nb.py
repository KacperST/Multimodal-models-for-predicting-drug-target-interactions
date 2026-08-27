import json

with open('src/dataset_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = {
    'Ten notebook ładuje zbiór danych DTI i analizuje rozkład długości sekwencji białek oraz uśmiechów (SMILES), a także podstawowe statystyki par DTI.': 'This notebook loads the DTI dataset and analyzes the sequence length distribution of proteins and SMILES, as well as basic statistics of DTI pairs.',
    '# Wczytanie danych': '# Load data',
    'Liczba wszystkich par:': 'Total number of pairs:',
    'Unikalne leki (SMILES):': 'Unique drugs (SMILES):',
    'Unikalne białka:': 'Unique proteins:',
    'Rozkład klas (aktywne/nieaktywne):': 'Class distribution (active/inactive):',
    '# Obliczanie długości znakowych (jest to rewelacyjne przybliżenie liczby tokenów dla SMILES i białek)': '# Calculating character lengths (this is an excellent approximation of token count for SMILES and proteins)',
    'Statystyki długości znakowej SMILES:': 'SMILES character length statistics:',
    'Statystyki długości znakowej Białek:': 'Protein character length statistics:',
    '# Wykresy rozkładów': '# Distribution plots',
    '# Używamy 99. percentyla by usunąć ekstrema do ładnego wykresu': '# We use the 99th percentile to remove extremes for a nicer plot',
    'Rozkład długości SMILES (odcięto 1% ekstremów > ': 'SMILES length distribution (cut off 1% extremes > ',
    'Liczba znaków': 'Number of characters',
    'Liczba próbek': 'Number of samples',
    'Rozkład długości Białek (odcięto 1% ekstremów > ': 'Protein length distribution (cut off 1% extremes > ',
    '### Analiza ucięcia długości (Truncation)': '### Length Truncation Analysis',
    'W plikach konfiguracyjnych mamy ustawione `max_length` (np. 1024 dla ESM2 i 256 dla ChemBERT).': 'In the configuration files we have set `max_length` (e.g. 1024 for ESM2 and 256 for ChemBERT).',
    'Zobaczmy, jaki procent naszego zbioru danych przekracza te wartości.': 'Let\'s see what percentage of our dataset exceeds these values.',
    'Procent leków, które zostaną obcięte przy długości ': 'Percentage of drugs that will be truncated at length ',
    'Procent białek, które zostaną obcięte przy długości ': 'Percentage of proteins that will be truncated at length ',
    '## Dodatkowa analiza (Sparsity, Stopień Wierzchołków, Różnice w Klasach)': '## Additional Analysis (Sparsity, Degree Distribution, Class Differences)',
    '### 1. Rzadkość macierzy DTI (Sparsity)': '### 1. Sparsity of the DTI matrix',
    'Sprawdźmy jaki ułamek wszystkich teoretycznie możliwych interakcji pokrywa nasz zbiór danych.': 'Let\'s check what fraction of all theoretically possible interactions our dataset covers.',
    'Całkowita możliwa liczba kombinacji lek-białko: ': 'Total possible number of drug-protein combinations: ',
    'Liczba par w naszym zbiorze: ': 'Number of pairs in our dataset: ',
    'Pokrycie macierzy (Sparsity): ': 'Matrix coverage (Sparsity): ',
    'Nasz zbiór pokrywa zaledwie około ': 'Our dataset covers only about ',
    '% wszystkich możliwych interakcji.': '% of all possible interactions.',
    '### 2. Rozkład stopnia wierzchołków (Degree Distribution)': '### 2. Degree Distribution',
    'Pokazuje jak dobrze zbadane są poszczególne białka (ile leków do nich przypisano) oraz jak zróżnicowane są leki (z iloma białkami zbadano ich interakcję). Oczekujemy tzw. long-tail distribution.': 'Shows how well-studied individual proteins are (how many drugs are assigned to them) and how diverse the drugs are (how many proteins their interaction was tested with). We expect a so-called long-tail distribution.',
    'Rozkład stopnia białek (leki per białko) - skala log': 'Protein degree distribution (drugs per protein) - log scale',
    'Liczba przypisanych leków': 'Number of assigned drugs',
    'Liczba białek': 'Number of proteins',
    'Rozkład stopnia leków (białka per lek) - skala log': 'Drug degree distribution (proteins per drug) - log scale',
    'Liczba przypisanych białek': 'Number of assigned proteins',
    'Liczba leków': 'Number of drugs',
    'Białka (Ile leków przypada na białko):': 'Proteins (How many drugs per protein):',
    'Średnia: ': 'Mean: ',
    ' | Mediana: ': ' | Median: ',
    'Leki (Z iloma białkami łączy się lek):': 'Drugs (How many proteins a drug connects to):',
    '### 3. Porównanie długości między klasami (Active vs Inactive)': '### 3. Length comparison between classes (Active vs Inactive)',
    'Weryfikujemy, czy klasa pozytywna i negatywna mają zbliżony rozkład długości. Zbyt duże różnice mogą sprawić, że model zamiast uczyć się fizyki/chemii oddziaływań, nauczy się heurystyki "dłuższe/krótsze cząsteczki to inna klasa" (bias zbioru).': 'We verify whether the positive and negative classes have a similar length distribution. Differences that are too large could cause the model to learn the heuristic "longer/shorter molecules are a different class" (dataset bias) instead of learning the physics/chemistry of interactions.',
    '# SMILES length by class (odcinamy 99 percentyl dla czytelności wykresu, tak jak w poprzednich)': '# SMILES length by class (we cut off the 99th percentile for chart readability, as in previous ones)',
    'Rozkład długości SMILES według klas aktywności': 'SMILES length distribution by activity class',
    'Rozkład długości sekwencji Białek według klas aktywności': 'Protein sequence length distribution by activity class'
}

for cell in nb.get('cells', []):
    new_source = []
    for line in cell.get('source', []):
        for pol, eng in replacements.items():
            line = line.replace(pol, eng)
        new_source.append(line)
    cell['source'] = new_source

with open('src/dataset_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook translated to English successfully!")
