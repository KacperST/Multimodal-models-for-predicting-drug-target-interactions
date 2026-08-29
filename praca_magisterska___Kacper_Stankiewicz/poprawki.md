
1. Ocena ogólna
Rdzeń eksperymentalny pracy jest solidny i pracochłonny: 451 858 par lek–białko z BindingDB, podział scaffoldowy, 21 konfiguracji enkoderów w Fazie 1, 10 w Fazie 2, rozbudowana EDA i analiza błędów. To realny wkład — systematyczne badanie kombinatoryczne modalności jest rzadsze w literaturze DTI niż kolejny pojedynczy model, i praca ma w tym punkcie własną, obronioną tezę.

Ale w obecnym stanie praca nie jest gotowa do złożenia. Trzy powody:

Rozdział 1 (Wstęp) i rozdział 5 (Podsumowanie) są puste — same nagłówki. Nie ma motywacji, sformułowanych celów, wniosków, ograniczeń ani kierunków dalszych prac. To dyskwalifikujące formalnie.
Rysunki z Fazy 1 nie zgadzają się liczbowo z Tabelą 4.1 (szczegóły w §4.1 poniżej). To znaczy, że tabele i wykresy pochodzą z różnych uruchomień. Recenzent to zauważy.
Wnioski są mocniejsze niż dane. Różnice AUC między czołowymi modelami (0,0014–0,010) są tego samego rzędu co rozrzut międzyuruchomieniowy, który daje się odczytać z punktu 2 (0,003–0,006). Przy jednym uruchomieniu bez ziaren losowych ranking modeli nie jest uzasadniony.

Do tego dochodzi rozjazd między opisem tematu a realizacją (§2) oraz wyraźne ślady generowania tekstu przez LLM (§7).

Prognoza oceny: po naprawie punktów krytycznych (§4) — solidne 4,5. Po dodaniu eksperymentu cold-start i wielokrotnych uruchomień (§9, priorytet A) — 5. W obecnym stanie — praca do zwrotu.


2. Realizacja celów z opisu tematu
Cel z opisu
Status
Uwaga
SMILES, grafy molekularne
✅
GCN + ECFP + ChemBERTa
Sekwencje białkowe
✅
CNN + ESM-2
Dane strukturalne białek (3D)
❌
brak zupełnie
Profile ekspresji genów, sygnatury transkryptomiczne
❌
tylko opis LINCS L1000 w §2.3, zero użycia
Dane kliniczne
❌
tylko opis w §2.4
GNN, Transformery, CNN
⚠️
jest, ale tylko GCN — bez GAT/GIN/GraphSAGE, choć opis mówi „Graph Neural Networks" w liczbie mnogiej
Cross-modal attention
✅
Faza 2
Late / early fusion
⚠️
tylko late fusion; early fusion nietestowany
Dynamiczna agregacja modalności
❌
brak
Wiedza dziedzinowa: ontologie biologiczne, profile aktywności ligandów, homologia białek
❌
to jest w tytule pracy — patrz niżej
BindingDB
✅


DrugBank, KEGG Drug, LINCS, ChEMBL
❌
żaden nieużyty
AUC, F1
✅


Scenariusze cold-start
❌
§2.6 buduje pod to cały wątek i nie ma realizacji
Wpływ wyboru cech i strategii fuzji
✅✅
najsilniejsza część pracy

Dwa najpoważniejsze rozjazdy
(a) „Integration of domain knowledge" w tytule nie ma pokrycia. Jedyna „wiedza dziedzinowa" w pracy to ECFP i wagi modeli pretrenowanych. To nie jest to, co rozumie się pod tym pojęciem — i praca sama to pokazuje, bo §2.6 opisuje KEPLA (GO + Ligand Properties w grafie wiedzy) jako wzorzec, którego potem nie realizuje. Dwie drogi:

tanio i skutecznie: dodać jedną gałąź z rzeczywistą wiedzą dziedzinową. Najniższy koszt: wektor ~200 deskryptorów fizykochemicznych RDKit (jak w MIFAM-DTI) albo one-hot/embedding rodziny białkowej (kinaza / GPCR / proteaza / …) lub terminów GO. To 2–3 dni pracy i jeden dodatkowy wiersz w tabeli 21 konfiguracji;
albo zmienić tytuł na coś w rodzaju „Multimodal deep learning for drug–target interaction prediction: a systematic comparison of molecular and protein representations" i przeformułować cele w §1.2.

Sugeruję pierwsze — tytuł jest dobry, a brakująca gałąź jest tania.

(b) Cold-start. §2.6 to półtorej strony wprowadzenia do problemu cold-start (DTIAM, KEPLA), a eksperymenty go nie badają. Podział scaffoldowy testuje generalizację po stronie ligandów, ale białka są wspólne dla train i test — to nie jest cold-start w sensie literatury. Model może zapamiętać a priori aktywności per-białko.

To jest najtańsza duża wygrana w całej pracy: pipeline już istnieje, wystarczy dodać trzy podziały i przetrenować 3–5 najlepszych konfiguracji:

unseen drug (już masz — scaffold split),
unseen target (podział po białkach; lepiej: klastrowanie MMseqs2 przy 40% identyczności, żeby homologi nie przeszły przez sito),
unseen both (iloczyn obu).

To zamienia „nie zrealizowano celu" w mocny rozdział 4.4 i daje realny materiał do rozdziału 5.


3. Czy poziom jest wystarczający na pracę magisterską?
Mocne strony (warte podkreślenia w rozdziale 5)
Design kombinatoryczny 7 × 3 = 21 jest poprawny i elegancki — pełna krata podzbiorów enkoderów. Warto podać wzór explicite, teraz czytelnik musi go odtworzyć.
Podział scaffoldowy jako domyślny, nie losowy — dobra decyzja metodologiczna, dobrze uzasadniona w §3.2.
Świadome szukanie biasów w danych (§3.3.5 długości, §3.3.8 MW/LogP, §3.3.9 skład aminokwasowy) — dokładnie to, czego się oczekuje przed treningiem.
Rozdział 4.3 (analiza błędów wg cech molekularnych) — samodzielne, nietrywialne myślenie. Obserwacja o próg pKi = 7,0 przecinającym gęsty obszar rozkładu (§3.3.6) jest bardzo dobra.
Rachunki się zgadzają tam, gdzie je sprawdziłem: 846,2 mln par teoretycznych ✓, 0,053% pokrycia ✓, 81 344/236 379 = 34,41% ✓, wzór pKi = 9 − log₁₀(Kᵢ[nM]) ✓, próg pKi ≥ 7 ⟺ Kᵢ ≤ 100 nM ✓, skalowanie MLP (1408 → 704/176; 512 → 256/64) ✓, dropout 0,3 + 0,05(N−2) → 0,3…0,45 ✓.
Słabości poziomu magisterskiego
Brak hiperparametrów (optymalizator, LR, batch, liczba epok, early stopping, seed). Praca jest nieodtwarzalna.
Brak wielokrotnych uruchomień, przedziałów ufności, testów istotności.
Brak jakiegokolwiek odniesienia ilościowego do literatury (TransformerCPI, MFD-GDrug, MCL-DTI są omówione na 4 stronach i nigdy nie porównane). Nawet jeśli porównanie 1:1 jest niemożliwe (inny zbiór, inny podział) — trzeba to napisać, a najlepiej dodać jeden przebieg na standardowym benchmarku (Human / C. elegans / BIOSNAP).
Brak zadania regresyjnego, mimo że pKᵢ jest ciągłe i cała literatura raportuje RMSE/CI na DAVIS/KIBA.
Brak sekcji o dostępności kodu i danych (repozytorium, wersja BindingDB, data pobrania).
Objętość: ~35 stron treści po odjęciu pustych rozdziałów. Po napisaniu 1 i 5 oraz dodaniu cold-startu będzie w porządku.


4. Błędy krytyczne — do naprawy bezwzględnie
4.1. Rysunki Fazy 1 nie zgadzają się z Tabelą 4.1
Odtworzyłem metryki z macierzy pomyłek i porównałem z tabelą:

Model
Rys. AUC
Tab. AUC
P z macierzy
Tab. P
R z macierzy
Tab. R
gcn_and_cnn (Rys. 4.1)
0,8925
0,8957
0,8128
0,8157
0,8038
0,8109
gcn_fp_chembert_and_cnn_esm2 (Rys. 4.2)
0,8820
0,8853
0,8102
0,8306
0,7872
0,7581
chembert_and_esm2 (Rys. 4.3)
0,8527
0,8591
0,7653
0,7754
0,7746
0,7724


Dla porównania — rysunki Fazy 2 zgadzają się z Tabelą 4.3 (gcn_chembert_and_cnn: 0,8858 vs 0,8856; P z macierzy 0,7877 vs 0,7881; R 0,8252 vs 0,8250 ✓).

Wniosek: tabela i rysunki Fazy 1 pochodzą z różnych uruchomień. Trzeba to uspójnić — wygenerować rysunki z tego samego checkpointu, z którego pochodzi tabela.

Ale jest tu też prezent. Ten rozjazd mierzy rozrzut międzyuruchomieniowy: 0,003–0,006 AUC. Tymczasem:

odstęp między 1. i 2. miejscem w Tabeli 4.1 to 0,0014 — poniżej szumu,
odstęp między 1. i 6. miejscem to 0,010 — ledwie 2–3× szum,
spadek Faza 1 → Faza 2 (~0,010) — również ledwie 2–3× szum.

Zdanie z §4.1 „the baseline gcn_and_cnn model achieved the highest overall AUC (0.8957)" jako podstawa rankingu jest nieuprawnione. Minimum: 3 ziarna losowe dla top-5 konfiguracji, raportowanie średniej ± odchylenia, i przeformułowanie wniosków na „grupa modeli z GCN jest nieodróżnialna od siebie i wyraźnie lepsza od modeli bez GCN". Ta ostatnia teza jest mocno wsparta danymi (0,89 vs 0,86 to ~10× szum) — i to powinna być główna konkluzja, a nie mikro-ranking.
4.2. Faza 2 ma dwie zmienne naraz — wnioski o cross-attention nie są uprawnione
§3.5.4 mówi, że w Fazie 2 jednocześnie (a) MLP zastąpiono cross-attention i (b) włączono dostrajanie LoRA na ChemBERTa/ESM-2 (Faza 1 używała zamrożonych, cache'owanych embeddingów). Spadek ~1 pp AUC nie może być przypisany cross-attention, bo zmieniły się dwie rzeczy — a nawet trzy, bo przejście z embeddingów cache'owanych na żywy forward pass to trzecia różnica.

Zdanie „the cross-attention approach did not improve the overall predictive performance" trzeba albo usunąć, albo dołożyć dwie komórki ablacji na 2–3 konfiguracjach: MLP + LoRA oraz cross-attention + frozen. Bez tego cała §4.2 wisi w powietrzu.
4.3. Sprzeczność między §3.3.8 a §4.3.2
§3.3.8: „the MW and LogP density curves for active and inactive pairs are nearly indistinguishable, confirming the absence of trivial size or solubility biases".
§4.3.2: „For metrics such as Molecular Weight and Heavy Atom Count, the True Positive distributions are consistently shifted to the right […] This proves that the networks […] correctly deduced that typical active pharmaceutical ingredients are generally larger".

Jeśli rozkłady MW dla klas są nierozróżnialne, to przesunięcie TP względem TN nie może świadczyć o odkryciu prawdziwej zależności — świadczy o tym, że model stosuje heurystykę rozmiaru, której w danych nie ma. Logika jest odwrócona: eksploatowanie korelacji z rozmiarem to skrót (shortcut), a nie dowód przeciw zapamiętywaniu. Do rozstrzygnięcia: porównać rozkłady TP vs FP (a nie TP vs TN) — jeśli FP też są przesunięte w prawo, to potwierdza skrót.

To samo dotyczy §4.3.4 (aromatyczność): §3.3.8 stwierdza, że aktywne mają zwykle 3 pierścienie aromatyczne, nieaktywne 2 — a §4.3.4 pokazuje, że model jest dokładniejszy dla ≥3 pierścieni. To spójny obraz skrótu wyuczonego z etykiet, nie „inductive bias". Trzeba to przeformułować.
4.4. Bilans danych się nie domyka
§3.1.2: start 3 187 250 rekordów → „over 2 million records (almost 80%) did not have a defined Ki" → „the remaining 600,000 drug-target pairs".

3 187 250 − 2 000 000 = 1 187 250, nie 600 000.
„prawie 80%" z 3 187 250 to ~2,55 mln — czyli filtr pojedynczego łańcucha musiał zredukować pulę wcześniej, ale ta liczba nie jest podana.
Dalej 600 000 → „roughly 450,000", przy udokumentowanych stratach ~1 000 błędnych SMILES i „a small subset" niestandardowych aminokwasów. Brakuje wyjaśnienia ~150 000 rekordów (25%).

Do zrobienia: tabela przepływu danych (jak CONSORT), wiersz na filtr, z liczbą przed i po: surowe → pojedynczy łańcuch → Kᵢ obecne → parsowanie numeryczne → walidacja RDKit → usunięcie 'X' → deduplikacja par → 451 858. Podać wersję BindingDB i datę pobrania.
4.5. Suma podziału ≠ rozmiar zbioru
317 600 + 44 695 + 90 103 = 452 398, a zbiór ma 451 858 rekordów. Różnica 540. Ponadto 317 600 to 70,29%, a nie 70%.
4.6. Deduplikacja par nie jest opisana
BindingDB ma wiele pomiarów Kᵢ dla tej samej pary. Nie wiadomo, czy 451 858 to unikalne pary czy rekordy pomiarowe, ani jak agregowano duplikaty (mediana? minimum? pierwszy?). Przy progu pKᵢ = 7,0 wybór agregacji zmienia etykiety. To musi być w §3.1.2.
4.7. Obsługa wartości ocenzurowanych jest metodologicznie wątpliwa
„«< 100» became just «100»" — ale „< 100 nM" znaczy „silniejsze niż 100 nM", czyli aktywny, a przypisanie dokładnie 100 daje pKᵢ = 7,0, czyli graniczny przypadek. Dla „> 10000" obcięcie jest nieszkodliwe, dla „<" — nie. Trzeba: podać ile rekordów miało operator, rozdzielić „<" i „>", i albo usunąć te z „<" w pobliżu progu, albo uzasadnić decyzję.
4.8. Porównania precision/recall przy stałym progu 0,5 nie są porównaniami architektur
Cała argumentacja z §4.1 („gcn_and_cnn_esm2 daje najszerszą siatkę bezpieczeństwa dla wirtualnego przesiewu", „fp_chembert_and_esm2 jest ekstremalnie konserwatywny") opiera się na P/R przy progu 0,5. Ale kolumna Loss w Tabeli 4.1 waha się od 0,4460 do 0,7650 przy niemal identycznym AUC — to znaczy, że modele są różnie skalibrowane, nie że mają różny profil wykrywania. Każdy z nich można przesunąć po własnej krzywej ROC do dowolnego recall.

Do zrobienia: albo wybierać próg na zbiorze walidacyjnym per model, albo raportować metryki przy dopasowanym recall (np. precision @ recall = 0,80), albo — najlepiej dla narracji o wirtualnym przesiewie — dodać enrichment factor / hit rate w top-k%. To jest metryka, która faktycznie odpowiada na pytanie „ile prawdziwych trafień w budżecie 1000 testów laboratoryjnych".

Przy okazji: „unprecedented number of False Positives (10,447)" (§4.2) — w Fazie 1 chembert_and_esm2 miał 10 636. Poprawić na „najwięcej wśród modeli Fazy 2".
4.9. Mechanizm „fair benchmarking" wprowadza własny confound
Adaptacyjny dropout 0,3 + 0,05(N−2) sprawia, że modele 5-enkoderowe dostają regularyzację 0,45, a 2-enkoderowe 0,30. Więc różnica między nimi to łącznie efekt zestawu enkoderów i efekt siły regularyzacji. Nie można potem pisać (§4.1), że pogorszenie wynika z „curse of dimensionality" — mogło wynikać z 50% wyższego dropoutu. To samo dotyczy skalowanej szerokości MLP.

Rozwiązanie: kontrolna seria z ustalonym dropoutem 0,3 dla 3–4 konfiguracji o różnym N. Plus uczciwe zdanie w §3.5.4, że mechanizm wyrównujący sam jest zmienną.
4.10. „Curse of dimensionality" jest używane jako uniwersalne wyjaśnienie i prawdopodobnie błędnie
Termin pojawia się trzy razy (§4.1, §4.2, opisy Rys. 4.2 i 4.5) jako wyjaśnienie pogorszenia. Przy 1408 wymiarach i 317 600 przykładach treningowych to nie jest przekleństwo wymiarowości — to raczej redundancja cech, trudniejsza optymalizacja i mocniejsza regularyzacja (§4.9). Sugeruję zastąpić konkretnym mechanizmem albo uczciwym „przyczyna nie została ustalona".
4.11. Metryki pooled są zdominowane przez huby
§3.3.4 sam pokazuje rozkład potęgowy: jedno białko ma >8000 ligandów, mediana ma kilka. Globalne AUC liczone po wszystkich parach jest więc zdominowane przez garść najlepiej zbadanych celów. Dodać AUC/AUPRC makro-uśrednione per białko (średnia po celach z ≥N par). Bardzo prawdopodobne, że ranking się zmieni — i to będzie ciekawy wynik.
4.12. Architektura fuzji Fazy 2 jest niedospecyfikowana
§3.5.4 pisze, że cechy leku „can directly attend to specific amino acid motifs of the protein". Ale §3.5.2/3.5.3 mówią, że embeddingi ESM-2/ChemBERTa były cache'owane i rzutowane do wektora 1024-wymiarowego. Jeśli to wektory zpoolowane, to cross-attention działa na sekwencjach długości 1 — czyli jest zdegenerowany, co zresztą praca sama przeczuwa w §4.2 („the module tries to align highly compressed global vectors"). Trzeba jednoznacznie podać: co jest Q, K, V, jakiej długości sekwencje, ile głów, czy pooling był po CLS czy średni. To jest największa luka w odtwarzalności całej pracy.


5. Weryfikacja bibliografii — halucynacje
Dobra wiadomość: wszystkie 17 pozycji istnieją. Sprawdziłem każdą.

#
Pozycja
Status
[1]
Fan, Fu, Zhang — Progress in molecular docking, Quant. Biol. 7(2):83–89, 2019
✅ DOI 10.1007/s40484-019-0172-y
[2]
Verdonk i in. — GOLD, Proteins 52(4):609–623, 2003
✅
[3]
Lang i in. — DOCK 6, RNA 15(6):1219–1230, 2009
✅
[4]
Chen i in. — FRoGS, Nat. Commun. 15:1853, 2024
✅ DOI 10.1038/s41467-024-46089-y (29.02.2024)
[5]
Weininger — SMILES, JCICS 28(1):31–36, 1988
✅
[6]
Li i in. — MIFAM-DTI, Front. Genet. 15:1381997, 2024
✅ DOI 10.3389/fgene.2024.1381997
[7]
Landrum i in. — RDKit, Zenodo
✅
[8]
Chen i in. — TransformerCPI, Bioinformatics 36(16):4406–4414, 2020
✅
[9]
Mikolov i in. — word2vec, arXiv:1301.3781
✅
[10]
Zhang i in. — GCN: a comprehensive review, Comput. Soc. Netw. 6(1), 2019
✅ istnieje, ale zła w tym miejscu — patrz niżej
[11]
Gu i in. — MFD-GDrug, Methods 223:75–82, 2024
✅ DOI 10.1016/j.ymeth.2024.01.017
[12]
Lin i in. — ESM-2/ESMFold, Science 379(6637):1123–1130, 2023
✅
[13]
Jaeger, Fulle, Turk — Mol2vec, JCIM 58(1):27–35, 2018
✅
[14]
Qian i in. — MCL-DTI, BMC Bioinformatics 24(1):323, 2023
✅ DOI 10.1186/s12859-023-05447-1
[15]
Zhang i in. — arXiv:2407.04055
✅ ale zły tytuł — patrz niżej
[16]
Lu i in. — DTIAM, Nat. Commun. 16(1):2548, 2025
✅ DOI 10.1038/s41467-025-57828-0
[17]
Liu i in. — KEPLA, arXiv:2506.13196
✅ (Han Liu, Keyan Ding, Peilin Chen, Yinwei Wei, Liqiang Nie, Dapeng Wu, Shiqi Wang)

Problemy z cytowaniami (nie halucynacje, ale błędy)
[15] ma tytuł z wersji v1. arXiv:2407.04055 v1 (07.2024) = „…from a Structure Perspective"; v2 (11.2025) = „…from a Drug Structure Perspective". Bibliografia ma v1, tekst §2.5 ma v2. Uspójnić i podać numer wersji + datę dostępu.

[10] jest cytowane jako źródło GCN („Graph Convolutional Networks (GCN) [10]"). To przegląd, nie praca oryginalna. Dodać Kipf & Welling, ICLR 2017 — to jest właściwe cytowanie dla GCNConv, którego praca faktycznie używa.

[4] jest cytowane w dwóch niepasujących rolach: raz jako dowód na zwrot pola w stronę deep learningu (§2.1), raz jako źródło faktów o LINCS L1000 i „1,3 mln profili" (§2.3). FRoGS nie jest przeglądem pola, a liczba 1,3 mln pochodzi z jego wstępu, który cytuje Subramanian i in. Dodać Subramanian i in., Cell 171(6):1437–1452, 2017 jako źródło pierwotne dla L1000.

„ESM32" (§2.6, dwa razy — przy DTIAM i KEPLA). Taki model nie istnieje. Powinno być „ESM-2" (albo — sprawdź w źródłach — DTIAM i KEPLA mogą używać ESM-1b; KEPLA pisze ogólnie „pre-trained ESM model").

Twierdzenia o pracach źródłowych do zweryfikowania z tekstem oryginałów:

„removing the GCN or the convolutional networks led to […] decreasing the accuracy by as much as 18 percentage points" (MFD-GDrug) — sprawdź w tabeli ablacji;
„with only 20% of samples, DTIAM can outperform models learned on 80%" — sprawdź;
„The AutoML model receives drug and protein embeddings" (DTIAM) — publikacja opisuje „downstream drug-target prediction module", nie AutoML;
„MIFAM-DTI utilizes […] 202 parameters calculated using the RDKit library" — sprawdź liczbę. Uwaga: MIFAM-DTI używa MACCS (nie ECFP) i ESM-1b (nie ESM-2), a wymiar redukuje PCA. Warto to poprawić w §2.4;
„KEPLA […] outperform not only interaction-free methods but also structure-based methods (which require 3D coordinates)" — abstrakt mówi tylko o „state-of-the-art baselines" na dwóch benchmarkach.

Nadmierna generalizacja: MFD-GDrug jest modelem specyficznym dla GPCR. Wniosek z jego ablacji („local features are more critical than global dependencies", §2.5) nie przenosi się automatycznie na cały DTI. Dodać zastrzeżenie.

Napięcie MFD-GDrug ↔ benchmark [15] jest dobrze zauważone (§2.5), ale rozumowanie jest niedokładne: ablacja (usuwanie modułu z modelu, w którym pozostają inne enkodery) to nie to samo co porównanie samodzielnych enkoderów. Warto to dopowiedzieć — to jedno zdanie, a pokazuje, że autor rozumie różnicę.
Cytowania, których brakuje (obowiązkowo do dodania)
Kluczowe komponenty pracy nie mają cytowań w ogóle albo tylko przypis z URL-em:

BindingDB — Gilson i in., BindingDB in 2015, Nucleic Acids Res. 44(D1):D1045–D1053, 2016 (obecnie tylko przypis z linkiem)
ChemBERTa — Chithrananda, Grand, Ramsundar, arXiv:2010.09885, 2020 — całkowicie niecytowane, a to rdzeń jednego z trzech enkoderów leku
LoRA — Hu i in., arXiv:2106.09685, 2021 — niecytowane, a to cała Faza 2
GCN — Kipf & Welling, ICLR 2017
ECFP — Rogers & Hahn, JCIM 50(5):742–754, 2010
Scaffold Bemisa–Murcko — Bemis & Murcko, J. Med. Chem. 39(15):2887–2893, 1996
scikit-fingerprints — Adamczyk & Ludynia, Scikit-fingerprints: Easy and efficient computation of molecular fingerprints in Python, SoftwareX 28:101944, 2024, DOI 10.1016/j.softx.2024.101944 (obecnie tylko przypis z URL-em)
OGB — Hu i in., NeurIPS 2020 (obecnie przypis)
t-SNE — van der Maaten & Hinton, JMLR 9:2579–2605, 2008
Podział scaffoldowy jako protokół — Wu i in., MoleculeNet, Chem. Sci. 9:513–530, 2018
Transformer / BERT / RoBERTa — Vaswani i in. 2017; Devlin i in. 2019; Liu i in. 2019 (BERT jest wymieniony z nazwy bez cytowania)
PyTorch, PyTorch Geometric (Fey & Lenssen 2019) — dla odtwarzalności


6. Błędy merytoryczne i terminologiczne
Poważne
§3.5.3: „ChemBERTa […] Built upon the […] BERT architecture […] pre-trained on a massive corpus of over 100,000 SMILES strings". Dwa błędy faktograficzne:

ChemBERTa jest oparta na RoBERTa, nie BERT;
korpus to 77 mln unikalnych SMILES z PubChem; liczba 100 000 to najmniejszy podzbiór z ablacji skalowania w oryginalnej pracy (100K / 250K / 1M / 10M). Nazwanie 100 tys. „massive" jest wewnętrznie sprzeczne.
Dodatkowo: nie podano, który checkpoint użyto. To ma duże znaczenie (seyonec/ChemBERTa-zinc-base-v1 = ZINC-250k, DeepChem/ChemBERTa-77M-MLM = PubChem 77M). Musi być w §3.5.3.

§3.1.1, definicja Kᵢ: „It represents the exact concentration needed to fill 50% of the target sites". To jest opis Kd/IC₅₀, nie Kᵢ. Kᵢ to stała dysocjacji kompleksu inhibitor–cel wyznaczana z kinetyki inhibicji. Przeformułować: „Kᵢ to stała inhibicji, czyli stała dysocjacji kompleksu inhibitor–białko; niższe Kᵢ oznacza silniejsze wiązanie."

§3.4.1: „Global Add Pooling […] return[s] a single, fixed-size vector of 256 dimensions representing the physicochemical properties of the drug molecule". Sumowanie embeddingów węzłów daje reprezentację topologiczno-strukturalną, nie fizykochemiczną. Zwłaszcza że praca ma osobną gałąź ECFP i osobno mówi o deskryptorach fizykochemicznych. Zmienić na „strukturalną/topologiczną".

§3.4.2: „batch normalization layers are used to stabilize the training and prevent over-fitting and the cold start problem". BatchNorm nie zapobiega problemowi cold-start. Cold-start to własność protokołu podziału, nie warstwy sieci. Usunąć drugą część zdania.

§3.3.9 i §4.3.2: „negative sampling" i „inactive decoy molecules". W tej pracy nie ma negatywnego próbkowania — negatywy to zmierzone słabe wiązania (pKᵢ < 7). To nie decoye. Ma to duże znaczenie interpretacyjne: słabe wiązania są trudniejszymi negatywami niż losowe decoye, więc AUC ≈ 0,89 jest lepszym wynikiem, niż wygląda. Warto to wykorzystać jako argument, ale terminologię trzeba naprawić w obu miejscach.

§2.6, definicja cold-start: „refers to a situation in which the model shows high performance on test datasets but exhibits degraded performance when presented with previously unseen drugs or proteins". To opis objawu, nie definicja. Cold-start to protokół ewaluacji: zbiór testowy zawiera leki i/lub białka nieobecne w treningu (unseen-drug, unseen-target, unseen-both). Poprawić — zwłaszcza że to fundament pod nowy rozdział 4.4.

§3.3.4: „The interaction network follows a power-law degree distribution". Twierdzenie o rozkładzie potęgowym bez dopasowania. Albo dopasować (biblioteka powerlaw, estymacja α, test KS, porównanie z log-normalnym), albo osłabić do „ciężkoogonowy, w przybliżeniu bezskalowy". Na WI AGH to zdanie zostanie sprawdzone.

§3.2: podział scaffoldowy — brak specyfikacji. Z Rys. 3.5 wynika, że użyto scaffoldów generycznych (top scaffold to C1CCCCC1, czyli po wymazaniu typów atomów benzen i cykloheksan się zlewają). To uzasadniona i surowsza wersja, ale musi być napisane wprost: MurckoScaffoldSmiles vs MakeScaffoldGeneric, jak przypisano grupy scaffoldów do podziałów (deterministycznie od największej?), i co zrobiono z cząsteczkami bez pierścieni (pusty scaffold).
Mniejsze
Niespójność obcięcia: §3.3.3 mówi 256 dla leków i 1024 dla białek; §3.4.2 mówi 1000 znaków dla CNN. Prawdopodobnie CNN = 1000, ESM-2 = 1024 — trzeba to powiedzieć jawnie.
§3.5.2: „ESM-2 […] a context length of 1024 tokens". ESM-2 używa rotary embeddings i nie ma twardego limitu 1024 — to wybrane obcięcie. Przeformułować.
§3.5.1: „ECFP, which are mathematically equivalent to Morgan fingerprints". Zbyt mocne — ECFP to warianty odcisków Morgana/cyrkularnych z określonymi niezmiennikami atomowymi. „Are a variant of" wystarczy.
§3.5.1: „a radius of 2 was applied (capturing interactions up to two bonds away)". Nie „interactions" — „substructures/atom environments".
§3.3.7: obiecane liczby nie zostały podane. „By quantifying how many distinct scaffolds are required to cover 50% and 80% of the unique molecules, the analysis confirms…" — liczb nie ma. Dodać albo usunąć zdanie.
Rys. 3.5: podpis nie odpowiada rysunkowi. Podpis obiecuje „cumulative molecule coverage by scaffold rank", a rysunek pokazuje tylko słupki top-15. Tekst też odwołuje się do „log-scaled distribution of molecules per scaffold (Figure 3.5)", czego na rysunku nie ma.
§3.4.2: brak liczby kanałów w blokach CNN. Wynik 384 = 3 × 128 daje się odgadnąć, ale trzeba podać.
Brak wymiarów wyjściowych enkodera FP-MLP. Bez tego nie da się odtworzyć, skąd 1408 i 512 w przykładach z §3.5.4. Dodać tabelę: enkoder → wymiar wyjściowy, oraz wymiar wektora fuzji dla każdej z 21 konfiguracji (choćby w załączniku).
§4.2: „the 10 successfully trained architectures". Słowo „successfully" sugeruje, że niektóre się nie udały (OOM? crash?). Albo wyjaśnić, albo usunąć. Skład dziesiątki jest poprawnie zgodny z top-10 po AUC z Tabeli 4.1 — sprawdziłem.
§4.1 vs Tabela 4.2: „the straightforward CNN architecture consistently outperforms ESM-2". Dla AUC tak (7/7). Dla F1 nie — wiersz Fingerprint: 0,7738 (CNN) vs 0,7873 (ESM-2). Dodać zastrzeżenie.
§2.5: tabela bez podpisu i numeru (osiem par enkoderów z benchmarku). Nadać \caption i \label, dodać cytowanie w podpisie.
§2.6: „protein sequences are processed into singular residuals" → „residues" (reszty aminokwasowe).
§2.6: „has two training phases: -" — zbłąkany dywiz.
Brak wzmianki o możliwym przecieku wiedzy przez pretrenowanie: ESM-2 i ChemBERTa widziały te białka i cząsteczki w korpusach publicznych. To nie przeciek etykiet, ale jedno zdanie w dyskusji ograniczeń jest na miejscu.


7. Artefakty AI i język
Tekst czyta się jak wygenerowany albo mocno przeredagowany przez LLM. To trzeba naprawić, bo dziś jest to sprawdzane — i bo obniża wiarygodność części merytorycznej. Konkretne wzorce:

(a) Meta-narracja rozdziałów. §3, wstęp: „This chapter presents the comprehensive methodology […] smoothly transitioning from rigorous raw data curation […] The narrative guides the reader through foundational feature extraction techniques before culminating in the integration of highly advanced, pre-trained multi-modal fusion networks." To czysty wypełniacz. Zamienić na trzy zdania faktograficznej mapy rozdziału albo usunąć. Podobnie §3.1 i §3.5.

(b) Wtrącenia myślnikiem/półpauzą jako aparycja stylistyczna. Wzorzec „X-consisting of Y-was introduced" jest charakterystyczny:

„a projection head-consisting of a Linear layer, Layer Normalization, and a ReLU activation function-was introduced"
„Evolutionary Scale Modeling (specifically ESM-2)-a state-of-the-art transformer-based protein language model-was utilized"
„«gene signatures»-large-scale profiles characterizing…"
„The diverse data modalities described in the previous section-ranging from […] metadata-represent a rich but heterogeneous information space"

W PDF-ie renderują się jako zwykłe dywizy, co dodatkowo wygląda źle typograficznie. Rozwiązanie: przepisać na zdania podrzędne lub nawiasy, a tam gdzie myślnik ma zostać — użyć --- w LaTeX-u. Docelowo w całej pracy powinno zostać najwyżej kilka takich wtrąceń.

(c) Przymiotniki marketingowe / superlatywy. Do wycięcia lub zastąpienia liczbą:

phenomenally robust (§4.3.5), massive spike / massive overlap, absolute necessity, exceptionally well-balanced (rozkład 48/52 — po prostu „zbalansowany"), unprecedented number, severe impact, the widest and safest safety net, groundbreaking drug, aggressive interaction hunting, boasts the highest Precision, highly cohesive separation, smoothly transitioning, culminating in, highly advanced, the absolute bottom, at the absolute bottom of the evaluation table.

Osobno: „rigorous"/„rigorously" i „comprehensive" występują ~6 i ~8 razy. Zredukować do 1–2 wystąpień każde.

(d) Personifikacja modeli. „models […] act completely differently", „the model boasts", „The attention heads are overwhelmed", „ChemBERTa introduced structural noise that effectively tricked the attention mechanism", „casting a wider net". Ładne obrazowo, ale w pracy technicznej to za dużo. Zostawić jedno–dwa, resztę zamienić na opis mechanizmu.

(e) Nadmierna moc twierdzeń. „This proves that…" pojawia się co najmniej cztery razy przy dowodach korelacyjnych; do tego „confirms", „This confirms that the dataset is unbiased", „prove that the negative sampling […] did not introduce a […] bias". Zastąpić: suggests, is consistent with, we found no evidence of. To nie jest kosmetyka — obecne sformułowania są po prostu nieprawdziwe logicznie (§4.3 wyżej).

(f) Termin z wyższej półki użyty niepoprawnie — to najbardziej zdradliwy sygnał. „Curse of dimensionality" (§4.10) i „inductive bias" używane luźno. Recenzent, który zapyta „a dlaczego to przekleństwo wymiarowości przy 300 tys. przykładach?", postawi autora w trudnej sytuacji na obronie. Lepiej napisać prościej i prawdziwie.

(g) Niespójność osoby i głosu. Dominuje bezosobowa strona bierna, ale wchodzą „our dataset" (§3.3.2) i „we can analyze" (§4.3). Wybrać jedną konwencję — na AGH standardem jest forma bezosobowa.

(h) Niespójność nazw (to też sygnał składania tekstu z kawałków):

ChemBERTa vs ChemBERT (Rys. 3.12, 3.13: „Cached ChemBERT Embedding")
ECFP vs ECPF (Rys. 3.10 i jej podpis: „Fingerprint (ECPF) Architecture")
ESM-2 / ESM2 / ESM32
gcn_and_cnn w tekście vs gcn_vs_cnn w tytułach rysunków — to szczególnie myli, bo „vs" sugeruje porównanie dwóch modeli, a chodzi o jeden model łączący dwa enkodery. Ujednolicić na GCN+CNN
multimodal / multi-modal (także tytuł pracy vs tytuł rozdz. 3)
1D-CNN / CNN / Conv1D
MFD-GDrug / MFD-GDRUG

(i) Zalecenie formalne: sprawdź wymagania AGH dotyczące oświadczenia o wykorzystaniu narzędzi AI. Niezależnie od zakresu użycia, tekst wymaga przejścia „ludzkiego" — nie kosmetycznego, ale przepisania fragmentów wymienionych wyżej we własnym, prostszym rejestrze.


8. Poprawki redakcyjne i edytorskie
Braki formalne
Rozdział 1 (Motivation, Research Objectives) — pusty. Cele muszą być sformułowane jako weryfikowalne pytania badawcze, do których rozdz. 5 wprost odpowie.
Rozdział 5 (Summary and Conclusions) — pusty. Powinien zawierać: odpowiedzi na pytania z §1.2, listę ograniczeń (single run, brak cold-startu jeśli nie zostanie dodany, brak porównania z literaturą, negatywy jako słabe wiązania a nie decoye), kierunki dalszych prac.
Brak streszczenia w języku polskim i angielskim.
Brak spisu rysunków i spisu tabel.
Brak wykazu skrótów — przy DTI, GCN, CNN, ESM, ECFP, LoRA, MLP, AUPRC, GO, TAR czytelnik będzie wdzięczny.
Brak sekcji o odtwarzalności: repozytorium kodu, wersja BindingDB + data pobrania, wersje bibliotek (RDKit, PyG, transformers, scikit-fingerprints), sprzęt, czas treningu, liczba parametrów per konfiguracja.
Brak tabeli syntetyzującej related work (metoda × modalności × zbiory × metryki × cold-start). Rozdział 2 to obecnie ciąg opisów; tabela na końcu §2.5 zamieniłaby go w przegląd.
Rysunki
Kolizje tekstu: na Rys. 4.2c, 4.4c i 4.5c tytuł macierzy pomyłek nachodzi na etykietę paska kolorów („…cnn_esm2" + „35000"). Naprawić przez tight_layout() / krótsze tytuły / fontsize.
Rys. 4.7 jest nieczytelny w druku — 12 podwykresów × 2 modele na jednej stronie. Podzielić na dwa rysunki albo przenieść do załącznika, a w tekście zostawić 3–4 najważniejsze cechy.
Rys. 3.5: osi Y są ciągi SMILES — nieczytelne i nieinformatywne. Znacznie lepiej: siatka struktur (RDKit MolsToGridImage) z liczbami pod spodem.
Wszystkie wykresy matplotlib eksportować jako PDF/wektor, nie raster.
Rys. 3.1: podpis mówi „truncated at the 99th percentile", a tytuły osi „cut off 1% extremes > 250" / „> 2004". Uspójnić.
Tabele
Tabela 4.3 jest za szeroka (12 kolumn numerycznych, mikroskopijna czcionka). Podzielić na dwie (Faza 1 / Faza 2) albo raportować tylko ΔAUC, ΔF1 itd. z odsyłaczem do Tabeli 4.1.
Podpisy tabel muszą zawierać protokół: „zbiór testowy (90 103 pary), próg klasyfikacji 0,5, jedno uruchomienie, ziarno = X".
Tabela w §2.5 — bez podpisu i numeru (patrz §6.19).
Typografia i notacja
Kᵢ i pKᵢ — konsekwentnie w trybie matematycznym ($K_i$, $pK_i$); teraz mieszane „Ki" i $pK_i$.
Jednostki z niełamliwą spacją: 100\,nM, 500\,Da, 10\,\mu\text{M}.
Nazwy modeli w \texttt{} konsekwentnie (teraz raz gcn and cnn, raz gcn\_and\_cnn).
Niełamliwe spacje przed odsyłaczami: Figure~\ref{...}, Table~\ref{...}.
BibTeX: chronić nazwy własne i akronimy nawiasami klamrowymi — {SMILES}, {ESM}-2, {GPCR}, {DOCK}, {GOLD} — inaczej styl je zamieni na małe litery. Ujednolicić kapitalizację tytułów (teraz mieszanka „Progress in molecular docking" i „Improved protein–ligand docking using GOLD").
Tytuł rozdz. 2 „Related work" — kapitalizacja niezgodna z pozostałymi rozdziałami.
Przypisy z URL-ami (BindingDB, OGB, scikit-fingerprints) zamienić na pełne cytowania + datę dostępu.


9. Plan działań — priorytety
Priorytet A — bez tego nie ma obrony (szacunkowo 2–3 tygodnie)
Napisać rozdziały 1 i 5. Rozdz. 1 formułuje 3–4 pytania badawcze; rozdz. 5 odpowiada na nie punkt po punkcie plus ograniczenia.
Uspójnić rysunki Fazy 1 z Tabelą 4.1 (§4.1).
Naprawić bilans danych i sumy podziału (§4.4, 4.5), dodać tabelę przepływu danych i opis deduplikacji (§4.6) oraz obsługi operatorów </> (§4.7).
3 ziarna losowe dla top-5 konfiguracji, raportować średnią ± σ, przeformułować wnioski §4.1 na „grupa z GCN vs bez GCN" (§4.1).
Usunąć sprzeczność §3.3.8 ↔ §4.3.2 i odwróconą logikę „proves that the networks did not memorize" (§4.3).
Uzupełnić hiperparametry i specyfikację modułu fuzji Fazy 2 (§4.12) — tabela wymiarów enkoderów, Q/K/V, pooling, LR, epoki, early stopping, sprzęt.
Poprawić błędy faktograficzne: ChemBERTa (RoBERTa, 77M, checkpoint), Kᵢ, sum pooling, BatchNorm/cold-start, „decoys", „ESM32" (§6.1–6.5).
Dodać brakujące cytowania — ChemBERTa, LoRA, Kipf & Welling, BindingDB, ECFP, Bemis–Murcko, scikit-fingerprints, OGB, t-SNE (§5).
Przejście redakcyjne po artefaktach AI z §7 (a)–(h).
Priorytet B — podnosi ocenę z 4,5 na 5 (szacunkowo dodatkowe 2–3 tygodnie)
Eksperyment cold-start — trzy podziały (unseen drug / unseen target / unseen both), 3–5 konfiguracji. To domyka najważniejszą lukę względem opisu tematu i daje treść rozdz. 4.4 oraz mocne wnioski w rozdz. 5.
Jedna gałąź z prawdziwą wiedzą dziedzinową — deskryptory fizykochemiczne RDKit albo rodzina białkowa / terminy GO. Domyka tytuł pracy.
Rozplecenie confoundu Fazy 2 — ablacja MLP+LoRA i cross-attention+frozen (§4.2).
Metryki makro per białko obok pooled (§4.11).
Metryki wirtualnego przesiewu — enrichment factor / precision@k, zamiast P/R przy stałym progu 0,5 (§4.8).
Priorytet C — jeśli zostanie czas
Analiza kalibracji (reliability diagram, ECE, Brier) — praca już mówi o „overconfidence", warto to zmierzyć, a nie wnioskować z histogramów.
Wariant regresyjny na pKᵢ (RMSE, Pearson, Spearman, CI) — otwiera porównanie z DAVIS/KIBA.
Ablacja pasma niepewności: usunąć pKᵢ ∈ [6,5; 7,5] i przeliczyć — bezpośrednio testuje hipotezę z §3.3.6.
Jeden przebieg na standardowym benchmarku (Human / C. elegans / BIOSNAP) dla zakotwiczenia w literaturze.
Kontrola confoundu adaptacyjnego dropoutu (§4.9).
Dopasowanie rozkładu potęgowego z testem statystycznym (§6.7).


10. Podsumowanie dla studenta (wersja krótka)
Masz dobry, samodzielny eksperyment i jedną naprawdę mocną tezę: jawna topologia grafu (GCN) jest lepszym induktywnym obciążeniem dla DTI niż odciski o stałej długości albo kontekstowe reprezentacje SMILES, a naiwne doklejanie kolejnych modalności nie pomaga. Ta teza wytrzyma obronę.

Czego brakuje: (1) dwóch rozdziałów, (2) spójności liczb między tabelami a rysunkami, (3) powtórzeń uruchomień, żeby ranking miał sens, (4) eksperymentu cold-start, który sam obiecałeś w rozdz. 2, (5) czegokolwiek, co uzasadnia „domain knowledge" w tytule.

Bibliografia jest w porządku — wszystkie 17 pozycji istnieją, co przy tej ilości prac z lat 2024–2025 nie jest oczywiste. Ale brakuje cytowań do rzeczy, których faktycznie używasz (ChemBERTa, LoRA, ECFP, Kipf & Welling), i to jest łatwiejsze do zauważenia niż zły numer strony.

I ostatnia rzecz: pisz prościej. „Phenomenally robust" i „curse of dimensionality" nie dodają wiarygodności — odbierają ją, bo pierwsze jest pustym superlatywem, a drugie prawdopodobnie nie jest właściwym wyjaśnieniem tego, co widzisz w danych. Zdanie „nie ustaliliśmy przyczyny tego spadku" jest w pracy magisterskiej mocniejsze niż nietrafiony termin z wyższej półki.


