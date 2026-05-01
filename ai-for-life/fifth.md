15.04.2026
# The role of AI in Drug Discovery (Joerg Wichard, AI-Team, Selvita)
## Selvita
- 40% pracownikow ma PhD
## Main part
- high asset of data is most important
- ADMET: Absorption, Distribiution, Metabolism, Excretion, Toxicity

### QSAR - Quantitative Structure Activity Relation
Do przewidywana wlasciwosci biologicznych czasteczek (najprostszy przyklad, ile mg aspiriyny rozpusci sie w wodzie)

- Descriptors - liczbowe wartosci opisujace cechy danej czasteczki np masa czasteczokwa, fingerptinty itp

# Deep dive into molecular fingerprints (Jakub Adamczyk, MLCIL, Faculty of Computer Science, AGH)

## Molecular graph
- smiles
- matematically:
    - attributed graph, topology (structure), atom features, bond features
flat 2d representation
## EMbedding
- feature vector for molecular graph, usable by ml models, obtained by feature extraction algorithm

## Molecular fingerprints
- automated feature extraction:
    - input: molecular graph
    - output: feature vector
- deterministic
### Hashed fingerpints
idea:
- define a shape of subgraph e.g atom pair, circural nehgiborhoods
- each unique getds int ID and gets mapped to feautre vector
hashing:
- mapping number into smaller range
### High level picture
- define subgraph shape
- calculate all subgraph
- ?
- map each id into index with hashing funtion
ECPF - circual neighbours
atomic pairs - shortest path between 2 nodes
mapped fingerprints- polaczenie ecpf z atomic pairs
