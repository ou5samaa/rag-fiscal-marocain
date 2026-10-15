# Assistant RAG pour le Code général des impôts marocain

Ce projet permet d’interroger en français les éditions 2023 à 2026 du Code général des impôts (CGI) marocain. L’utilisateur choisit une édition, pose une question et obtient une réponse accompagnée des articles et des pages du PDF sur lesquels elle s’appuie. L’objectif est de retrouver **la bonne disposition dans la bonne édition**, même lorsque plusieurs versions contiennent des passages proches.

## Ce que fait l’application

- Recherche dans les PDF du CGI en respectant l’édition sélectionnée.
- Affiche une réponse rédigée, les passages cités et les pages sources consultables.
- S’abstient ou rejette une réponse lorsque les preuves disponibles sont insuffisantes.
- Fonctionne localement avec une interface Streamlit et un modèle Qwen servi par Ollama.

Une question sur le contenu d’une édition ne détermine pas nécessairement la règle applicable à une opération datée : les dispositions transitoires et d’autres textes peuvent intervenir.

## Méthode

Les PDF sont transformés en articles avec leur hiérarchie, leurs notes et leurs références de pages. Les passages sont recherchés par **BM25** et par similarité sémantique avec **multilingual E5**, stockée dans **Chroma**. Leurs classements sont fusionnés après filtrage par édition. Un contexte plus large autour des passages retenus est transmis à **Qwen** ; les citations sont ensuite rattachées aux lignes sources et contrôlées. **MLflow** conserve les traces des nouvelles expériences locales.

```text
PDF → articles et pages → BM25 + E5/Chroma → filtre d’édition
    → contexte parent → Qwen → réponse citée → contrôle des preuves
```

## Résultats observés

| Évaluation | Recherche de la preuve annotée au Top 5 | Réponses délivrées |
| --- | ---: | ---: |
| Campagne consolidée, 76 questions issues de plusieurs phases | 68/76 | 74/76 |
| Test supplémentaire, 8 nouvelles questions | 7/8 | 7/8 |

Ces chiffres décrivent des **lots limités**, et « réponse délivrée » ne signifie pas « réponse toujours exacte ». La campagne consolidée réunit des résultats obtenus à différentes étapes ; elle ne constitue pas un nouveau test simultané. Les erreurs, abstentions et critères de mesure sont détaillés dans le [rapport d’évaluation](docs/03_evaluation.pdf).

## Contenu du dépôt

- [Notebook de démonstration](demo.ipynb) : parcours commenté des sources, de la recherche, des réponses citées et de l’évaluation.
- `rag_fiscal/` : extraction, recherche, génération et vérification des citations.
- `sources/` : PDF originaux et manifeste de provenance ; `benchmark/` : questions annotées.
- `data/candidates/full_cgi/` : corpus et index prêts à consulter ; `data/evaluation/` : résultats enregistrés.
- `streamlit_app.py` : interface ; `config/` : paramètres de la chaîne ; `docs/` : rapports PDF.

## Démarrage sur Windows

Prévoir Python 3.11 ou 3.12 et, pour la recherche hybride, un GPU NVIDIA compatible CUDA. Depuis la racine du dépôt :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Installer ensuite une version CUDA de PyTorch adaptée au GPU selon [pytorch.org](https://pytorch.org/get-started/locally/), puis vérifier que `torch.cuda.is_available()` renvoie `True`. Les poids E5 et Qwen ne sont pas inclus dans le dépôt ; les commandes suivantes les installent aux versions contrôlées par le projet :

```powershell
.\.venv\Scripts\python.exe scripts\install_e5.py
.\.venv\Scripts\python.exe scripts\install_ollama.py
.\scripts\start_ollama.ps1
```

Dans un **deuxième terminal PowerShell**, télécharger Qwen sur le serveur Ollama local :

```powershell
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_MODELS = (Join-Path (Get-Location) 'models/ollama')
.\runtime\ollama\0.40.1\ollama.exe pull qwen3:4b-instruct-2507-q4_K_M
.\scripts\start_ui.ps1
```

Ouvrir **http://127.0.0.1:8501**, choisir l’édition et consulter la page PDF sous une citation. Pour arrêter Ollama : `.\scripts\start_ollama.ps1 -Stop`.

Un contrôle des sources, de l’index et du filtre d’édition fonctionne sans télécharger les modèles :

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Le notebook lit les résultats déjà enregistrés ; seule la question libre lance une nouvelle génération. La recherche lexicale BM25 peut aussi être utilisée sans GPU. Les PDF proviennent des sources indiquées dans `sources/sources_manifest.json`. L’archive préparée pour GitHub contient le corpus et les index, mais pas les gros poids des modèles ni les journaux MLflow historiques.
