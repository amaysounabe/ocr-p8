<h1 style="text-align: center; font-size: 35px;">Veille technologique - Modèle E5</h1>

## Description
Taille du fichier **CSV** : **428 Ko** <br>

*Les données utilisées au cours de ce projet ont déjà été traitées dans un projet antérieur. Les étapes de nettoyage ont donc déjà été réalisées au cours du projet "Classification automatique de biens" (lien [ici](https://github.com/amaysounabe/ocr-p6))*

*Ce projet contenait initialement une partie consistant à enrichir un dashboard Streamlit déjà créé dans un projet antérieur. Le dashboard est donc dans sa version définitive dans le projet "Implémentation d'un modèle de Scoring" (lien [ici](https://github.com/amaysounabe/ocr-p7)).
Cependant, les slides de présentation n'ont pas été modifiées et incluent toujours des éléments relatifs à cette ancienne partie du projet.*

## Procédure
- Modélisation avec un modèle standard : Universal Sentence Encoder
- Modélisation avec les modèles RoBERTa, T5, E5 (base & Large)
- Entrainement d'un RandomForestClassifier


## Dossiers & fichiers
- **data** &rarr; Dossier contenant le jeu de données.
- **notebook_modelisation.ipynb** &rarr; Notebook comportant les modélisations effectuées.
- **report.pdf** &rarr; Rapport détaillé sur les performances du modèle E5 dans le cadre de ce projet.
- **slides_presentation.pdf** &rarr; Fichier pdf des slides de présentation de l'analyse exploratoire utilisées à lors de la soutenance du projet.
- **requirements.txt** &rarr; Fichier texte contenant la liste des bibliothèques utilisées

## Installation
Pour une utilisation en local :

```bash
git clone https://github.com/amaysounabe/ocr-p8.git
cd ocr-p8
pip install -r requirements.txt
```