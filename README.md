# Modélisation de la Consommation d'Énergie des Bâtiments de Seattle

Projet du parcours Master 2 Data Science d'OpenClassrooms - Central/Supélec.

## Projet Principal

Dans ce projet, on réalise une modélisation de la consommation d'énergie annuelle des batîments de Seattle à partir de [données](https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking#2015-building-energy-benchmarking.csv) récoltées par la ville de Seattle. L'objectif est de tester plusieurs algorithmes d'apprentissage machine, et de choisir la solution la plus adaptée à notre problématique. Ce choix s'appuie sur trois **critères** : 
- les performances moyennes du modèle
- le pouvoir de généralisation du modèle
- la complexité/temps de calcul de la modélisation

L'analyse principale est divisée en trois **notebooks** :
- une [sélection/nettoyage des données](https://nbviewer.jupyter.org/github/EloiLQ/EnergyBuildingModelisation/blob/main/SampleSelection.ipynb) (SampleSelection.ipynb)
- une [exploration/visualisation des données](https://nbviewer.jupyter.org/github/EloiLQ/EnergyBuildingModelisation/blob/main/Exploration.ipynb) (Exploration.ipynb)
- une [étude de modélisation de la consommation d'énergie](https://nbviewer.jupyter.org/github/EloiLQ/EnergyBuildingModelisation/blob/main/MachineLearning.ipynb) (MachineLearning.ipynb)

Les **modèles testés** dans le fichier principal MachineLearning.ipynb sont : 
- les modèles linéaires de Scikit-Learn : régression linéaire, Lasso, Ridge
- les modèles basés sur des arbres de décision d'XGBoost : arbre de décision, forêt aléatoire, arbre de décision boosté, forêt aléatoire boostée

On a également étudié plusieurs **encodages de variables catégorielles** :
- l'encodage One Hot
- l'encodage Target Mean
- les encodages Target de Moments supérieurs (2, 3, 4)

Enfin, on a réalisé une **recherche sur grille** des hyperparamètres optimaux du modèle le plus performant obtenu, la forêt aléatoire boostée d'XGBoost.

## Projet Parallèle

Ce projet comporte également une étude sur l'impact des transformations de données réalisées avant l'entraînement d'algorithmes d'apprentissage machine. Trois **transformantions de données** ont été considérées ici : 
- la suppression des outliers
- la standardisation des variables d'entrée
- le passage au logarithme de la variable cible

Cette analyse secondaire est divisée en deux **notebooks** : 
- une [sélection des données](https://nbviewer.jupyter.org/github/EloiLQ/EnergyBuildingModelisation/blob/main/LinearModelData.ipynb) (LinearModelData.ipynb)
- une [étude des transformations de données](https://nbviewer.jupyter.org/github/EloiLQ/EnergyBuildingModelisation/blob/main/LinearStudy.ipynb) (LinearStudy.ipynb)

Le modèle utilisé est une simple régression linéaire.
