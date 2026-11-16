# Objectif du projet

- **But** : prédire la prime pure incendie d'un contrat multirisque agricole.
  La charge (`CHARGE`) est le produit de la fréquence des sinistres
  (`FREQ`), du coût moyen (`CM`) et des années assurées
  (`ANNEE_ASSURANCE`). On choisit le modèle au RMSE sur cette charge,
  en partant d'une baseline Poisson (fréquence) + Tweedie (coût moyen).
- **Origine** : brief de challenge (Pacifica / multirisque agricole) et
  midterm ML Zoomcamp. Le dictionnaire des variables et un notebook
  d'exemple sont dans `experiments/`. Les CSV sont dans `data/raw/`,
  non versionnés.
- **Contraintes de départ** :
  - Barème Zoomcamp : plusieurs modèles, réglage, `train.py`, service
    web, dépendances, Docker. Le cloud est un bonus, l'hébergeur se
    choisit à la phase 9.
  - Stack validée : pandas, scikit-learn, XGBoost, Flask. Pas
    d'autre outil sans un nouveau oui.
  - Le coût moyen s'apprend seulement sur les lignes avec au moins un
    sinistre. Les 6 coûts négatifs sortent de cet apprentissage et
    restent dans le RMSE. Split interne aléatoire par ligne, graine 42.
  - Ne pas committer `data/raw/`.

Ce fichier capture le but brut, tel qu'il a été formulé au départ. Les
versions polies destinées au portfolio vivent dans `README.md` et `docs/`.

Métier, domaine et stack se remplissent dans `brief/identite.md` (source
du bandeau README / covers / topics GitHub), pas ici.
