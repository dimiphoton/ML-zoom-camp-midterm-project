# Changelog

## [Non publié]

- Exploration : 99,25 % de contrats sans sinistre, exposition, neuf millésimes, 93 zones, colonnes redondantes (surfaces, capitaux, somme météo). Notebook `notebooks/eda.ipynb`, note `docs/eda.md`, première version des quatre présentations.
- Contrôle de grain testé (`check_train`) : 383 610 lignes, `ID` unique, formule de `CHARGE`, nombre de sinistres entier, mêmes colonnes train et test. Note dans `docs/qualite-donnees.md`.
- Package `agri_fire`. Dépendances : pandas, numpy, scikit-learn, xgboost, flask, matplotlib, pytest. Venv documenté dans le README.
- README : problème, usage Pacifica, RMSE sur `CHARGE`, plancher ≈ 6 803, emplacement des quatre CSV.
- Cadrage : prime pure incendie agricole, RMSE sur la charge, roadmap en 9 phases.
- Initialisation du projet à partir du template portfolio.
