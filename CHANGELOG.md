# Changelog

## [Non publié]

- Sélection au RMSE sur le même hold-out : ridge retenue (pénalité 10 000 sur la fréquence, 100 sur le coût), RMSE 7 039,79. XGBoost 7 041,55, plancher « prédire 0 » 7 045,28, baseline Poisson–Tweedie 7 047,10. Le réglage se fait sur une découpe interne. Le test officiel n'est pas utilisé.
- Baseline Poisson (fréquence) et Tweedie (coût moyen), hold-out 20 % du train, graine 42. RMSE 7 047,10 contre 7 045,28 en prédisant 0. Le test officiel n'est pas utilisé.
- Exploration : 99,25 % de contrats sans sinistre, exposition, neuf millésimes, 93 zones, colonnes redondantes (surfaces, capitaux, somme météo). Notebook `notebooks/eda.ipynb`, note `docs/eda.md`, première version des quatre présentations.
- Contrôle de grain testé (`check_train`) : 383 610 lignes, `ID` unique, formule de `CHARGE`, nombre de sinistres entier, mêmes colonnes train et test. Note dans `docs/qualite-donnees.md`.
- Package `agri_fire`. Dépendances : pandas, numpy, scikit-learn, xgboost, flask, matplotlib, pytest. Venv documenté dans le README.
- README : problème, usage Pacifica, RMSE sur `CHARGE`, plancher ≈ 6 803, emplacement des quatre CSV.
- Cadrage : prime pure incendie agricole, RMSE sur la charge, roadmap en 9 phases.
- Initialisation du projet à partir du template portfolio.
