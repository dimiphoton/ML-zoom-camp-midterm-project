# Journal de développement

## AAAA-MM-JJ — Initialisation du projet

- Repo créé à partir du template portfolio.

## 2026-11-16 — Cadrage

- Sujet : prime pure incendie agricole. Métier machine learning, domaine
  agriculture, stack Python / scikit-learn / XGBoost / Flask.
- Train 383 610 × 374, test 95 852. Charge = fréquence × coût moyen ×
  années assurées, formule vérifiée. 99,25 % des contrats sans sinistre.
  RMSE du modèle « prédire 0 » ≈ 6 803 sur le train.
- Roadmap en 9 phases. Les CSV restent dans `data/raw/`, hors git.
  L'enchaînement automatique est en pause : une phase par prompt.

## 2026-11-16 — Phase 1 : problème dans le README

- README en anglais : prime pure incendie, usage chez Pacifica (tarification),
  RMSE sur `CHARGE`, plancher « prédire 0 » ≈ 6 803 sur le train, quatre CSV
  à déposer dans `data/raw/` (non versionnés).
- Bandeau Role / Domain / Stack inchangé.

## 2026-11-16 — Phase 2 : environnement

- Package `src/mon_projet` renommé `src/agri_fire`. Les tests importent
  `agri_fire` et `agri_fire.cli`.
- `pyproject.toml` déclare uniquement pandas, numpy, scikit-learn,
  xgboost, flask, matplotlib et pytest. Installés dans `.venv`.
- README : création du venv, activation (PowerShell et bash), `pytest`.

## 2026-11-16 — Phase 3 : contrôle des données

- `check_train` charge le train et vérifie 383 610 lignes, l'unicité de
  `ID`, `CHARGE` = `FREQ` × `CM` × `ANNEE_ASSURANCE` (tolérance 1e-6),
  un nombre de sinistres entier, et les mêmes colonnes que le test.
- Les tests unitaires tournent sur un petit CSV fabriqué. Les 697 Mo
  ne sont pas lus par pytest.
- `docs/qualite-donnees.md` : 278 colonnes avec manquants, types
  (58 entiers, 36 flottants, 280 textes), quatre colonnes sans libellé
  (`ADOSS`, `DUREE_REQANEUF`, `ID`, `NBJFXI3S10_MM_A`), `ESPINSEE`
  manquant à 39,5 % sur le train.

## 2026-11-16 — Phase 4 : exploration

- `notebooks/eda.ipynb` appelle `agri_fire.eda`. 99,25 % des contrats
  sans sinistre. Exposition médiane 0,88 an, 54,7 % en dessous d'un an.
  Neuf millésimes, taux de sinistre entre 0,49 % et 1,08 %. 93 zones,
  trois en ont moins de 100.
- Colonnes redondantes : `SURFACE1` ≈ `SURFACE2` (5 écarts),
  `SURFACE4` = `SURFACE6`, `KAPITAL10` recopie `KAPITAL11`, la somme
  météo a le même rang que la moyenne sur 30 grandeurs. Aucune colonne
  n'est retirée à ce stade.
- Première version des 4 présentations : cadrage du problème, questions
  de couverture encore ouvertes. Graphes dans `pictures/presentations/`.
