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

## 2026-11-16 — Phase 6 : sélection

- Ridge et XGBoost sur le même hold-out que la baseline (graine 42, 76 722
  lignes). Le coût moyen ne s'apprend que sur les sinistres à coût positif
  ou nul. Le réglage utilise une découpe interne, pas le hold-out.
- RMSE : ridge 7 039,79 (pénalité 10 000 et 100), XGBoost 7 041,55
  (profondeur 4, 250 arbres), prédire 0 fait 7 045,28, Poisson–Tweedie
  7 047,10. On retient la droite. Les arbres gagnaient la découpe interne.
- Présentations : l'écart avec « aucun sinistre » pour un lecteur non
  spécialiste, et la charge observée par tranche de prédiction.

## 2026-11-16 — Phase 7 : script d'entraînement

- `train.py` ajuste la ridge retenue (pénalité 10 000 sur la fréquence,
  100 sur le coût) sur toutes les lignes du train et écrit
  `models/ridge.pkl`. Il appelle `fit_ridge` : pas une seconde formule.
- Le hold-out n'est pas refait. Le test officiel n'est pas lu. Un test
  sur un petit jeu vérifie que l'artefact se recharge et renvoie une
  `CHARGE`.
- Les decks techniques citent `python train.py`. Le récit recruteur
  ne change pas : le modèle servi est le même.

## 2026-11-16 — Phase 5 : baseline

- Poisson scikit-learn sur toutes les lignes d'un split aléatoire (graine 42,
  20 % tenus à part). Tweedie `power` 1,5 sur les sinistres à coût positif
  ou nul. Charge prédite = fréquence × coût × exposition.
- Jeu court : activité, vocation, zone, exposition, surfaces agrégées hors
  copies, capitaux en montant, indicateurs de capital. Pas les 374 colonnes.
- Hold-out de 76 722 lignes. RMSE 7 047,10, contre 7 045,28 en prédisant 0.
  La baseline ne bat pas ce plancher. Quatre des six coûts négatifs sortent
  de l'apprentissage du coût ; les deux autres restent dans le RMSE.
- Le test officiel (95 852 lignes) n'est pas lu. Les présentations ne font
  qu'enregistrer ce constat : le récit du résultat est pour la phase 6.

## 2026-11-16 — Phase 8 : service local

- `predict.py` charge `models/ridge.pkl` et expose `POST /predict`.
  Entrée : les sept colonnes du modèle. Sortie : `FREQ`, `CM`, `CHARGE`.
- Dockerfile et mode d'emploi dans le README : construire l'image,
  la lancer sur le port 9696, l'interroger en local. Pas de cloud.
- Présentations : un contrat décrit, un coût attendu en retour, lien
  vers le dépôt. Les commandes Docker sont sur les decks techniques.
