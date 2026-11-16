# Roadmap

Prime pure incendie d'un contrat multirisque agricole.

Une ligne = un contrat sur un exercice (`AN_EXERC`, 9 années anonymisées).
Cible finale : `CHARGE` = `FREQ` × `CM` × `ANNEE_ASSURANCE`.
Métrique : RMSE sur `CHARGE`. Plancher observé sur le train : prédire 0
partout, RMSE ≈ 6 803.

Apprentissage : 383 610 lignes × 374 colonnes. Test : 95 852 lignes, mêmes
colonnes, sans cibles. Données locales dans `data/raw/`, non versionnées.

- [ ] Phase 1 — Décrire le problème dans le README (contexte, usage, métrique, données locales)
- [ ] Phase 2 — Renommer le package `agri_fire`, déclarer les dépendances validées, documenter le venv
- [ ] Phase 3 — Rapport testé : grain, manquants, formule de la charge
- [ ] Phase 4 — EDA (zéros, exposition, années, zones, blocs redondants) et première version des 4 présentations
- [ ] Phase 5 — Baseline Poisson (fréquence) + Tweedie (coût moyen), RMSE sur un jeu tenu à part
- [ ] Phase 6 — Modèles linéaire et arbres, réglage, choix documenté, présentations du résultat
- [ ] Phase 7 — `train.py` entraîne le modèle retenu et le sauvegarde
- [ ] Phase 8 — `predict.py`, Dockerfile, mode d'emploi, présentations d'usage
- [ ] Phase 9 — Bonus cloud, après choix de l'hébergeur
