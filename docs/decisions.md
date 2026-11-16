# Décisions

| Date | Décision | Alternative envisagée | Raison |
|---|---|---|---|
| 2026-11-16 | RMSE sur `CHARGE`, avec `CHARGE` = `FREQ` × `CM` × `ANNEE_ASSURANCE` | Un seul modèle direct sur la charge | Le brief décompose fréquence et coût moyen. La formule est exacte sur les 383 610 lignes du train (écart < 10⁻¹⁰). |
| 2026-11-16 | Le coût moyen s'apprend sur les 2 894 lignes avec au moins un sinistre | L'apprendre sur toutes les lignes, comme le notebook d'exemple | Sans sinistre, `CM` vaut 0 par définition, pas parce qu'un coût a été observé. |
| 2026-11-16 | Les 6 coûts négatifs sortent de l'apprentissage du coût et restent dans le RMSE | Les tronquer à 0 | Une Tweedie attend un montant ≥ 0. Ces 6 lignes ne doivent pas régler le modèle, mais le score final les compte. |
| 2026-11-16 | Split interne aléatoire par ligne, graine 42. Le test officiel ne sert pas à choisir le modèle | Split selon `AN_EXERC` | `ANNEE1`–`ANNEE9` ne sont pas des dates, et il n'y a pas d'identifiant client. |
| 2026-11-16 | Stack : pandas, scikit-learn, XGBoost, Flask | LightGBM, statsmodels, category_encoders, BentoML | Validé au cadrage. Poisson et Tweedie sont dans scikit-learn. Flask couvre le service demandé par Zoomcamp. |
| 2026-11-16 | L'EDA décrit les doublons et n'en retire aucun | Retirer tout de suite la somme météo et les surfaces copiées | Le choix des variables appartient à l'ajustement des modèles. |
