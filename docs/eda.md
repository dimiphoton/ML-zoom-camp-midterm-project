# Exploration

Lecture du train local (383 610 contrats). Les fonctions sont dans
`agri_fire.eda`, les graphes dans `agri_fire.plots`, l'appel dans
`notebooks/eda.ipynb`. Les CSV restent dans `data/raw/`.

## Sans sinistre

380 716 contrats ont `FREQ` = 0, soit **99,25 %**. 2 894 ont au moins
un sinistre. Parmi eux, 536 ont un coût moyen nul et 6 un coût négatif.
381 252 charges valent 0.

Prédire 0 partout donne un RMSE d'environ 6 803 sur ce train. Le
plancher vient de la masse de zéros.

## Exposition

`ANNEE_ASSURANCE` est la même dans le fichier de variables et dans les
cibles (écart maximum 0).

| | |
|---|---:|
| Médiane | 0,88 an |
| Moins d'un an | 54,7 % |
| Exactement un an | 44,6 % |
| Plus d'un an, jusqu'à 2 | 0,7 % |
| Exposition totale | 269 124 années |
| Sinistres (`FREQ` × exposition) | 3 004 |
| Sinistres pour 100 années | 1,1 |

## Millésimes

`AN_EXERC` prend les libellés `ANNEE1` à `ANNEE9`. Ce n'est pas une
date. La part sans sinistre reste entre 98,9 % (`ANNEE5`) et 99,5 %
(`ANNEE8`). Les effectifs vont de 26 573 à 77 819 (`ANNEE6`).

Le taux de contrats sinistrés va de 0,49 % à 1,08 % : du simple au
double, autour d'un contrat sur cent.

## Zones

93 zones. Effectif de 4 à 13 006 contrats, médiane 4 035. Trois zones
ont moins de 100 contrats.

Parmi les 86 zones d'au moins 1 000 contrats, le taux va d'environ
0,3 % à 1,9 % (un facteur six). Les zones plus petites bougent
davantage, jusqu'à 2,6 % sur 39 contrats.

## Blocs redondants

**Surfaces.** `SURFACE1` et `SURFACE2` diffèrent sur 5 contrats.
`SURFACE4` et `SURFACE6` sont identiques sur les 375 382 lignes où les
deux sont remplies. `SURFACE14`, `SURFACE15` et `SURFACE20` ne sortent
de zéro que sur 30, 31 et 15 contrats.

**Capitaux.** `KAPITAL10` et `KAPITAL11` sont égaux sur les 264 605
lignes où `KAPITAL11` est rempli. Là où `KAPITAL11` manque (31 %),
`KAPITAL10` vaut 0. `KAPITAL1` vaut 1 exactement quand `KAPITAL13`
vaut 5 000 ou 10 000 (9 246 contrats).

**Météo.** Trente grandeurs ont une moyenne, un maximum et une somme,
codés en paliers `01` à `04`. Le rang de la somme est celui de la
moyenne sur chaque ligne comparée. Le maximum n'est pas une copie :
la corrélation médiane des rangs avec la moyenne est 0,84.

Ces constats ne retirent pas encore de colonnes. Le choix des
variables vient avec les modèles.

## Rejouer

Depuis la racine du dépôt, ouvrir `notebooks/eda.ipynb`. Les tests
(`tests/test_eda.py`) appellent les mêmes fonctions sur un petit jeu
fabriqué. Ils ne lisent pas les 697 Mo.
