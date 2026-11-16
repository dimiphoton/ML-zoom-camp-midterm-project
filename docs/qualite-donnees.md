# Qualité des données

Contrôle du train local (`train_input_Z61KlZo.csv`, 697 Mo, et
`train_output_DzPxaPY.csv`) et du test (`test_input_5qJzHrr.csv`).
Les CSV restent dans `data/raw/`, hors git. La fonction qui refait
les contrôles de grain est `check_train`, dans `agri_fire.data`.

## Grain

Une ligne est un contrat sur un exercice (`AN_EXERC`).

| | Train | Test |
|---|---:|---:|
| Lignes | 383 610 | 95 852 |
| Colonnes | 374 | 374 |

`ID` est unique dans le train, dans les cibles et dans le test. Les
identifiants du train et des cibles sont les mêmes. Les 374 colonnes
du train et du test sont les mêmes, dans le même ordre. Les cibles
(`FREQ`, `CM`, `ANNEE_ASSURANCE`, `CHARGE`) ne sont pas dans le test.

## Formule de la charge

Sur les 383 610 lignes de cibles, aucune valeur manquante.

`CHARGE` = `FREQ` × `CM` × `ANNEE_ASSURANCE`. L'écart absolu maximum
est 5,82 × 10⁻¹¹, sous la tolérance du contrôle (10⁻⁶).

`FREQ` × `ANNEE_ASSURANCE` est le nombre de sinistres. Il tombe sur un
entier : l'écart maximum à l'entier le plus proche est 2,2 × 10⁻¹⁶.

380 716 contrats n'ont aucun sinistre (`FREQ` = 0). Six coûts moyens
sont négatifs. La formule reste exacte sur ces six lignes.

## Valeurs manquantes

Lecture du train en un seul bloc (`low_memory=False`). 96 colonnes
sont complètes. 278 en ont au moins une valeur manquante.

| Part de manquants | Colonnes |
|---|---:|
| 0 | 96 |
| plus de 0 à 1 % | 13 |
| plus de 1 % à 10 % | 71 |
| plus de 10 % à 50 % | 11 |
| plus de 50 % à 90 % | 176 |
| plus de 90 % | 7 |

Les sept colonnes au-dessus de 90 % :

| Colonne | Manquants | Libellé |
|---|---:|---|
| `DEROG14` | 99,998 % (8 valeurs `D18`) | Dérogation tarifaire |
| `DEROG13` | 99,66 % | Dérogation tarifaire |
| `DEROG16` | 99,15 % | Dérogation tarifaire |
| `CARACT2` | 96,00 % | Caractéristique du bâtiment |
| `CARACT3` | 96,00 % | Caractéristique du bâtiment |
| `TYPBAT1` | 92,11 % | Type bâtiment |
| `DEROG12` | 91,00 % | Dérogation tarifaire |

Le bloc suivant (hauteurs, distances, pluviométrie) se situe surtout
entre 50 % et 62 %.

### ESPINSEE

`ESPINSEE` (libellé : espace INSEE) est un code, `ESP1` à `ESP4`, pas
un nombre. Il manque sur **151 542 lignes du train, soit 39,5 %**, et
sur 38 026 lignes du test, soit 39,7 %.

## Types

En lecture complète : 58 colonnes `int64`, 36 `float64`, 280 `object`.

Si pandas découpe le fichier, 13 colonnes mélangent des codes et des
nombres et changent de type selon le bloc : `FRCH1`, `FRCH2`,
`DEROG12`, `DEROG13`, `DEROG14`, `RISK6`, `RISK8`, `RISK9`, `RISK12`,
`RISK13`, `EQUIPEMENT2`, `EQUIPEMENT5`, `ESPINSEE`. En lecture
complète elles restent en texte. `ESPINSEE` ne contient que les quatre
codes et les manquants.

## Quatre colonnes sans libellé

Le dictionnaire `experiments/Descriptif_données.xlsx` a 373 libellés.
Beaucoup de noms n'y diffèrent du CSV que par la casse (`altitude_1`
dans le dictionnaire, `ALTITUDE_1` dans le fichier) : ces colonnes ont
un libellé.

En ignorant la casse, quatre colonnes du train n'ont pas de libellé :

- `ADOSS`
- `DUREE_REQANEUF`
- `ID`
- `NBJFXI3S10_MM_A`

`FREQ`, `CM` et `CHARGE` sont dans le dictionnaire. Ils sont dans le
fichier de cibles, pas dans les 374 colonnes d'entrée.

## Rejouer le contrôle

```python
from agri_fire.data import check_train

check_train(
    "data/raw/train_input_Z61KlZo.csv",
    "data/raw/train_output_DzPxaPY.csv",
    "data/raw/test_input_5qJzHrr.csv",
)
```

Le test unitaire (`tests/test_checks.py`) appelle la même fonction sur
un petit CSV fabriqué. Il ne lit pas les 697 Mo.
