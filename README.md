# Fire pure premium on French farm contracts

| | |
|---|---|
| **Role** | Machine learning |
| **Domain** | Agriculture |
| **Stack** | ![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?logo=xgboost&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) |
| **Level** | Intermediate |
| **Status** | Draft |

## The problem

A French farmer buys a multi-risk policy, the *multirisque agricole*, so a fire does not stop the farm. The cover includes the business, the farm buildings, the equipment stored there, and the financial protection around them. Pacifica manages the contract. Fire is a large share of its claims cost.

The number to predict is the fire pure premium: the claim cost expected for one contract over one exercise year, before commercial loadings. In the files, the observed cost on that row is `CHARGE`.

`CHARGE` = `FREQ` × `CM` × `ANNEE_ASSURANCE`

| Column | Meaning |
|---|---|
| `FREQ` | Fire claim frequency: number of claims / years insured |
| `CM` | Average cost of a fire claim: claim cost / number of claims |
| `ANNEE_ASSURANCE` | Number of years the contract has been insured |
| `CHARGE` | Fire claim cost on the row, equal to the product of the three columns |

One row is one contract over one exercise year. The year, `AN_EXERC`, is anonymised into nine labels (`ANNEE1`–`ANNEE9`). The training table has 383,610 rows and 374 columns. The test table has 95,852 rows, the same feature columns, and no targets.

## Who uses it

Pacifica's pricing team uses a prediction of `CHARGE` as the fire pure premium of the farm multi-risk contract. The decision is which expected fire claim cost to attach to a contract-year. This repository is the ML Zoomcamp midterm written on that brief. The brief's reference is a Poisson GLM for frequency and a Tweedie GLM for average cost, both with a log link, scored once the two pieces are combined into `CHARGE`.

## Metric

Models are compared by the RMSE of `CHARGE`.

On the training set, predicting `0` for every contract reaches an RMSE of about **6,803**. That floor is the score of "predict zero everywhere". It sits there because 99.25% of training contracts have no fire claim (`FREQ` = 0). The official test extract has no `CHARGE` column, so this floor is measured on the training targets.

## Local data

The four challenge CSV files stay on the machine that runs the project. `.gitignore` excludes `data/raw/*`. The repository only keeps `data/raw/.gitkeep`.

Put the four files in `data/raw/`:

| File | Contents |
|---|---|
| `train_input_Z61KlZo.csv` | Training features, 383,610 rows × 374 columns |
| `train_output_DzPxaPY.csv` | Targets for those rows: `ID`, `FREQ`, `CM`, `ANNEE_ASSURANCE`, `CHARGE` |
| `test_input_5qJzHrr.csv` | Test features, 95,852 rows × 374 columns, no targets |
| `submission_csv_file_random_example_3fbDtrr.csv` | Example submission, one row per test id, same five columns as the training targets |

Column labels are in `experiments/Descriptif_données.xlsx`. The challenge notebook `experiments/Exemple.ipynb` reads shorter names from the working directory (`train_input.csv`, `train_output.csv`, `test_input.csv`), which are not the filenames above.

## Result

No model is selected yet. The training-set floor is an RMSE of about 6,803, from predicting zero charge on every contract.

## What the training rows show

99.25% of training contracts have no fire claim. Insured time is under one year on 54.7% of rows and exactly one year on 44.6%. The nine anonymised years stay in a narrow band of claim rates, while row counts run from about 27,000 to 78,000. There are 93 zones. Some surface, capital and weather columns repeat the same fact. The walkthrough is `notebooks/eda.ipynb`; the figures are in `docs/eda.md`.

## Reproduce

Python 3.11 or newer. From the repository root, create a virtual environment, activate it, install the declared dependencies, and run the tests.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
pytest
```

macOS and Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
pytest
```

`pip install -e .` reads `pyproject.toml` and installs pandas, numpy, scikit-learn, xgboost, flask, matplotlib, and pytest. Training and the prediction service follow the later steps in `ROADMAP.md`. The data step for those steps is the table above: the four CSV files in `data/raw/`.

## Repo structure

- `data/raw/` — the four local CSV files, not versioned
- `experiments/` — variable dictionary and example notebook
- `src/agri_fire/` — Python package
- `docs/decisions.md` — metric, split, and stack already chosen
- `docs/qualite-donnees.md` — row count, missing values, charge formula
- `docs/eda.md` — zeros, exposure, years, zones, duplicated columns
- `notebooks/eda.ipynb` — calls `agri_fire.eda` on the local train
- `docs/presentations/` — slide sources
- `ROADMAP.md`, `JOURNAL.md` — plan and development log, in French

## Presentations

Two audiences × two languages (Marp theme `portfolio`, HTML on GitHub Pages).
The recruiter deck is a ~6-minute pitch; the technical deck is a ~12-minute
deep dive. They may diverge a lot — the bar is attractive and informative
for each audience, not a mirrored pair of slides.

- [Recruiter overview (EN)](docs/slides/presentation-recruteur-en.html)
- [Technical deep dive (EN)](docs/slides/presentation-technique-en.html)
- [Présentation grand public (FR)](docs/slides/presentation-recruteur-fr.html)
- [Présentation technique (FR)](docs/slides/presentation-technique-fr.html)
