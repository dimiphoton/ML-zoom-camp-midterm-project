# Project name

| | |
|---|---|
| **Role** | Machine learning |
| **Domain** | Agriculture |
| **Stack** | ![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?logo=xgboost&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) |
| **Level** | Intermediate |
| **Status** | Draft |

The Role · Domain · Stack line is filled from `brief/identite.md` at kickoff.

## Objective

Predict the fire pure premium of a French multi-risk farm insurance
contract. The pure premium (`CHARGE`) is claim frequency times average
claim cost times years insured. The model is chosen by RMSE on that
charge. A frequency model and a severity model are trained separately,
then multiplied.

## Data

Local challenge extracts in `data/raw/` (gitignored, not in this
repository): 383,610 training rows and 95,852 test rows, 374 columns,
nine anonymised exercise years. One row is one contract over one
exercise year. Targets (`FREQ`, `CM`, `CHARGE`) are in the training
output file only.

## Result

Not estimated yet. On the training set, predicting zero charge everywhere
already sits at an RMSE of about 6,803, because 99.25% of contracts have
no fire claim.

## Reproduce

TBD: how to install dependencies and rerun the pipeline.

```bash
pip install -e .
python -m mon_projet.cli --help
```

## Repo structure

TBD, filled in as the project progresses — see also `ROADMAP.md` and
`JOURNAL.md` (kept in French, like the rest of the codebase).

## Presentations

Two audiences × two languages (Marp theme `portfolio`, HTML on GitHub Pages).
The recruiter deck is a ~6-minute pitch; the technical deck is a ~12-minute
deep dive. They may diverge a lot — the bar is attractive and informative
for each audience, not a mirrored pair of slides.

- [Recruiter overview (EN)](docs/slides/presentation-recruteur-en.html)
- [Technical deep dive (EN)](docs/slides/presentation-technique-en.html)
- [Présentation grand public (FR)](docs/slides/presentation-recruteur-fr.html)
- [Présentation technique (FR)](docs/slides/presentation-technique-fr.html)
