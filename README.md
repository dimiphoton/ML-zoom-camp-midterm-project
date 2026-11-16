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

The selected model is a ridge regression. It uses the same 20% row hold-out as the baseline (seed 42, 76,722 contracts) and the same inputs: activity, vocation, zone, exposure, and three aggregates. The official test file is not used.

Frequency is a ridge fit on every learning row. Average cost is a ridge fit only on rows with at least one claim and a non-negative cost. Predicted `CHARGE` = predicted frequency × predicted average cost × `ANNEE_ASSURANCE`. A negative component is clipped to 0 before that product.

Penalties and the tree settings are chosen on an inner 20% split of the learning rows, same seed. The hold-out is used once, to compare the finished models.

| Candidate | Hold-out RMSE |
|---|---|
| Ridge (frequency penalty 10,000, cost penalty 100) | **7,039.79** |
| XGBoost (depth 4, learning rate 0.05, 250 trees, leaf weight 40) | 7,041.55 |
| Predict 0 | 7,045.28 |
| Poisson–Tweedie baseline | 7,047.10 |

Ridge is the one we keep. It beats the floor and the baseline. The trees also beat them, by less. On the inner split the trees scored better (6,764.82 against 6,769.35). That split is not where the choice is made. Depth 6 was the worst of the eight tree settings. XGBoost fits frequency with squared error, because `FREQ` is a rate, and fits cost with a Tweedie objective (power 1.5), the same variance family as the baseline.

`train.py` refits this ridge on every training row, with those two penalties left as they are. The hold-out is not carved out again.

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

`pip install -e .` reads `pyproject.toml` and installs pandas, numpy, scikit-learn, xgboost, flask, matplotlib, and pytest. The four CSV files in `data/raw/` are the data step for training. The service below reads the model file that training writes.

The comparison is recomputed from the two training files only. It takes about two minutes.

```python
from agri_fire.models.selection import evaluate_selection_files

evaluate_selection_files(
    "data/raw/train_input_Z61KlZo.csv",
    "data/raw/train_output_DzPxaPY.csv",
)
```

`python scripts/select_models.py` runs the same comparison and rewrites the two figures in `pictures/presentations/`.

## Train the retained model

`train.py` fits the ridge above on every training row and writes `models/ridge.pkl`. Frequency penalty 10,000, cost penalty 100. It calls the same `fit_ridge` path as the comparison: same columns, same rule that keeps negative costs out of the cost fit, same clip of a negative component to 0, then frequency × average cost × `ANNEE_ASSURANCE`. It does not retune. It does not open the official test file.

From the repository root, with the virtual environment active:

```powershell
python train.py
```

The same command works in bash. It reads `data/raw/train_input_Z61KlZo.csv` and `data/raw/train_output_DzPxaPY.csv`.

## Serve the premium locally

`predict.py` loads `models/ridge.pkl` and serves it with Flask on port 9696. The container listens on the machine that built it. `PORT` changes that port. Leave it unset locally.

A request sends the seven columns below. Surfaces and capitals are already summed. The response is the predicted frequency `FREQ`, the predicted average cost `CM`, and the pure premium `CHARGE` (frequency × average cost × `ANNEE_ASSURANCE`). A predicted frequency or average cost below zero is brought up to 0 before that product.

| Column | What to send |
|---|---|
| `ACTIVIT2` | Activity |
| `VOCATION` | Farm vocation |
| `ZONE` | Zone |
| `ANNEE_ASSURANCE` | Years insured on the row. Zero is allowed. A negative value is rejected |
| `surface_sum` | Sum of the surface columns, without the copies `SURFACE2` and `SURFACE6` |
| `capital_amount` | Sum of the amounts `KAPITAL11` to `KAPITAL33` |
| `capital_flags` | Count of the capital indicators |

Train first, so `models/ridge.pkl` exists. Then build the image and start it, from the repository root. The same two Docker commands work in PowerShell and in bash.

```bash
docker build -t agri-fire .
docker run --rm -p 9696:9696 agri-fire
```

Ask it from another terminal.

Windows (PowerShell):

```powershell
$body = @{
  ACTIVIT2 = "ACT1"
  VOCATION = "VOC1"
  ZONE = "1"
  ANNEE_ASSURANCE = 1
  surface_sum = 1200
  capital_amount = 50000
  capital_flags = 3
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:9696/predict `
  -Method Post `
  -Body $body `
  -ContentType "application/json; charset=utf-8"
```

macOS and Linux:

```bash
curl -s -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"ACTIVIT2":"ACT1","VOCATION":"VOC1","ZONE":"1","ANNEE_ASSURANCE":1,"surface_sum":1200,"capital_amount":50000,"capital_flags":3}'
```

The answer is a JSON object with `FREQ`, `CM`, and `CHARGE`. A JSON array of contracts returns an array, in the same order. A missing column returns HTTP 400. The numbers above are an example, not a row from the files.

`python predict.py` starts the same service without Docker, from the virtual environment, after `python train.py`. `MODEL_PATH` overrides the pickle path.

## Deploy on Render

The same image can answer on the public internet. [Render](https://render.com) runs it as a free web service and gives it an `onrender.com` URL. No credit card. The free instance sleeps after 15 minutes without a request and takes about a minute to wake up.

This repository does not publish a live URL. Creating the Docker Hub login and the Render account is yours. The steps below are the whole deployment.

`models/ridge.pkl` is not in git. Build the image on your machine, after `python train.py`, so the pickle is inside it. Render pulls that image. It does not see `data/raw/` and it does not train.

Render sets `PORT` (10000). `predict.py` listens on it. Locally, without `PORT`, the port stays 9696.

From the repository root, with Docker running. The commands are the same in PowerShell and in bash. Replace `YOUR_DOCKERHUB_USER` with your Docker Hub login.

```bash
python train.py
docker build -t agri-fire .
docker tag agri-fire YOUR_DOCKERHUB_USER/agri-fire:latest
docker push YOUR_DOCKERHUB_USER/agri-fire:latest
```

`docker push` asks you to `docker login` first.

In the [Render dashboard](https://dashboard.render.com):

1. **New → Web Service**.
2. **Existing Image**. Image URL: `docker.io/YOUR_DOCKERHUB_USER/agri-fire:latest`.
3. Instance type: **Free**. Leave the start command empty, so the image runs `python predict.py`.
4. Create the service. The public URL looks like `https://agri-fire.onrender.com`.

`render.yaml` is that service written down (free instance, Frankfurt, the same image). Replace `YOUR_DOCKERHUB_USER` in the file, then in the dashboard choose **New → Blueprint** and point it at the repository. The dashboard clicks above are enough if you skip the blueprint.

Ask the public URL from another terminal. The first call after a sleep can take about a minute. The body is the same example as the local service.

Windows (PowerShell):

```powershell
$body = @{
  ACTIVIT2 = "ACT1"
  VOCATION = "VOC1"
  ZONE = "1"
  ANNEE_ASSURANCE = 1
  surface_sum = 1200
  capital_amount = 50000
  capital_flags = 3
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri https://YOUR_SERVICE.onrender.com/predict `
  -Method Post `
  -Body $body `
  -ContentType "application/json; charset=utf-8"
```

macOS and Linux:

```bash
curl -s -X POST https://YOUR_SERVICE.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"ACTIVIT2":"ACT1","VOCATION":"VOC1","ZONE":"1","ANNEE_ASSURANCE":1,"surface_sum":1200,"capital_amount":50000,"capital_flags":3}'
```

The answer is again `FREQ`, `CM`, and `CHARGE`.

## Repo structure

- `train.py` — fits the retained ridge and writes `models/ridge.pkl`
- `predict.py` — loads that file and serves `POST /predict`
- `Dockerfile` — image of the service, port 9696 locally, `PORT` on Render
- `render.yaml` — Render blueprint for that image (fill in the Docker Hub user)
- `models/` — saved model, produced when you run `train.py`
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
