"""Baseline Poisson (fréquence) et Tweedie (coût moyen).

La charge prédite est ``fréquence prédite × coût moyen prédit × exposition``.
Le score est le RMSE sur une partie tenue à part du train. Le fichier de
test du challenge n'entre pas dans ce calcul.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agri_fire.features.baseline import (
    AGGREGATE_COLUMNS,
    CATEGORICAL_COLUMNS,
    EXPOSURE_COLUMN,
    MODEL_COLUMNS,
    BaselineError,
    assemble_baseline_frame,
    read_baseline_inputs,
)

SEED = 42
HOLD_OUT_FRACTION = 0.2
# Power 1,5 : Poisson composé, qui accepte un coût moyen nul.
# Une Gamma (power 2) refuserait les 536 coûts nuls avec sinistre.
TWEEDIE_POWER = 1.5
GLM_ALPHA = 0.0
# Sur le train complet, la Tweedie converge vers 2 100 itérations.
# 500 s'arrête trop tôt.
GLM_MAX_ITER = 3_000


@dataclass
class BaselineGlm:
    """Deux GLM déjà ajustés et l'encodage appris sur le même train.

    Attributes
    ----------
    pipeline_frequency : Pipeline
        Poisson sur ``FREQ``, toutes les lignes d'apprentissage.
    pipeline_cost : Pipeline
        Tweedie sur ``CM``. Même type d'encodage, réappris sur les
        seules lignes avec sinistre et coût positif ou nul.
    n_frequency_rows : int
        Lignes utilisées par le Poisson.
    n_cost_rows : int
        Lignes utilisées par la Tweedie.
    n_negative_cost_excluded : int
        Lignes avec sinistre et coût négatif, retirées de la Tweedie.
    """

    pipeline_frequency: Pipeline
    pipeline_cost: Pipeline
    n_frequency_rows: int
    n_cost_rows: int
    n_negative_cost_excluded: int

    def predict_frequency(self, frame: pd.DataFrame) -> np.ndarray:
        """Fréquence prédite, une valeur par ligne.

        Parameters
        ----------
        frame : DataFrame
            Tableau au format de ``assemble_baseline_frame``.

        Returns
        -------
        ndarray
            ``FREQ`` prédite, positive.
        """
        return self.pipeline_frequency.predict(_model_matrix(frame))

    def predict_cost(self, frame: pd.DataFrame) -> np.ndarray:
        """Coût moyen prédit, une valeur par ligne.

        Parameters
        ----------
        frame : DataFrame
            Tableau au format de ``assemble_baseline_frame``.

        Returns
        -------
        ndarray
            ``CM`` prédit, positif.
        """
        return self.pipeline_cost.predict(_model_matrix(frame))

    def predict_charge(self, frame: pd.DataFrame) -> np.ndarray:
        """Charge prédite : fréquence × coût moyen × années assurées.

        Parameters
        ----------
        frame : DataFrame
            Doit contenir ``ANNEE_ASSURANCE`` et les variables du modèle.

        Returns
        -------
        ndarray
            ``CHARGE`` prédite.
        """
        exposure = frame[EXPOSURE_COLUMN].to_numpy(dtype=float)
        return self.predict_frequency(frame) * self.predict_cost(frame) * exposure


@dataclass(frozen=True)
class HoldoutScore:
    """RMSE de la baseline et du plancher « prédire 0 », même hold-out.

    Attributes
    ----------
    rmse : float
        RMSE de la charge prédite sur la partie tenue à part.
    zero_rmse : float
        RMSE obtenu en prédisant 0 sur ces mêmes lignes.
    n_holdout : int
        Nombre de lignes tenues à part.
    n_frequency_rows : int
        Lignes d'apprentissage du Poisson.
    n_cost_rows : int
        Lignes d'apprentissage de la Tweedie.
    n_negative_cost_excluded : int
        Coûts négatifs retirés de la Tweedie, sur la partie apprise.
    """

    rmse: float
    zero_rmse: float
    n_holdout: int
    n_frequency_rows: int
    n_cost_rows: int
    n_negative_cost_excluded: int


def split_rows(
    frame: pd.DataFrame,
    *,
    test_size: float = HOLD_OUT_FRACTION,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpe le tableau au hasard, ligne par ligne.

    Parameters
    ----------
    frame : DataFrame
        Une ligne par contrat.
    test_size : float, optional
        Part tenue à part. Défaut : 0,2.
    seed : int, optional
        Graine du mélange. Défaut : 42.

    Returns
    -------
    tuple of DataFrame
        Partie d'apprentissage, puis partie tenue à part.
    """
    train, holdout = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    return (
        train.reset_index(drop=True),
        holdout.reset_index(drop=True),
    )


def cost_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Lignes dont le coût moyen peut entrer dans la Tweedie.

    Il faut au moins un sinistre (``FREQ`` > 0) et un coût positif ou
    nul. Un coût négatif reste dans le tableau, donc dans le RMSE s'il
    tombe dans le hold-out, mais il ne règle pas la Tweedie.

    Parameters
    ----------
    frame : DataFrame
        Doit contenir ``FREQ`` et ``CM``.

    Returns
    -------
    DataFrame
        Sous-ensemble, index d'origine conservé.
    """
    mask = frame["FREQ"].gt(0) & frame["CM"].ge(0)
    return frame.loc[mask]


def fit_baseline(train: pd.DataFrame) -> BaselineGlm:
    """Ajuste le Poisson sur toutes les lignes, la Tweedie sur les sinistres.

    Le Poisson voit toutes les lignes. La Tweedie réapprend le même
    encodage sur les seules lignes de coût. Une zone sans sinistre dans
    cette partie n'a pas de coefficient : à la prédiction elle retombe
    sur le niveau de référence, comme une zone inconnue.

    Parameters
    ----------
    train : DataFrame
        Partie d'apprentissage, au format de ``assemble_baseline_frame``.

    Returns
    -------
    BaselineGlm
        Modèles ajustés et effectifs utilisés.

    Raises
    ------
    BaselineError
        Si ``FREQ`` est négatif, ou s'il n'y a aucune ligne de coût
        positif ou nul.
    """
    _require_model_frame(train)
    frequency = train["FREQ"].to_numpy(dtype=float)
    if np.isnan(frequency).any() or (frequency < 0).any():
        raise BaselineError("FREQ manquant ou négatif : le Poisson ne s'applique pas.")

    frequency_pipeline = _glm_pipeline(
        PoissonRegressor(alpha=GLM_ALPHA, max_iter=GLM_MAX_ITER)
    )
    frequency_pipeline.fit(_model_matrix(train), frequency)

    cost = cost_training_rows(train)
    if cost.empty:
        raise BaselineError(
            "Aucune ligne avec sinistre et coût positif ou nul pour la Tweedie."
        )
    cost_pipeline = _glm_pipeline(
        TweedieRegressor(
            power=TWEEDIE_POWER,
            alpha=GLM_ALPHA,
            link="log",
            max_iter=GLM_MAX_ITER,
        )
    )
    # Même définition des colonnes que le Poisson : l'encodeur est
    # réajusté sur le sous-ensemble, les niveaux inconnus à la
    # prédiction retombent sur des zéros.
    cost_pipeline.fit(
        _model_matrix(cost),
        cost["CM"].to_numpy(dtype=float),
    )
    n_negative = int((train["FREQ"].gt(0) & train["CM"].lt(0)).sum())
    return BaselineGlm(
        pipeline_frequency=frequency_pipeline,
        pipeline_cost=cost_pipeline,
        n_frequency_rows=len(train),
        n_cost_rows=len(cost),
        n_negative_cost_excluded=n_negative,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Racine de l'erreur quadratique moyenne.

    Parameters
    ----------
    y_true : array-like
        Charge observée.
    y_pred : array-like
        Charge prédite, même longueur.

    Returns
    -------
    float
        RMSE.
    """
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def score_holdout(model: BaselineGlm, holdout: pd.DataFrame) -> HoldoutScore:
    """Compare la baseline au plancher « prédire 0 » sur le hold-out.

    Parameters
    ----------
    model : BaselineGlm
        Modèles ajustés sur la partie complémentaire.
    holdout : DataFrame
        Lignes tenues à part, avec ``CHARGE``.

    Returns
    -------
    HoldoutScore
        Les deux RMSE et les effectifs d'apprentissage.
    """
    observed = holdout["CHARGE"].to_numpy(dtype=float)
    predicted = model.predict_charge(holdout)
    return HoldoutScore(
        rmse=rmse(observed, predicted),
        zero_rmse=rmse(observed, np.zeros(len(holdout), dtype=float)),
        n_holdout=len(holdout),
        n_frequency_rows=model.n_frequency_rows,
        n_cost_rows=model.n_cost_rows,
        n_negative_cost_excluded=model.n_negative_cost_excluded,
    )


def evaluate_baseline(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    test_size: float = HOLD_OUT_FRACTION,
    seed: int = SEED,
) -> HoldoutScore:
    """Découpe, ajuste et score la baseline sur un hold-out du train.

    Parameters
    ----------
    features : DataFrame
        Variables, une ligne par contrat.
    targets : DataFrame
        Cibles reliées par ``ID``.
    test_size : float, optional
        Part tenue à part. Défaut : 0,2.
    seed : int, optional
        Graine du mélange. Défaut : 42.

    Returns
    -------
    HoldoutScore
        RMSE de la baseline et du plancher zéro.
    """
    frame = assemble_baseline_frame(features, targets)
    train, holdout = split_rows(frame, test_size=test_size, seed=seed)
    model = fit_baseline(train)
    return score_holdout(model, holdout)


def evaluate_training_files(
    train_path: str,
    target_path: str,
    *,
    test_size: float = HOLD_OUT_FRACTION,
    seed: int = SEED,
) -> HoldoutScore:
    """Évalue la baseline à partir des deux CSV d'apprentissage.

    Parameters
    ----------
    train_path : str
        CSV des variables d'apprentissage.
    target_path : str
        CSV des cibles.
    test_size : float, optional
        Part tenue à part. Défaut : 0,2.
    seed : int, optional
        Graine du mélange. Défaut : 42.

    Returns
    -------
    HoldoutScore
        RMSE de la baseline et du plancher zéro.
    """
    features, targets = read_baseline_inputs(train_path, target_path)
    return evaluate_baseline(
        features,
        targets,
        test_size=test_size,
        seed=seed,
    )


def _glm_pipeline(estimator: PoissonRegressor | TweedieRegressor) -> Pipeline:
    """Encode les catégories et met les grandeurs à l'échelle, puis le GLM.

    L'échelle ne change pas le modèle quand ``alpha`` vaut 0 : elle
    évite qu'un capital en centaines de milliers écrase l'exposition,
    proche de 1, dans l'optimiseur. ``drop='first'`` retire le niveau
    de référence, sinon l'indicatrice complète est colinéaire avec
    la constante. Une modalité inconnue à la prédiction devient des
    zéros, donc ce niveau de référence.
    """
    encoder = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), list(AGGREGATE_COLUMNS) + [EXPOSURE_COLUMN]),
            (
                "category",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                list(CATEGORICAL_COLUMNS),
            ),
        ]
    )
    return Pipeline([("encode", encoder), ("glm", estimator)])


def _model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Colonnes dans l'ordre attendu par l'encodeur."""
    return frame.loc[:, list(MODEL_COLUMNS)]


def _require_model_frame(frame: pd.DataFrame) -> None:
    """Vérifie les colonnes du tableau déjà assemblé."""
    needed = MODEL_COLUMNS + ("FREQ", "CM")
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise BaselineError(
            "Colonnes absentes du tableau modèle : " + ", ".join(missing) + "."
        )
