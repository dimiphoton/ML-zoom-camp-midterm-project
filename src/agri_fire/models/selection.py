"""Régression ridge et arbres XGBoost, puis choix au RMSE sur la charge.

Même découpe que la baseline : 20 % des lignes tenus à part, graine 42.
Même règle sur le coût moyen : il ne s'apprend que sur les lignes avec
au moins un sinistre et un coût positif ou nul. Les coûts négatifs
restent dans le RMSE.

Le réglage ne voit pas la partie tenue à part. Il utilise une seconde
découpe, à l'intérieur de la partie apprise, avec la même graine et la
même fraction. La partie tenue à part sert une fois, pour comparer le
modèle linéaire, les arbres, la baseline Poisson–Tweedie et le plancher
« prédire 0 ».
"""

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from agri_fire.features.baseline import (
    AGGREGATE_COLUMNS,
    CATEGORICAL_COLUMNS,
    EXPOSURE_COLUMN,
    MODEL_COLUMNS,
    assemble_baseline_frame,
    read_baseline_inputs,
)
from agri_fire.models.baseline import (
    HOLD_OUT_FRACTION,
    SEED,
    TWEEDIE_POWER,
    cost_training_rows,
    fit_baseline,
    rmse,
    split_rows,
)

# Pénalité de la droite. Une valeur haute ramène les pentes vers zéro
# et laisse l'ordonnée à l'origine près de la moyenne : c'est ce que le
# RMSE récompense quand presque toutes les charges sont nulles.
DEFAULT_RIDGE_ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)

# Ordre du plus simple au plus flexible, en cas d'égalité du RMSE.
_PREFERENCE = ("zero", "poisson_tweedie", "ridge", "xgboost")
_TIE = 1e-9


class SelectionError(ValueError):
    """Le tableau ou la grille ne permet pas d'ajuster les deux morceaux."""


@dataclass(frozen=True)
class TreeParams:
    """Un réglage d'arbre, commun à la fréquence et au coût.

    Attributes
    ----------
    max_depth : int
        Profondeur maximale.
    learning_rate : float
        Pas de l'amplification.
    n_estimators : int
        Nombre d'arbres.
    min_child_weight : float
        Effectif minimum dans une feuille, au sens XGBoost.
    """

    max_depth: int
    learning_rate: float
    n_estimators: int
    min_child_weight: float


DEFAULT_TREE_GRID: tuple[TreeParams, ...] = (
    TreeParams(2, 0.08, 120, 40),
    TreeParams(2, 0.08, 280, 80),
    TreeParams(3, 0.05, 200, 40),
    TreeParams(3, 0.05, 400, 80),
    TreeParams(4, 0.05, 250, 40),
    TreeParams(4, 0.10, 150, 15),
    TreeParams(6, 0.05, 200, 20),
    TreeParams(6, 0.10, 100, 5),
)


@dataclass
class TwoPartModel:
    """Fréquence et coût déjà ajustés sur la même partie apprise.

    La charge prédite est le produit des deux, fois les années assurées.
    Une composante négative est ramenée à zéro : elle ne décrit pas une
    prime. La charge observée, même négative, n'est pas modifiée.

    Attributes
    ----------
    kind : str
        ``ridge`` ou ``xgboost``.
    encoder_frequency, encoder_cost : ColumnTransformer
        Encodages appris, le second sur les seules lignes de coût.
    frequency_model, cost_model
        Modèles déjà ajustés.
    n_frequency_rows, n_cost_rows, n_negative_cost_excluded : int
        Effectifs utilisés.
    n_frequency_features, n_cost_features : int
        Colonnes après encodage.
    settings : dict
        Pénalités ou réglage d'arbre retenus.
    """

    kind: str
    encoder_frequency: ColumnTransformer
    encoder_cost: ColumnTransformer
    frequency_model: Ridge | XGBRegressor
    cost_model: Ridge | XGBRegressor
    n_frequency_rows: int
    n_cost_rows: int
    n_negative_cost_excluded: int
    n_frequency_features: int
    n_cost_features: int
    settings: dict

    def predict_frequency(self, frame: pd.DataFrame) -> np.ndarray:
        """Fréquence prédite, positive ou nulle.

        Parameters
        ----------
        frame : DataFrame
            Tableau au format de ``assemble_baseline_frame``.

        Returns
        -------
        ndarray
            ``FREQ`` prédite.
        """
        encoded = _transform(self.encoder_frequency, frame)
        return _non_negative(self.frequency_model.predict(encoded))

    def predict_cost(self, frame: pd.DataFrame) -> np.ndarray:
        """Coût moyen prédit, positif ou nul.

        Parameters
        ----------
        frame : DataFrame
            Tableau au format de ``assemble_baseline_frame``.

        Returns
        -------
        ndarray
            ``CM`` prédit.
        """
        encoded = _transform(self.encoder_cost, frame)
        return _non_negative(self.cost_model.predict(encoded))

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
class CandidateScore:
    """RMSE d'un candidat sur la partie tenue à part.

    Attributes
    ----------
    name : str
        ``zero``, ``poisson_tweedie``, ``ridge`` ou ``xgboost``.
    rmse : float
        RMSE de la charge, toutes les lignes tenues à part.
    rmse_zero_charge, rmse_nonzero_charge : float
        Même RMSE, restreint aux charges nulles ou non nulles.
        Une charge négative reste dans le second groupe.
    n_zero_charge, n_nonzero_charge : int
        Effectifs de ces deux groupes.
    mean_predicted : float
        Moyenne de la charge prédite.
    """

    name: str
    rmse: float
    rmse_zero_charge: float
    rmse_nonzero_charge: float
    n_zero_charge: int
    n_nonzero_charge: int
    mean_predicted: float


@dataclass
class SelectionResult:
    """Comparaison sur une même partie tenue à part, et le nom retenu.

    Attributes
    ----------
    chosen : str
        Candidat au RMSE le plus bas. À égalité, le plus simple gagne :
        plancher, puis baseline, puis ridge, puis arbres.
    scores : tuple of CandidateScore
        Les quatre scores, dans cet ordre de simplicité.
    ridge_alpha_frequency, ridge_alpha_cost : float
        Pénalités retenues après la recherche interne.
    ridge_inner_rmse, tree_inner_rmse : float
        RMSE de ces réglages sur la découpe interne, pas sur le hold-out.
    tree_params : TreeParams
        Réglage d'arbre retenu.
    trials : DataFrame
        Chaque essai de la recherche interne (``inner_rmse``, ``selected``).
    calibration : DataFrame
        Charge moyenne prédite et observée par tranche de prédiction.
    n_holdout, n_frequency_rows, n_cost_rows : int
        Effectifs de la comparaison et de l'apprentissage final.
    n_negative_cost_excluded : int
        Coûts négatifs retirés de l'apprentissage du coût.
    n_negative_cost_in_holdout : int
        Ces coûts tombés dans la partie tenue à part.
    n_fit_rows, n_tune_rows : int
        Découpe interne : lignes qui ajustent, lignes qui règlent.
    n_frequency_features, n_cost_features : int
        Largeur après encodage, sur le modèle ridge final.
    """

    chosen: str
    scores: tuple[CandidateScore, ...]
    ridge_alpha_frequency: float
    ridge_alpha_cost: float
    ridge_inner_rmse: float
    tree_params: TreeParams
    tree_inner_rmse: float
    trials: pd.DataFrame
    calibration: pd.DataFrame
    n_holdout: int
    n_frequency_rows: int
    n_cost_rows: int
    n_negative_cost_excluded: int
    n_negative_cost_in_holdout: int
    n_fit_rows: int
    n_tune_rows: int
    n_frequency_features: int
    n_cost_features: int

    def score(self, name: str) -> CandidateScore:
        """Retrouve un des quatre scores par son nom.

        Parameters
        ----------
        name : str
            ``zero``, ``poisson_tweedie``, ``ridge`` ou ``xgboost``.

        Returns
        -------
        CandidateScore
            Le score de ce candidat.

        Raises
        ------
        SelectionError
            Si le nom n'est pas dans la comparaison.
        """
        for item in self.scores:
            if item.name == name:
                return item
        raise SelectionError(f"Candidat inconnu : {name}.")


def fit_ridge(
    train: pd.DataFrame,
    alpha_frequency: float,
    alpha_cost: float,
) -> TwoPartModel:
    """Ajuste deux régressions ridge, fréquence puis coût.

    Parameters
    ----------
    train : DataFrame
        Partie d'apprentissage, toutes les lignes que le modèle final voit.
    alpha_frequency, alpha_cost : float
        Pénalités L2. La même valeur n'est pas imposée aux deux morceaux.

    Returns
    -------
    TwoPartModel
        Droites ajustées.

    Raises
    ------
    SelectionError
        Si ``FREQ`` est négatif ou s'il n'y a aucune ligne de coût.
    """
    _require_frame(train)
    frequency, cost, n_negative = _targets(train)
    encoder_frequency, encoder_cost, x_frequency, x_cost = _fit_encoders(train, cost)
    frequency_model = Ridge(alpha=float(alpha_frequency))
    frequency_model.fit(x_frequency, frequency)
    cost_model = Ridge(alpha=float(alpha_cost))
    cost_model.fit(x_cost, cost["CM"].to_numpy(dtype=float))
    return TwoPartModel(
        kind="ridge",
        encoder_frequency=encoder_frequency,
        encoder_cost=encoder_cost,
        frequency_model=frequency_model,
        cost_model=cost_model,
        n_frequency_rows=len(train),
        n_cost_rows=len(cost),
        n_negative_cost_excluded=n_negative,
        n_frequency_features=int(x_frequency.shape[1]),
        n_cost_features=int(x_cost.shape[1]),
        settings={
            "alpha_frequency": float(alpha_frequency),
            "alpha_cost": float(alpha_cost),
        },
    )


def fit_trees(
    train: pd.DataFrame,
    params: TreeParams,
    *,
    n_jobs: int = 1,
) -> TwoPartModel:
    """Ajuste la fréquence au carré de l'erreur, le coût en Tweedie.

    ``FREQ`` est un taux, pas un comptage entier : l'objectif poisson
    d'XGBoost n'est pas celui retenu. Le coût reste dans la même famille
    de variance que la baseline (``power`` 1,5) et accepte les coûts nuls.

    Parameters
    ----------
    train : DataFrame
        Partie d'apprentissage.
    params : TreeParams
        Profondeur, pas, nombre d'arbres, taille de feuille.
    n_jobs : int, optional
        Fils d'exécution XGBoost. Défaut : 1.

    Returns
    -------
    TwoPartModel
        Arbres ajustés.

    Raises
    ------
    SelectionError
        Si ``FREQ`` est négatif ou s'il n'y a aucune ligne de coût.
    """
    _require_frame(train)
    frequency, cost, n_negative = _targets(train)
    encoder_frequency, encoder_cost, x_frequency, x_cost = _fit_encoders(train, cost)
    y_cost = cost["CM"].to_numpy(dtype=float)
    frequency_model = _frequency_tree(params, n_jobs=n_jobs, base_score=_base_score(frequency))
    frequency_model.fit(x_frequency, frequency)
    cost_model = _cost_tree(params, n_jobs=n_jobs, base_score=_base_score(y_cost))
    cost_model.fit(x_cost, y_cost)
    return TwoPartModel(
        kind="xgboost",
        encoder_frequency=encoder_frequency,
        encoder_cost=encoder_cost,
        frequency_model=frequency_model,
        cost_model=cost_model,
        n_frequency_rows=len(train),
        n_cost_rows=len(cost),
        n_negative_cost_excluded=n_negative,
        n_frequency_features=int(x_frequency.shape[1]),
        n_cost_features=int(x_cost.shape[1]),
        settings=asdict(params),
    )


def select_models(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    test_size: float = HOLD_OUT_FRACTION,
    seed: int = SEED,
    inner_size: float = HOLD_OUT_FRACTION,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    tree_grid: tuple[TreeParams, ...] = DEFAULT_TREE_GRID,
    n_jobs: int = 1,
    n_bins: int = 6,
    verbose: bool = False,
) -> SelectionResult:
    """Règle les deux familles et choisit sur la partie tenue à part.

    Parameters
    ----------
    features, targets : DataFrame
        Variables et cibles reliées par ``ID``.
    test_size : float, optional
        Part tenue à part pour la comparaison. Défaut : 0,2.
    seed : int, optional
        Graine des deux découpes. Défaut : 42.
    inner_size : float, optional
        Part de l'apprentissage réservée au réglage. Défaut : 0,2.
    ridge_alphas : tuple of float, optional
        Pénalités essayées, séparément pour la fréquence et le coût.
    tree_grid : tuple of TreeParams, optional
        Réglages d'arbres essayés.
    n_jobs : int, optional
        Fils d'exécution des arbres. Défaut : 1.
    n_bins : int, optional
        Tranches de la courbe de calibration. Défaut : 6.
    verbose : bool, optional
        Affiche l'avancement. Défaut : faux.

    Returns
    -------
    SelectionResult
        Scores, réglages retenus et nom choisi.

    Raises
    ------
    SelectionError
        Si une grille est vide ou si un morceau ne peut pas s'ajuster.
    """
    if len(ridge_alphas) == 0 or len(tree_grid) == 0:
        raise SelectionError("La grille de réglage est vide.")
    frame = assemble_baseline_frame(features, targets)
    train, holdout = split_rows(frame, test_size=test_size, seed=seed)
    fit_part, tune_part = split_rows(train, test_size=inner_size, seed=seed)
    _say(
        verbose,
        f"Découpe : {len(holdout)} lignes tenues à part, {len(train)} pour apprendre "
        f"dont {len(tune_part)} pour le réglage.",
    )

    _say(verbose, "Ridge : recherche de la pénalité.")
    alpha_frequency, alpha_cost, ridge_inner, ridge_trials = _search_ridge(
        fit_part, tune_part, ridge_alphas
    )
    _say(
        verbose,
        "Ridge retenu : "
        f"alpha fréquence {alpha_frequency:g}, alpha coût {alpha_cost:g}, "
        f"RMSE interne {ridge_inner:.2f}.",
    )
    ridge = fit_ridge(train, alpha_frequency, alpha_cost)

    _say(verbose, "Arbres : recherche des réglages.")
    tree_params, tree_inner, tree_trials = _search_trees(
        fit_part, tune_part, tree_grid, n_jobs=n_jobs, verbose=verbose
    )
    _say(
        verbose,
        "Arbres retenus : "
        f"profondeur {tree_params.max_depth}, "
        f"pas {tree_params.learning_rate:g}, "
        f"{tree_params.n_estimators} arbres, "
        f"feuille {tree_params.min_child_weight:g}, "
        f"RMSE interne {tree_inner:.2f}.",
    )
    trees = fit_trees(train, tree_params, n_jobs=n_jobs)

    _say(verbose, "Baseline Poisson–Tweedie.")
    baseline = fit_baseline(train)
    observed = holdout["CHARGE"].to_numpy(dtype=float)
    predicted = {
        "zero": np.zeros(len(holdout), dtype=float),
        "poisson_tweedie": baseline.predict_charge(holdout),
        "ridge": ridge.predict_charge(holdout),
        "xgboost": trees.predict_charge(holdout),
    }
    scores = tuple(_candidate(name, observed, predicted[name]) for name in _PREFERENCE)
    chosen = _prefer({item.name: item.rmse for item in scores})
    _say(verbose, "Retenu : " + chosen + f" (RMSE { _rmse_of(scores, chosen):.2f}).")

    calibration = pd.concat(
        [
            prediction_bins(observed, predicted[name], n_bins=n_bins).assign(model=name)
            for name in ("poisson_tweedie", "ridge", "xgboost")
        ],
        ignore_index=True,
    )
    trials = pd.concat([ridge_trials, tree_trials], ignore_index=True)
    n_negative_holdout = int((holdout["FREQ"].gt(0) & holdout["CM"].lt(0)).sum())
    return SelectionResult(
        chosen=chosen,
        scores=scores,
        ridge_alpha_frequency=alpha_frequency,
        ridge_alpha_cost=alpha_cost,
        ridge_inner_rmse=ridge_inner,
        tree_params=tree_params,
        tree_inner_rmse=tree_inner,
        trials=trials,
        calibration=calibration,
        n_holdout=len(holdout),
        n_frequency_rows=len(train),
        n_cost_rows=ridge.n_cost_rows,
        n_negative_cost_excluded=ridge.n_negative_cost_excluded,
        n_negative_cost_in_holdout=n_negative_holdout,
        n_fit_rows=len(fit_part),
        n_tune_rows=len(tune_part),
        n_frequency_features=ridge.n_frequency_features,
        n_cost_features=ridge.n_cost_features,
    )


def evaluate_selection_files(
    train_path: str,
    target_path: str,
    *,
    test_size: float = HOLD_OUT_FRACTION,
    seed: int = SEED,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    tree_grid: tuple[TreeParams, ...] = DEFAULT_TREE_GRID,
    n_jobs: int = 1,
    verbose: bool = False,
) -> SelectionResult:
    """Compare les modèles à partir des deux CSV d'apprentissage.

    Le fichier de test du challenge n'est pas ouvert.

    Parameters
    ----------
    train_path : str
        CSV des variables d'apprentissage.
    target_path : str
        CSV des cibles.
    test_size : float, optional
        Part tenue à part. Défaut : 0,2.
    seed : int, optional
        Graine des découpes. Défaut : 42.
    ridge_alphas : tuple of float, optional
        Pénalités essayées.
    tree_grid : tuple of TreeParams, optional
        Réglages d'arbres essayés.
    n_jobs : int, optional
        Fils d'exécution des arbres. Défaut : 1.
    verbose : bool, optional
        Affiche l'avancement. Défaut : faux.

    Returns
    -------
    SelectionResult
        Scores et modèle retenu.
    """
    features, targets = read_baseline_inputs(train_path, target_path)
    return select_models(
        features,
        targets,
        test_size=test_size,
        seed=seed,
        ridge_alphas=ridge_alphas,
        tree_grid=tree_grid,
        n_jobs=n_jobs,
        verbose=verbose,
    )


def prediction_bins(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    n_bins: int = 6,
) -> pd.DataFrame:
    """Découpe la prédiction en tranches d'effectif égal.

    Parameters
    ----------
    observed, predicted : array-like
        Charge observée et prédite, même longueur.
    n_bins : int, optional
        Nombre de tranches visé. Défaut : 6. Plafonné par le nombre de lignes.

    Returns
    -------
    DataFrame
        ``bin``, ``n``, ``mean_predicted``, ``mean_observed``, dans l'ordre
        croissant de la prédiction.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(observed) != len(predicted) or len(observed) == 0:
        raise SelectionError("Les charges observées et prédites n'ont pas la même longueur.")
    bins = min(int(n_bins), len(observed))
    order = np.argsort(predicted, kind="mergesort")
    edges = np.linspace(0, len(observed), bins + 1, dtype=int)
    rows = []
    for index in range(bins):
        chosen = order[edges[index] : edges[index + 1]]
        if len(chosen) == 0:
            continue
        rows.append(
            {
                "bin": index + 1,
                "n": int(len(chosen)),
                "mean_predicted": float(predicted[chosen].mean()),
                "mean_observed": float(observed[chosen].mean()),
            }
        )
    return pd.DataFrame(rows)


def format_selection(result: SelectionResult) -> str:
    """Résumé texte des RMSE et des réglages retenus.

    Parameters
    ----------
    result : SelectionResult
        Sortie de ``select_models``.

    Returns
    -------
    str
        Quelques lignes, sans tableau large.
    """
    lines = [
        f"retenu : {result.chosen}",
        (
            f"lignes : hold-out {result.n_holdout}, apprentissage {result.n_frequency_rows}, "
            f"coût {result.n_cost_rows}, coûts négatifs exclus {result.n_negative_cost_excluded}, "
            f"dont dans le hold-out {result.n_negative_cost_in_holdout}"
        ),
        (
            f"réglage interne : {result.n_fit_rows} ajustent, {result.n_tune_rows} règlent, "
            f"colonnes fréquence {result.n_frequency_features}, coût {result.n_cost_features}"
        ),
    ]
    for item in result.scores:
        lines.append(
            f"{item.name:16} RMSE {item.rmse:10.2f}  "
            f"charge nulle {item.rmse_zero_charge:10.2f}  "
            f"autre {item.rmse_nonzero_charge:10.2f}  "
            f"moyenne prédite {item.mean_predicted:8.2f}"
        )
    lines.append(
        f"ridge alpha fréquence {result.ridge_alpha_frequency:g}, "
        f"alpha coût {result.ridge_alpha_cost:g}, "
        f"RMSE interne {result.ridge_inner_rmse:.2f}"
    )
    params = result.tree_params
    lines.append(
        "arbres profondeur {depth}, pas {rate:g}, {trees} arbres, feuille {leaf:g}, "
        "RMSE interne {inner:.2f}".format(
            depth=params.max_depth,
            rate=params.learning_rate,
            trees=params.n_estimators,
            leaf=params.min_child_weight,
            inner=result.tree_inner_rmse,
        )
    )
    for family in ("ridge", "xgboost"):
        part = result.trials.loc[result.trials["family"].eq(family), "inner_rmse"]
        lines.append(
            f"{family} essais {len(part)}, RMSE interne "
            f"de {float(part.min()):.2f} à {float(part.max()):.2f}"
        )
    return "\n".join(lines)


def _search_ridge(
    fit_part: pd.DataFrame,
    tune_part: pd.DataFrame,
    alphas: tuple[float, ...],
) -> tuple[float, float, float, pd.DataFrame]:
    """Choisit les deux pénalités sur la découpe interne.

    Chaque pénalité n'est ajustée qu'une fois. Le score du couple est le
    RMSE de la charge sur les lignes de réglage : les deux morceaux ne
    sont pas notés séparément.
    """
    encoded = _encode_pair(fit_part, tune_part)
    frequency_hat = {}
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(encoded.x_frequency, encoded.y_frequency)
        frequency_hat[float(alpha)] = model.predict(encoded.x_frequency_tune)
    cost_hat = {}
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(encoded.x_cost, encoded.y_cost)
        cost_hat[float(alpha)] = model.predict(encoded.x_cost_tune)

    best_score = None
    best_pair = (float(alphas[0]), float(alphas[0]))
    rows = []
    for alpha_frequency in frequency_hat:
        for alpha_cost in cost_hat:
            predicted = _charge(
                frequency_hat[alpha_frequency],
                cost_hat[alpha_cost],
                encoded.exposure_tune,
            )
            score = rmse(encoded.charge_tune, predicted)
            rows.append(
                {
                    "family": "ridge",
                    "alpha_frequency": alpha_frequency,
                    "alpha_cost": alpha_cost,
                    "max_depth": np.nan,
                    "learning_rate": np.nan,
                    "n_estimators": np.nan,
                    "min_child_weight": np.nan,
                    "inner_rmse": score,
                }
            )
            # À score égal, la plus forte pénalité : moins de pente.
            more_penalty = (alpha_frequency, alpha_cost) > best_pair
            if best_score is None or score < best_score - _TIE or (
                abs(score - best_score) <= _TIE and more_penalty
            ):
                best_score = score
                best_pair = (alpha_frequency, alpha_cost)
    trials = pd.DataFrame(rows)
    trials["selected"] = trials["alpha_frequency"].eq(best_pair[0]) & trials["alpha_cost"].eq(
        best_pair[1]
    )
    return best_pair[0], best_pair[1], float(best_score), trials


def _search_trees(
    fit_part: pd.DataFrame,
    tune_part: pd.DataFrame,
    grid: tuple[TreeParams, ...],
    *,
    n_jobs: int,
    verbose: bool = False,
) -> tuple[TreeParams, float, pd.DataFrame]:
    """Choisit un réglage d'arbre sur la découpe interne."""
    encoded = _encode_pair(fit_part, tune_part)
    best_score = None
    best_params = grid[0]
    best_key = _tree_simplicity(grid[0])
    rows = []
    for params in grid:
        _say(
            verbose,
            f"  profondeur {params.max_depth}, pas {params.learning_rate:g}, "
            f"{params.n_estimators} arbres, feuille {params.min_child_weight:g}.",
        )
        frequency_model = _frequency_tree(
            params, n_jobs=n_jobs, base_score=_base_score(encoded.y_frequency)
        )
        frequency_model.fit(encoded.x_frequency, encoded.y_frequency)
        cost_model = _cost_tree(params, n_jobs=n_jobs, base_score=_base_score(encoded.y_cost))
        cost_model.fit(encoded.x_cost, encoded.y_cost)
        predicted = _charge(
            frequency_model.predict(encoded.x_frequency_tune),
            cost_model.predict(encoded.x_cost_tune),
            encoded.exposure_tune,
        )
        score = rmse(encoded.charge_tune, predicted)
        _say(verbose, f"  RMSE interne {score:.2f}.")
        rows.append(
            {
                "family": "xgboost",
                "alpha_frequency": np.nan,
                "alpha_cost": np.nan,
                "max_depth": params.max_depth,
                "learning_rate": params.learning_rate,
                "n_estimators": params.n_estimators,
                "min_child_weight": params.min_child_weight,
                "inner_rmse": score,
            }
        )
        key = _tree_simplicity(params)
        if best_score is None or score < best_score - _TIE or (
            abs(score - best_score) <= _TIE and key < best_key
        ):
            best_score = score
            best_params = params
            best_key = key
    trials = pd.DataFrame(rows)
    trials["selected"] = (
        trials["max_depth"].eq(best_params.max_depth)
        & trials["learning_rate"].eq(best_params.learning_rate)
        & trials["n_estimators"].eq(best_params.n_estimators)
        & trials["min_child_weight"].eq(best_params.min_child_weight)
    )
    return best_params, float(best_score), trials


@dataclass
class _EncodedPair:
    """Matrices de la découpe interne, fréquence et coût séparés."""

    x_frequency: np.ndarray
    y_frequency: np.ndarray
    x_cost: np.ndarray
    y_cost: np.ndarray
    x_frequency_tune: np.ndarray
    x_cost_tune: np.ndarray
    exposure_tune: np.ndarray
    charge_tune: np.ndarray


def _encode_pair(fit_part: pd.DataFrame, tune_part: pd.DataFrame) -> _EncodedPair:
    """Apprend l'encodage sur la partie qui ajuste, l'applique au réglage.

    L'encodage du coût ne voit que les lignes de coût. Il est réappris
    plus tard sur toute la partie d'apprentissage : celui-ci ne sert qu'à
    choisir le réglage.
    """
    _require_frame(fit_part)
    frequency, cost, _n_negative = _targets(fit_part)
    encoder_frequency, encoder_cost, x_frequency, x_cost = _fit_encoders(fit_part, cost)
    return _EncodedPair(
        x_frequency=x_frequency,
        y_frequency=frequency,
        x_cost=x_cost,
        y_cost=cost["CM"].to_numpy(dtype=float),
        x_frequency_tune=_transform(encoder_frequency, tune_part),
        x_cost_tune=_transform(encoder_cost, tune_part),
        exposure_tune=tune_part[EXPOSURE_COLUMN].to_numpy(dtype=float),
        charge_tune=tune_part["CHARGE"].to_numpy(dtype=float),
    )


def _fit_encoders(
    train: pd.DataFrame,
    cost: pd.DataFrame,
) -> tuple[ColumnTransformer, ColumnTransformer, np.ndarray, np.ndarray]:
    """Deux encodeurs : toutes les lignes, puis les seules lignes de coût."""
    encoder_frequency = _encoder()
    encoder_cost = _encoder()
    x_frequency = encoder_frequency.fit_transform(_matrix(train))
    x_cost = encoder_cost.fit_transform(_matrix(cost))
    return encoder_frequency, encoder_cost, x_frequency, x_cost


def _encoder() -> ColumnTransformer:
    """Même préparation que la baseline : échelle, puis indicatrices.

    ``drop='first'`` retire le niveau de référence. Une modalité absente
    de l'apprentissage du coût devient des zéros à la prédiction.
    """
    return ColumnTransformer(
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


def _frequency_tree(params: TreeParams, *, n_jobs: int, base_score: float) -> XGBRegressor:
    """Arbres sur le taux de sinistre, erreur au carré."""
    return XGBRegressor(
        objective="reg:squarederror",
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        n_estimators=params.n_estimators,
        min_child_weight=params.min_child_weight,
        base_score=base_score,
        random_state=SEED,
        n_jobs=n_jobs,
        tree_method="hist",
        verbosity=0,
    )


def _cost_tree(params: TreeParams, *, n_jobs: int, base_score: float) -> XGBRegressor:
    """Arbres Tweedie. Le point de départ est la moyenne, pas 0,5."""
    return XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=TWEEDIE_POWER,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        n_estimators=params.n_estimators,
        min_child_weight=params.min_child_weight,
        base_score=base_score,
        random_state=SEED,
        n_jobs=n_jobs,
        tree_method="hist",
        verbosity=0,
    )


def _targets(train: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Fréquence de toutes les lignes, et lignes de coût utilisables."""
    frequency = train["FREQ"].to_numpy(dtype=float)
    if np.isnan(frequency).any() or (frequency < 0).any():
        raise SelectionError("FREQ manquant ou négatif.")
    cost = cost_training_rows(train)
    if cost.empty:
        raise SelectionError(
            "Aucune ligne avec sinistre et coût positif ou nul pour le coût moyen."
        )
    n_negative = int((train["FREQ"].gt(0) & train["CM"].lt(0)).sum())
    return frequency, cost, n_negative


def _candidate(name: str, observed: np.ndarray, predicted: np.ndarray) -> CandidateScore:
    """RMSE global et RMSE selon que la charge observée est nulle."""
    silent = observed == 0
    return CandidateScore(
        name=name,
        rmse=rmse(observed, predicted),
        rmse_zero_charge=_rmse_on(observed, predicted, silent),
        rmse_nonzero_charge=_rmse_on(observed, predicted, ~silent),
        n_zero_charge=int(silent.sum()),
        n_nonzero_charge=int((~silent).sum()),
        mean_predicted=float(np.mean(predicted)),
    )


def _rmse_on(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    """RMSE sur un masque. NaN si le masque est vide."""
    if not mask.any():
        return float("nan")
    return rmse(observed[mask], predicted[mask])


def _prefer(scores: dict[str, float]) -> str:
    """Le RMSE le plus bas. À égalité, le modèle le plus simple."""
    best = min(scores.values())
    for name in _PREFERENCE:
        if abs(scores[name] - best) <= _TIE:
            return name
    return min(scores, key=scores.get)


def _tree_simplicity(params: TreeParams) -> tuple[float, ...]:
    """Plus petit veut dire plus contraint : moins profond, feuilles plus lourdes."""
    return (
        params.max_depth,
        -params.min_child_weight,
        params.n_estimators,
        params.learning_rate,
    )


def _charge(frequency: np.ndarray, cost: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Produit des composantes ramenées à zéro si elles sont négatives."""
    return _non_negative(frequency) * _non_negative(cost) * exposure


def _non_negative(values: np.ndarray) -> np.ndarray:
    """Une fréquence ou un coût prédit sous zéro ne sort pas du modèle."""
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _base_score(values: np.ndarray) -> float:
    """Moyenne de la cible, strictement positive pour la Tweedie."""
    mean = float(np.mean(values))
    if mean <= 0:
        return 1.0
    return mean


def _transform(encoder: ColumnTransformer, frame: pd.DataFrame) -> np.ndarray:
    """Encode. Une modalité inconnue devient des zéros, sans avertissement.

    C'est le comportement voulu : une zone sans sinistre dans
    l'apprentissage du coût retombe sur le niveau de référence.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories",
            category=UserWarning,
        )
        return encoder.transform(_matrix(frame))


def _matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Colonnes dans l'ordre attendu par l'encodeur."""
    return frame.loc[:, list(MODEL_COLUMNS)]


def _require_frame(frame: pd.DataFrame) -> None:
    """Vérifie les colonnes du tableau déjà assemblé."""
    needed = list(MODEL_COLUMNS) + ["FREQ", "CM", "CHARGE"]
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise SelectionError(
            "Colonnes absentes du tableau modèle : " + ", ".join(missing) + "."
        )


def _rmse_of(scores: tuple[CandidateScore, ...], name: str) -> float:
    """RMSE d'un nom dans le tuple déjà construit."""
    for item in scores:
        if item.name == name:
            return item.rmse
    return float("nan")


def _say(verbose: bool, message: str) -> None:
    """Avancement d'un apprentissage long. Rien en mode silencieux."""
    if verbose:
        print(message, flush=True)
