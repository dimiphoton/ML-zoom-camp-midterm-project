"""Ridge et XGBoost sur un petit jeu fabriqué.

Les CSV du challenge ne sont pas lus ici. Le test officiel ne sert pas
à choisir le modèle. La grille est réduite à un réglage pour que le test
reste court.
"""

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from agri_fire.features.baseline import assemble_baseline_frame, baseline_source_columns
from agri_fire.models import selection as selection_module
from agri_fire.models.baseline import SEED, evaluate_baseline
from agri_fire.models.selection import (
    TreeParams,
    evaluate_selection_files,
    fit_ridge,
    fit_trees,
    prediction_bins,
    select_models,
)
from agri_fire.plots import CREAM, save_calibration_chart, save_error_gap_chart

_TINY_TREES = (TreeParams(2, 0.1, 5, 1.0),)


def _features(rows: list[dict]) -> pd.DataFrame:
    """Complète chaque ligne avec des surfaces et capitaux à zéro."""
    blank = {name: 0 for name in baseline_source_columns() if name != "ID"}
    records = []
    for row in rows:
        full = dict(blank)
        full.update(row)
        records.append(full)
    return pd.DataFrame(records)


def _paired() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six contrats, dont un coût négatif et un coût nul avec sinistre."""
    features = _features(
        [
            {"ID": 1, "ACTIVIT2": "ACT1", "VOCATION": "VOC1", "ZONE": 1, "ANNEE_ASSURANCE": 1.0},
            {"ID": 2, "ACTIVIT2": "ACT1", "VOCATION": "VOC1", "ZONE": 1, "ANNEE_ASSURANCE": 0.5},
            {"ID": 3, "ACTIVIT2": "ACT5", "VOCATION": "VOC6", "ZONE": 2, "ANNEE_ASSURANCE": 1.0},
            {"ID": 4, "ACTIVIT2": "ACT5", "VOCATION": "VOC6", "ZONE": 2, "ANNEE_ASSURANCE": 2.0},
            {"ID": 5, "ACTIVIT2": "ACT1", "VOCATION": "VOC6", "ZONE": 1, "ANNEE_ASSURANCE": 1.0},
            {"ID": 6, "ACTIVIT2": "ACT5", "VOCATION": "VOC1", "ZONE": 3, "ANNEE_ASSURANCE": 0.8},
        ]
    )
    targets = pd.DataFrame(
        [
            {"ID": 1, "FREQ": 0.0, "CM": 0.0, "CHARGE": 0.0},
            {"ID": 2, "FREQ": 2.0, "CM": 0.0, "CHARGE": 0.0},
            {"ID": 3, "FREQ": 1.0, "CM": 10.0, "CHARGE": 10.0},
            {"ID": 4, "FREQ": 1.0, "CM": -4.0, "CHARGE": -8.0},
            {"ID": 5, "FREQ": 1.0, "CM": 30.0, "CHARGE": 30.0},
            {"ID": 6, "FREQ": 0.0, "CM": 0.0, "CHARGE": 0.0},
        ]
    )
    return features, targets


def _repeated(copies: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Répète les six contrats pour qu'une découpe à 20 % ait les deux morceaux."""
    features, targets = _paired()
    features = pd.concat(
        [features.assign(ID=features["ID"] + shift * 10) for shift in range(copies)],
        ignore_index=True,
    )
    targets = pd.concat(
        [targets.assign(ID=targets["ID"] + shift * 10) for shift in range(copies)],
        ignore_index=True,
    )
    return features, targets


def _select():
    """Sélection sur le jeu répété, avec une grille d'un seul réglage."""
    features, targets = _repeated()
    return select_models(
        features,
        targets,
        seed=SEED,
        ridge_alphas=(3.5,),
        tree_grid=_TINY_TREES,
        n_jobs=1,
    )


def test_ridge_and_trees_rebuild_charge_and_drop_negative_cost() -> None:
    """Le coût négatif n'ajuste pas le coût. La charge reste le produit."""
    frame = assemble_baseline_frame(*_paired())
    ridge = fit_ridge(frame, 1.0, 1.0)
    trees = fit_trees(frame, _TINY_TREES[0], n_jobs=1)

    for model in (ridge, trees):
        assert model.n_frequency_rows == len(frame)
        assert model.n_cost_rows == 3
        assert model.n_negative_cost_excluded == 1
        frequency = model.predict_frequency(frame)
        cost = model.predict_cost(frame)
        exposure = frame["ANNEE_ASSURANCE"].to_numpy(dtype=float)
        assert np.isfinite(frequency).all()
        assert np.isfinite(cost).all()
        assert (frequency >= 0).all()
        assert (cost >= 0).all()
        np.testing.assert_allclose(
            model.predict_charge(frame),
            frequency * cost * exposure,
        )


def test_select_models_uses_the_baseline_split_and_the_lowest_rmse() -> None:
    """Même hold-out que la baseline. Le retenu a le RMSE le plus bas."""
    features, targets = _repeated()
    baseline = evaluate_baseline(features, targets, seed=SEED)
    result = _select()

    assert result.n_holdout == baseline.n_holdout
    assert result.n_frequency_rows + result.n_holdout == len(features)
    assert result.n_fit_rows + result.n_tune_rows == result.n_frequency_rows
    assert abs(result.n_holdout / len(features) - 0.2) < 0.02
    assert abs(result.n_tune_rows / result.n_frequency_rows - 0.2) < 0.05
    assert result.score("poisson_tweedie").rmse == pytest.approx(baseline.rmse)
    assert result.score("zero").rmse == pytest.approx(baseline.zero_rmse)
    assert result.n_cost_rows < result.n_frequency_rows
    assert result.n_negative_cost_excluded > 0
    assert result.ridge_alpha_frequency == 3.5
    assert result.ridge_alpha_cost == 3.5
    assert result.tree_params == _TINY_TREES[0]
    assert result.trials["selected"].sum() == 2

    scores = {item.name: item.rmse for item in result.scores}
    assert scores[result.chosen] == pytest.approx(min(scores.values()))
    assert result.chosen in {"zero", "poisson_tweedie", "ridge", "xgboost"}
    assert result.calibration["model"].nunique() == 3
    assert result.score("zero").rmse_zero_charge == pytest.approx(0.0)


def test_prediction_bins_follow_the_predicted_charge() -> None:
    """La dernière tranche a la prédiction moyenne la plus haute."""
    observed = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 40.0])
    predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    table = prediction_bins(observed, predicted, n_bins=3)

    assert table["bin"].tolist() == [1, 2, 3]
    assert table["n"].sum() == 6
    assert table["mean_predicted"].is_monotonic_increasing
    assert table["mean_observed"].iloc[-1] == pytest.approx(20.0)


def test_gap_and_calibration_charts_use_the_cream_background(tmp_path: Path) -> None:
    """Les deux PNG ont le fond crème du thème, pas un cadre blanc."""
    scores = pd.DataFrame(
        {
            "key": ["zero", "poisson_tweedie", "ridge", "xgboost"],
            "rmse": [10.0, 12.0, 9.0, 11.0],
        }
    )
    bins = pd.DataFrame(
        {
            "model": ["ridge", "ridge", "xgboost", "xgboost", "poisson_tweedie", "poisson_tweedie"],
            "mean_predicted": [1.0, 4.0, 1.5, 5.0, 0.5, 3.0],
            "mean_observed": [0.2, 6.0, 0.4, 3.0, 0.1, 2.0],
        }
    )
    gap = save_error_gap_chart(scores, tmp_path / "ecart.png", language="fr", chosen="ridge")
    curve = save_calibration_chart(bins, tmp_path / "calibration.png", language="en", chosen="ridge")

    for path in (gap, curve):
        image = plt.imread(path)
        red, green, blue = (int(channel, 16) / 255 for channel in (CREAM[1:3], CREAM[3:5], CREAM[5:7]))
        corner = image[0, 0, :3]
        assert corner[0] == pytest.approx(red, abs=0.03)
        assert corner[1] == pytest.approx(green, abs=0.03)
        assert corner[2] == pytest.approx(blue, abs=0.03)
        assert float(corner.sum()) < 2.95


def test_selection_files_do_not_need_the_official_test(tmp_path: Path) -> None:
    """La lecture ne demande que le train et les cibles."""
    features, targets = _repeated(copies=12)
    train_path = tmp_path / "train.csv"
    target_path = tmp_path / "cibles.csv"
    features.to_csv(train_path, index=False)
    targets.to_csv(target_path, index=False)

    result = evaluate_selection_files(
        str(train_path),
        str(target_path),
        ridge_alphas=(1.0,),
        tree_grid=_TINY_TREES,
        n_jobs=1,
    )
    assert result.n_holdout > 0
    assert result.chosen in {"zero", "poisson_tweedie", "ridge", "xgboost"}


def test_selection_module_does_not_name_the_official_test() -> None:
    """Aucun chemin vers les 95 852 lignes du test officiel."""
    source = inspect.getsource(selection_module)
    assert "test_input" not in source
    assert "95852" not in source
    assert "95_852" not in source
