"""Baseline Poisson + Tweedie sur un petit jeu fabriqué.

Les CSV du challenge ne sont pas lus ici. Le test officiel (95 852 lignes)
ne sert pas à choisir le modèle.
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_fire.features.baseline import (
    MODEL_COLUMNS,
    BaselineError,
    assemble_baseline_frame,
    baseline_source_columns,
)
from agri_fire.models.baseline import (
    SEED,
    cost_training_rows,
    evaluate_baseline,
    fit_baseline,
    rmse,
    split_rows,
)
from agri_fire.models import baseline as baseline_module


def _features(rows: list[dict]) -> pd.DataFrame:
    """Complète chaque ligne avec des surfaces et capitaux à zéro."""
    blank = {name: 0 for name in baseline_source_columns() if name != "ID"}
    records = []
    for row in rows:
        full = dict(blank)
        full.update(row)
        records.append(full)
    return pd.DataFrame(records)


def _targets(rows: list[dict]) -> pd.DataFrame:
    """Cibles alignées sur les identifiants fournis."""
    return pd.DataFrame(rows)


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
    targets = _targets(
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


def test_source_columns_stay_short() -> None:
    """Activité, vocation, zone, exposition, surfaces et capitaux. Pas 374."""
    columns = baseline_source_columns()
    assert "ACTIVIT2" in columns
    assert "VOCATION" in columns
    assert "ZONE" in columns
    assert "ANNEE_ASSURANCE" in columns
    assert "SURFACE2" in columns
    assert "SURFACE6" in columns
    assert "KAPITAL10" in columns
    assert len(columns) == len(set(columns))
    assert len(columns) < 80
    assert len(MODEL_COLUMNS) == 7


def test_aggregates_skip_copies_and_read_open_bins() -> None:
    """La somme ne compte pas deux fois une copie, et lit ``7000+`` comme 7000."""
    features = _features(
        [
            {
                "ID": 1,
                "ACTIVIT2": "ACT1",
                "VOCATION": "VOC1",
                "ZONE": 4,
                "ANNEE_ASSURANCE": 1.0,
                "SURFACE1": 10,
                "SURFACE2": 99,
                "SURFACE3": np.nan,
                "SURFACE4": "7000+",
                "SURFACE6": "80",
                "KAPITAL1": 1,
                "KAPITAL2": 0,
                "KAPITAL10": 500,
                "KAPITAL11": 40,
                "KAPITAL12": 2,
                "KAPITAL34": "O",
                "KAPITAL35": "N",
            }
        ]
    )
    targets = _targets([{"ID": 1, "FREQ": 0.0, "CM": 0.0, "CHARGE": 0.0}])
    frame = assemble_baseline_frame(features, targets)

    assert frame.loc[0, "surface_sum"] == pytest.approx(7010.0)
    assert frame.loc[0, "capital_amount"] == pytest.approx(42.0)
    assert frame.loc[0, "capital_flags"] == pytest.approx(2.0)
    assert frame.loc[0, "ZONE"] == "4"


def test_missing_column_and_unmatched_id_fail() -> None:
    """Une colonne absente ou un ID orphelin arrête l'assemblage."""
    features, targets = _paired()
    with pytest.raises(BaselineError, match="ZONE"):
        assemble_baseline_frame(features.drop(columns="ZONE"), targets)
    orphan = targets.copy()
    orphan.loc[0, "ID"] = 99
    with pytest.raises(BaselineError, match="ID"):
        assemble_baseline_frame(features, orphan)


def test_split_is_random_by_row_with_seed_42() -> None:
    """Même graine, même découpe. Une autre graine mélange autrement."""
    frame = assemble_baseline_frame(*_paired())
    # On répète les six lignes pour que les deux graines puissent diverger.
    copies = [
        frame.assign(ID=frame["ID"] + shift * 10) for shift in range(8)
    ]
    frame = pd.concat(copies, ignore_index=True)
    train, holdout = split_rows(frame, seed=SEED)
    train_again, holdout_again = split_rows(frame, seed=SEED)
    pd.testing.assert_frame_equal(train, train_again)
    pd.testing.assert_frame_equal(holdout, holdout_again)

    other_train, other_holdout = split_rows(frame, seed=7)
    assert set(holdout["ID"]) != set(other_holdout["ID"])
    assert set(train["ID"]).isdisjoint(holdout["ID"])
    assert set(train["ID"]) | set(holdout["ID"]) == set(frame["ID"])
    assert len(train) + len(holdout) == len(frame)
    assert abs(len(holdout) / len(frame) - 0.2) < 0.02
    assert SEED == 42
    assert set(other_train["ID"]).isdisjoint(other_holdout["ID"])


def test_cost_rows_keep_zero_cost_and_drop_negatives() -> None:
    """Sinistre à coût nul : oui. Coût négatif : non. Ligne sans sinistre : non."""
    frame = assemble_baseline_frame(*_paired())
    cost = cost_training_rows(frame)

    assert set(cost["ID"]) == {2, 3, 5}
    assert (cost["CM"] >= 0).all()
    assert (cost["FREQ"] > 0).all()
    assert 0.0 in set(cost["CM"])
    assert -4.0 not in set(cost["CM"])


def test_poisson_uses_every_training_row_tweedie_does_not() -> None:
    """Le Poisson voit aussi le coût négatif. La Tweedie non."""
    frame = assemble_baseline_frame(*_paired())
    model = fit_baseline(frame)

    assert model.n_frequency_rows == len(frame)
    assert model.n_cost_rows == len(cost_training_rows(frame))
    assert model.n_negative_cost_excluded == 1


def test_predicted_charge_is_frequency_times_cost_times_exposure() -> None:
    """La formule de la charge est appliquée après les deux GLM."""
    frame = assemble_baseline_frame(*_paired())
    model = fit_baseline(frame)
    scored = frame.copy()
    scored["ZONE"] = "zone-inconnue"

    frequency = model.predict_frequency(scored)
    cost = model.predict_cost(scored)
    charge = model.predict_charge(scored)
    exposure = scored["ANNEE_ASSURANCE"].to_numpy(dtype=float)

    assert np.isfinite(charge).all()
    assert (frequency >= 0).all()
    assert (cost >= 0).all()
    np.testing.assert_allclose(charge, frequency * cost * exposure)


def test_holdout_rmse_includes_a_negative_charge() -> None:
    """Le coût négatif tenu à part compte dans le RMSE, pas dans la Tweedie."""
    features, targets = _paired()
    # Quatre premières lignes apprises, les deux dernières tenues à part
    # à la main : l'une a une charge négative.
    frame = assemble_baseline_frame(features, targets)
    # On force la découpe pour que le coût négatif soit dans le hold-out
    # et qu'il reste assez de sinistres à coût positif pour ajuster.
    train = frame[frame["ID"].isin([1, 2, 3, 5])].reset_index(drop=True)
    holdout = frame[frame["ID"].isin([4, 6])].reset_index(drop=True)
    # ID 4 est le seul sinistre du hold-out, et son coût est négatif.
    # L'apprentissage du coût ne le voit pas.
    model = fit_baseline(train)
    assert model.n_negative_cost_excluded == 0
    assert 4 not in set(cost_training_rows(train)["ID"])

    observed = holdout["CHARGE"].to_numpy(dtype=float)
    assert (observed < 0).any()
    predicted = model.predict_charge(holdout)
    assert rmse(observed, predicted) == pytest.approx(
        float(np.sqrt(np.mean((observed - predicted) ** 2)))
    )
    assert rmse(observed, np.zeros(len(observed))) == pytest.approx(
        float(np.sqrt(np.mean(observed**2)))
    )


def test_evaluate_baseline_scores_a_held_out_slice() -> None:
    """La fonction complète renvoie les deux RMSE, sans fichier de test."""
    features, targets = _paired()
    features = pd.concat(
        [features.assign(ID=features["ID"] + shift * 10) for shift in range(10)],
        ignore_index=True,
    )
    targets = pd.concat(
        [targets.assign(ID=targets["ID"] + shift * 10) for shift in range(10)],
        ignore_index=True,
    )
    score = evaluate_baseline(features, targets, seed=SEED)
    assert score.n_holdout > 0
    assert score.n_frequency_rows > score.n_holdout
    assert score.n_cost_rows < score.n_frequency_rows
    assert score.n_negative_cost_excluded > 0
    assert score.rmse >= 0
    assert score.zero_rmse > 0


def test_rmse_counts_a_negative_row() -> None:
    """Prédire 0 sur (0, -6) donne un RMSE de sqrt(18)."""
    assert rmse(np.array([0.0, -6.0]), np.array([0.0, 0.0])) == pytest.approx(18**0.5)


def test_training_csv_roundtrip_does_not_need_the_official_test(tmp_path: Path) -> None:
    """La lecture ne demande que le train et les cibles."""
    features, targets = _paired()
    features["SURFACE4"] = ["7000+", "0", "0", "0", "0", "0"]
    features["KAPITAL34"] = ["O", "N", "N", "N", "N", "N"]
    train_path = tmp_path / "train.csv"
    target_path = tmp_path / "cibles.csv"
    features.to_csv(train_path, index=False)
    targets.to_csv(target_path, index=False)

    from agri_fire.features.baseline import read_baseline_inputs

    read_features, read_targets = read_baseline_inputs(train_path, target_path)
    frame = assemble_baseline_frame(read_features, read_targets)
    assert frame.loc[frame["ID"] == 1, "surface_sum"].iloc[0] == pytest.approx(7000.0)
    assert frame.loc[frame["ID"] == 1, "capital_flags"].iloc[0] == pytest.approx(1.0)


def test_model_module_does_not_name_the_official_test() -> None:
    """Aucun chemin vers les 95 852 lignes du test officiel."""
    source = inspect.getsource(baseline_module)
    feature_source = inspect.getsource(
        __import__("agri_fire.features.baseline", fromlist=["baseline"])
    )
    for text in (source, feature_source):
        assert "test_input" not in text
        assert "95852" not in text
        assert "95_852" not in text
