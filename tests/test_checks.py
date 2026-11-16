"""Contrôles de grain sur un petit jeu fabriqué.

Les CSV du challenge (environ 697 Mo) ne sont pas lus ici.
"""

import inspect
from pathlib import Path

import pandas as pd
import pytest

from agri_fire.data.checks import (
    CHARGE_TOLERANCE,
    EXPECTED_TRAIN_ROWS,
    DataCheckError,
    check_train,
)


def _write_frames(
    directory: Path,
    train: pd.DataFrame,
    targets: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    """Écrit trois CSV et renvoie leurs chemins."""
    train_path = directory / "train.csv"
    target_path = directory / "cibles.csv"
    test_path = directory / "test.csv"
    train.to_csv(train_path, index=False)
    targets.to_csv(target_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, target_path, test_path


def _valid_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trois contrats dont la charge et le nombre de sinistres sont justes.

    La deuxième ligne a un coût moyen négatif : la formule reste exacte.
    La troisième a une exposition fractionnaire et un nombre de sinistres entier.
    """
    train = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "ADOSS": ["a", "b", "c"],
            "ZONE": ["z", "z", "z"],
        }
    )
    targets = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "FREQ": [0.0, 1.0, 2.0],
            "CM": [0.0, -5.0, 10.0],
            "ANNEE_ASSURANCE": [1.0, 1.0, 0.5],
            "CHARGE": [0.0, -5.0, 10.0],
        }
    )
    return train, targets, train.copy()


def test_default_contract_matches_the_challenge() -> None:
    """Le défaut vise les 383 610 lignes du train, tolérance 1e-6."""
    signature = inspect.signature(check_train)
    assert signature.parameters["expected_rows"].default == EXPECTED_TRAIN_ROWS
    assert EXPECTED_TRAIN_ROWS == 383_610
    assert signature.parameters["charge_tolerance"].default == CHARGE_TOLERANCE
    assert CHARGE_TOLERANCE == 1e-6


def test_valid_toy_passes(tmp_path: Path) -> None:
    """Un jeu cohérent passe, y compris un coût moyen négatif."""
    paths = _write_frames(tmp_path, *_valid_frames())
    result = check_train(*paths, expected_rows=3)

    assert result.n_rows == 3
    assert result.n_columns == 3
    assert result.max_charge_gap <= 1e-6
    assert result.max_claim_count_gap <= 1e-6


def test_gap_inside_tolerance_passes(tmp_path: Path) -> None:
    """Un écart de charge inférieur à 1e-6 reste accepté."""
    train, targets, test = _valid_frames()
    targets.loc[0, "CHARGE"] = 5e-7
    paths = _write_frames(tmp_path, train, targets, test)

    result = check_train(*paths, expected_rows=3)

    assert result.max_charge_gap == pytest.approx(5e-7)


def test_wrong_row_count_fails(tmp_path: Path) -> None:
    """Le contrôle refuse un effectif différent de celui attendu."""
    paths = _write_frames(tmp_path, *_valid_frames())

    with pytest.raises(DataCheckError, match="3 lignes, 383610 attendues"):
        check_train(*paths)


def test_duplicate_id_fails(tmp_path: Path) -> None:
    """Un ID répété dans le train est refusé."""
    train, targets, test = _valid_frames()
    train.loc[1, "ID"] = 1
    paths = _write_frames(tmp_path, train, targets, test)

    with pytest.raises(DataCheckError, match="ID du train en double"):
        check_train(*paths, expected_rows=3)


def test_charge_formula_fails(tmp_path: Path) -> None:
    """Une charge trop éloignée du produit est refusée."""
    train, targets, test = _valid_frames()
    targets.loc[2, "CHARGE"] = 10.0 + 1e-4
    paths = _write_frames(tmp_path, train, targets, test)

    with pytest.raises(DataCheckError, match="CHARGE s'écarte"):
        check_train(*paths, expected_rows=3)


def test_non_integer_claim_count_fails(tmp_path: Path) -> None:
    """FREQ × années non entier est refusé, même si la charge colle."""
    train, targets, test = _valid_frames()
    targets.loc[2, "FREQ"] = 0.5
    targets.loc[2, "ANNEE_ASSURANCE"] = 0.5
    targets.loc[2, "CM"] = 8.0
    targets.loc[2, "CHARGE"] = 2.0  # 0.5 × 8 × 0.5
    paths = _write_frames(tmp_path, train, targets, test)

    with pytest.raises(DataCheckError, match="nombre entier de sinistres"):
        check_train(*paths, expected_rows=3)


def test_different_columns_fail(tmp_path: Path) -> None:
    """Une colonne présente seulement d'un côté est refusée."""
    train, targets, test = _valid_frames()
    test = test.drop(columns=["ADOSS"])
    paths = _write_frames(tmp_path, train, targets, test)

    with pytest.raises(DataCheckError, match="ADOSS"):
        check_train(*paths, expected_rows=3)
