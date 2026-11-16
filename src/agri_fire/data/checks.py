"""Contrôles de grain sur le train et le test."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Effectif du train du challenge. Les tests passent un jeu plus petit.
EXPECTED_TRAIN_ROWS = 383_610
CHARGE_TOLERANCE = 1e-6
TARGET_COLUMNS = ("ID", "FREQ", "CM", "ANNEE_ASSURANCE", "CHARGE")


class DataCheckError(ValueError):
    """Un contrôle de grain a échoué."""


@dataclass(frozen=True)
class TrainCheck:
    """Résultat d'un contrôle de grain réussi.

    Attributes
    ----------
    n_rows : int
        Nombre de lignes du train.
    n_columns : int
        Nombre de colonnes du train, identique à celui du test.
    max_charge_gap : float
        Écart absolu maximal entre ``CHARGE`` et
        ``FREQ × CM × ANNEE_ASSURANCE``.
    max_claim_count_gap : float
        Écart absolu maximal entre ``FREQ × ANNEE_ASSURANCE`` et
        l'entier le plus proche.
    """

    n_rows: int
    n_columns: int
    max_charge_gap: float
    max_claim_count_gap: float


def check_train(
    train_path: Path | str,
    target_path: Path | str,
    test_path: Path | str,
    *,
    expected_rows: int = EXPECTED_TRAIN_ROWS,
    charge_tolerance: float = CHARGE_TOLERANCE,
) -> TrainCheck:
    """Charge le train et vérifie le grain, la formule et les colonnes.

    Le train est le CSV des variables. Les cibles (``FREQ``, ``CM``,
    ``ANNEE_ASSURANCE``, ``CHARGE``) sont lues à part et reliées par
    ``ID``. Le test n'est lu que pour ses noms de colonnes.

    Parameters
    ----------
    train_path : Path or str
        CSV du train, une ligne par contrat-exercice.
    target_path : Path or str
        CSV des cibles, une ligne par ``ID`` du train.
    test_path : Path or str
        CSV du test. Ses colonnes doivent être celles du train,
        dans le même ordre.
    expected_rows : int, optional
        Nombre de lignes attendu. Défaut : 383 610.
    charge_tolerance : float, optional
        Écart absolu admis sur la formule de la charge, et sur la
        distance du nombre de sinistres à un entier. Défaut : 1e-6.

    Returns
    -------
    TrainCheck
        Effectifs et écarts maximaux observés.

    Raises
    ------
    DataCheckError
        Si une vérification échoue. Le message liste chaque écart.
    """
    train = pd.read_csv(train_path)
    targets = pd.read_csv(target_path)
    test_columns = list(pd.read_csv(test_path, nrows=0).columns)

    problems: list[str] = []
    if len(train) != expected_rows:
        problems.append(
            f"Le train a {len(train)} lignes, {expected_rows} attendues."
        )

    problems.extend(_id_problems(train, targets))

    max_charge_gap = float("nan")
    max_claim_count_gap = float("nan")
    missing = [name for name in TARGET_COLUMNS if name not in targets.columns]
    if missing:
        problems.append(
            "Colonnes de cibles absentes : " + ", ".join(missing) + "."
        )
    else:
        max_charge_gap = _max_charge_gap(targets)
        max_claim_count_gap = _max_claim_count_gap(targets)
        if not np.isfinite(max_charge_gap) or max_charge_gap > charge_tolerance:
            problems.append(
                "CHARGE s'écarte de FREQ × CM × ANNEE_ASSURANCE "
                f"(écart max {max_charge_gap}, tolérance {charge_tolerance})."
            )
        if (
            not np.isfinite(max_claim_count_gap)
            or max_claim_count_gap > charge_tolerance
        ):
            problems.append(
                "FREQ × ANNEE_ASSURANCE n'est pas un nombre entier de "
                f"sinistres (écart max {max_claim_count_gap})."
            )

    if list(train.columns) != test_columns:
        problems.append(_column_problem(list(train.columns), test_columns))

    if problems:
        raise DataCheckError(" ".join(problems))

    return TrainCheck(
        n_rows=len(train),
        n_columns=len(train.columns),
        max_charge_gap=max_charge_gap,
        max_claim_count_gap=max_claim_count_gap,
    )


def _id_problems(train: pd.DataFrame, targets: pd.DataFrame) -> list[str]:
    """Vérifie que l'ID relie le train et les cibles un à un."""
    problems: list[str] = []
    if "ID" not in train.columns:
        return ["Colonne ID absente du train."]
    if not train["ID"].is_unique:
        problems.append("ID du train en double.")
    if "ID" not in targets.columns:
        problems.append("Colonne ID absente des cibles.")
        return problems
    if not targets["ID"].is_unique:
        problems.append("ID des cibles en double.")
    if set(train["ID"]) != set(targets["ID"]):
        problems.append("Les ID du train et des cibles ne correspondent pas.")
    return problems


def _max_charge_gap(targets: pd.DataFrame) -> float:
    """Écart absolu max entre la charge observée et le produit des trois facteurs."""
    product = (
        targets["FREQ"] * targets["CM"] * targets["ANNEE_ASSURANCE"]
    )
    gap = (targets["CHARGE"] - product).abs()
    if gap.isna().any():
        return float("nan")
    return float(gap.max())


def _max_claim_count_gap(targets: pd.DataFrame) -> float:
    """Écart max de FREQ × années à l'entier le plus proche.

    Ce produit est le nombre de sinistres. Il doit être entier, aux
    arrondis flottants près.
    """
    claim_count = targets["FREQ"] * targets["ANNEE_ASSURANCE"]
    if claim_count.isna().any():
        return float("nan")
    distance = (claim_count - claim_count.round()).abs()
    return float(distance.max())


def _column_problem(train_columns: list[str], test_columns: list[str]) -> str:
    """Décrit l'écart de schéma entre le train et le test."""
    only_train = [name for name in train_columns if name not in test_columns]
    only_test = [name for name in test_columns if name not in train_columns]
    if not only_train and not only_test:
        return "Les colonnes du train et du test ne sont pas dans le même ordre."
    details: list[str] = []
    if only_train:
        details.append("absentes du test : " + ", ".join(only_train))
    if only_test:
        details.append("absentes du train : " + ", ".join(only_test))
    return "Colonnes différentes (" + " ; ".join(details) + ")."
