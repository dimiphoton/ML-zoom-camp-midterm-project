"""Passe d'un contrat JSON aux trois sorties de la ridge.

Le modèle ne voit pas les 374 colonnes du fichier. Il voit les sept
colonnes déjà assemblées : activité, vocation, zone, exposition, et
les trois agrégats. La charge renvoyée est le produit fréquence ×
coût moyen × années assurées, après le plancher à zéro de chaque
composante.
"""

import math

import pandas as pd

from agri_fire.features.baseline import (
    AGGREGATE_COLUMNS,
    CATEGORICAL_COLUMNS,
    EXPOSURE_COLUMN,
    MODEL_COLUMNS,
)
from agri_fire.models.selection import TwoPartModel

_NUMERIC_COLUMNS = (EXPOSURE_COLUMN,) + AGGREGATE_COLUMNS


class PredictionError(ValueError):
    """La requête n'a pas les colonnes du modèle, ou un nombre illisible."""


def contracts_frame(records: list[dict]) -> pd.DataFrame:
    """Met les contrats au format que la ridge attend.

    Parameters
    ----------
    records : list of dict
        Un dictionnaire par contrat. Chaque dictionnaire porte les
        colonnes de ``MODEL_COLUMNS``. Les catégories sont lues comme
        du texte. Un nombre entier (la zone ``1``) devient ``"1"``,
        le même texte que pendant l'apprentissage.

    Returns
    -------
    DataFrame
        Une ligne par contrat, uniquement les colonnes du modèle.

    Raises
    ------
    PredictionError
        Si la liste est vide, si une colonne manque, si un nombre
        n'est pas fini, ou si l'exposition est négative.
    """
    if not records:
        raise PredictionError("Aucun contrat dans la requête.")
    rows = [_one_row(record, index) for index, record in enumerate(records, start=1)]
    frame = pd.DataFrame(rows, columns=list(MODEL_COLUMNS))
    for name in CATEGORICAL_COLUMNS:
        frame[name] = frame[name].astype("string")
    for name in _NUMERIC_COLUMNS:
        frame[name] = frame[name].astype(float)
    return frame


def predict_contracts(model: TwoPartModel, records: list[dict]) -> list[dict[str, float]]:
    """Prédit fréquence, coût moyen et charge pour chaque contrat.

    Parameters
    ----------
    model : TwoPartModel
        Ridge déjà chargée depuis l'artefact.
    records : list of dict
        Contrats au format de ``contracts_frame``.

    Returns
    -------
    list of dict
        Pour chaque contrat, ``FREQ``, ``CM`` et ``CHARGE``, dans
        le même ordre que l'entrée.

    Raises
    ------
    PredictionError
        Si un contrat n'est pas lisible.
    """
    frame = contracts_frame(records)
    frequency = model.predict_frequency(frame)
    cost = model.predict_cost(frame)
    charge = model.predict_charge(frame)
    return [
        {
            "FREQ": float(frequency[index]),
            "CM": float(cost[index]),
            "CHARGE": float(charge[index]),
        }
        for index in range(len(frame))
    ]


def _one_row(record: object, index: int) -> dict[str, object]:
    """Une ligne. Les clés en trop sont ignorées."""
    if not isinstance(record, dict):
        raise PredictionError(f"Le contrat {index} doit être un objet.")
    missing = [name for name in MODEL_COLUMNS if name not in record]
    if missing:
        listed = ", ".join(missing)
        raise PredictionError(f"Contrat {index} : colonnes absentes : {listed}.")
    row: dict[str, object] = {
        name: _as_category_text(record[name]) for name in CATEGORICAL_COLUMNS
    }
    for name in _NUMERIC_COLUMNS:
        row[name] = _as_finite_number(record[name], name, index)
    exposure = float(row[EXPOSURE_COLUMN])
    # Une durée négative n'est pas un contrat : elle rendrait la prime négative.
    if exposure < 0:
        raise PredictionError(
            f"Contrat {index} : ANNEE_ASSURANCE ne peut pas être négative."
        )
    return row


def _as_category_text(value: object) -> str:
    """Texte de modalité. Un vide devient ``manquant``, comme à l'apprentissage.

    Un flottant sans partie décimale est écrit sans ``.0``. Sinon la
    zone ``1`` envoyée en JSON ne retrouve pas la modalité ``"1"``.
    """
    if isinstance(value, bool):
        raise PredictionError("Une catégorie ne peut pas être un booléen.")
    if value is None:
        return "manquant"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "manquant"
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return "manquant"
    return text


def _as_finite_number(value: object, column: str, index: int) -> float:
    """Nombre fini. Un booléen n'est pas accepté : ``True`` vaudrait 1."""
    if isinstance(value, bool) or value is None:
        raise PredictionError(f"Contrat {index} : {column} doit être un nombre.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionError(
            f"Contrat {index} : {column} doit être un nombre."
        ) from exc
    if not math.isfinite(number):
        raise PredictionError(f"Contrat {index} : {column} doit être un nombre fini.")
    return number
