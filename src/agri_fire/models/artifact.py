"""Enregistre la ridge retenue, sans refaire le choix du modèle.

Les pénalités (10 000 sur la fréquence, 100 sur le coût) sont celles du
hold-out documenté dans ``docs/decisions.md``. L'ajustement passe par
``fit_ridge`` : même encodage, même exclusion des coûts négatifs, même
produit pour la charge. ``train.py`` appelle ce module, il ne réécrit
pas la formule.

L'artefact est ajusté sur toutes les lignes d'apprentissage. La partie
tenue à part a déjà servi à choisir ; elle n'est pas retirée ici.
"""

import pickle
from pathlib import Path

import pandas as pd

from agri_fire.features.baseline import assemble_baseline_frame, read_baseline_inputs
from agri_fire.models.selection import SelectionError, TwoPartModel, fit_ridge

# Pénalités figées après la comparaison du 2026-11-16.
RETAINED_ALPHA_FREQUENCY = 10_000.0
RETAINED_ALPHA_COST = 100.0


def fit_retained_model(frame: pd.DataFrame) -> TwoPartModel:
    """Ajuste la ridge retenue sur le tableau déjà assemblé.

    Parameters
    ----------
    frame : DataFrame
        Toutes les lignes à apprendre, au format de
        ``assemble_baseline_frame``. Pas de hold-out : les pénalités
        sont déjà choisies.

    Returns
    -------
    TwoPartModel
        Ridge fréquence et ridge coût.

    Raises
    ------
    SelectionError
        Si ``FREQ`` est négatif ou s'il n'y a aucune ligne de coût.
    """
    return fit_ridge(frame, RETAINED_ALPHA_FREQUENCY, RETAINED_ALPHA_COST)


def train_retained_files(
    train_path: Path | str,
    target_path: Path | str,
) -> TwoPartModel:
    """Lit les deux CSV d'apprentissage et ajuste la ridge retenue.

    Le fichier de test du challenge n'est pas ouvert.

    Parameters
    ----------
    train_path : Path or str
        CSV des variables d'apprentissage.
    target_path : Path or str
        CSV des cibles.

    Returns
    -------
    TwoPartModel
        Ridge ajustée sur toutes les lignes reliées par ``ID``.
    """
    features, targets = read_baseline_inputs(train_path, target_path)
    frame = assemble_baseline_frame(features, targets)
    return fit_retained_model(frame)


def save_model(model: TwoPartModel, path: Path | str) -> Path:
    """Écrit le modèle dans un fichier pickle.

    Parameters
    ----------
    model : TwoPartModel
        Ridge déjà ajustée, aux pénalités retenues.
    path : Path or str
        Destination. Le dossier parent est créé s'il manque.

    Returns
    -------
    Path
        Chemin écrit.

    Raises
    ------
    SelectionError
        Si le modèle n'est pas la ridge aux pénalités retenues.
    """
    _require_retained(model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Fichier local, relu par load_model. Pas un format d'échange.
    with destination.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def load_model(path: Path | str) -> TwoPartModel:
    """Recharge l'artefact écrit par ``save_model``.

    Parameters
    ----------
    path : Path or str
        Fichier pickle.

    Returns
    -------
    TwoPartModel
        Ridge prête pour ``predict_charge``.

    Raises
    ------
    SelectionError
        Si le fichier ne contient pas la ridge retenue.
    """
    with Path(path).open("rb") as handle:
        model = pickle.load(handle)
    _require_retained(model)
    return model


def _require_retained(model: object) -> None:
    """Refuse un autre objet que la ridge aux pénalités figées."""
    if not isinstance(model, TwoPartModel) or model.kind != "ridge":
        raise SelectionError("L'artefact attendu est la régression ridge retenue.")
    frequency = model.settings.get("alpha_frequency")
    cost = model.settings.get("alpha_cost")
    if frequency != RETAINED_ALPHA_FREQUENCY or cost != RETAINED_ALPHA_COST:
        raise SelectionError(
            "Les pénalités de l'artefact ne sont pas celles retenues "
            f"({RETAINED_ALPHA_FREQUENCY:g} et {RETAINED_ALPHA_COST:g})."
        )
