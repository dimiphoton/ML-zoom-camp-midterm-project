"""Jeu court de variables pour la baseline fréquence × coût."""

from pathlib import Path

import pandas as pd

# Copies connues : les compter deux fois ne fait que gonfler la somme.
SURFACE_COPY_COLUMNS = ("SURFACE2", "SURFACE6")
SURFACE_COLUMNS = tuple(
    f"SURFACE{index}"
    for index in range(1, 22)
    if f"SURFACE{index}" not in SURFACE_COPY_COLUMNS
)

# KAPITAL1 à 9 sont des indicateurs 0/1. KAPITAL34 à 43 sont O/N.
CAPITAL_FLAG_COLUMNS = tuple(
    f"KAPITAL{index}" for index in (*range(1, 10), *range(34, 44))
)
# KAPITAL10 recopie KAPITAL11. Les montants utiles vont de 11 à 33.
CAPITAL_AMOUNT_COLUMNS = tuple(f"KAPITAL{index}" for index in range(11, 34))

CATEGORICAL_COLUMNS = ("ACTIVIT2", "VOCATION", "ZONE")
EXPOSURE_COLUMN = "ANNEE_ASSURANCE"
AGGREGATE_COLUMNS = ("surface_sum", "capital_amount", "capital_flags")
MODEL_COLUMNS = CATEGORICAL_COLUMNS + (EXPOSURE_COLUMN,) + AGGREGATE_COLUMNS
TARGET_COLUMNS = ("FREQ", "CM", "CHARGE")


class BaselineError(ValueError):
    """Le tableau de la baseline n'a pas le grain ou les colonnes attendus."""


def baseline_source_columns() -> tuple[str, ...]:
    """Colonnes lues dans le fichier de variables, pas les 374.

    Returns
    -------
    tuple of str
        Identifiant, activité, vocation, zone, exposition, surfaces
        et capitaux qui entrent dans les agrégats.
    """
    return (
        ("ID",)
        + CATEGORICAL_COLUMNS
        + (EXPOSURE_COLUMN,)
        + SURFACE_COLUMNS
        + SURFACE_COPY_COLUMNS
        + CAPITAL_AMOUNT_COLUMNS
        + ("KAPITAL10",)
        + CAPITAL_FLAG_COLUMNS
    )


def assemble_baseline_frame(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Joint les variables utiles et les cibles, puis agrège surfaces et capitaux.

    ``SURFACE2`` et ``SURFACE6`` sont lus pour vérifier leur présence, mais
    ils ne sont pas additionnés : ce sont des copies de ``SURFACE1`` et
    ``SURFACE4``. ``KAPITAL10`` est la copie de ``KAPITAL11``. Un manquant
    dans une surface ou un capital compte comme zéro. Le code ``7000+``
    est le haut de classe, lu comme 7 000. ``O`` sur un indicateur vaut 1,
    ``N`` vaut 0.

    Parameters
    ----------
    features : DataFrame
        Une ligne par contrat, avec les colonnes de
        ``baseline_source_columns``.
    targets : DataFrame
        Cibles ``ID``, ``FREQ``, ``CM`` et ``CHARGE``, une ligne par
        contrat.

    Returns
    -------
    DataFrame
        Une ligne par contrat, avec les sept variables du modèle et les
        trois cibles.

    Raises
    ------
    BaselineError
        Si une colonne manque, si un identifiant ne relie pas les deux
        tables, ou si une cible ou l'exposition est manquante.
    """
    _require_columns(features, baseline_source_columns(), "variables")
    _require_columns(targets, ("ID",) + TARGET_COLUMNS, "cibles")
    merged = _merge_on_id(features, targets)

    frame = pd.DataFrame(
        {
            "ID": merged["ID"].to_numpy(),
            "ACTIVIT2": _as_category(merged["ACTIVIT2"]),
            "VOCATION": _as_category(merged["VOCATION"]),
            "ZONE": _as_category(merged["ZONE"]),
            EXPOSURE_COLUMN: merged[EXPOSURE_COLUMN].astype(float),
            "surface_sum": _sum_numbers(merged, SURFACE_COLUMNS),
            "capital_amount": _sum_numbers(merged, CAPITAL_AMOUNT_COLUMNS),
            "capital_flags": _sum_flags(merged, CAPITAL_FLAG_COLUMNS),
            "FREQ": merged["FREQ"].astype(float),
            "CM": merged["CM"].astype(float),
            "CHARGE": merged["CHARGE"].astype(float),
        }
    )
    required = (EXPOSURE_COLUMN,) + TARGET_COLUMNS
    if frame.loc[:, required].isna().any().any():
        raise BaselineError(
            "Exposition ou cible manquante après la jointure sur ID."
        )
    return frame


def read_baseline_inputs(
    train_path: Path | str,
    target_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lit uniquement les colonnes utiles du train et le fichier de cibles.

    Le fichier de test du challenge n'est pas ouvert : il n'a pas de
    cible et ne sert pas à mesurer cette baseline.

    Parameters
    ----------
    train_path : Path or str
        CSV des variables d'apprentissage.
    target_path : Path or str
        CSV des cibles, reliées par ``ID``.

    Returns
    -------
    tuple of DataFrame
        Variables réduites, puis cibles.
    """
    features = pd.read_csv(
        train_path,
        usecols=list(baseline_source_columns()),
        low_memory=False,
    )
    targets = pd.read_csv(target_path)
    return features, targets


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    """Signale les colonnes absentes d'une table."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise BaselineError(
            f"Colonnes absentes des {label} : " + ", ".join(missing) + "."
        )


def _merge_on_id(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Relie variables et cibles un pour un, sans suffixe sur l'exposition."""
    if not features["ID"].is_unique:
        raise BaselineError("ID des variables en double.")
    if not targets["ID"].is_unique:
        raise BaselineError("ID des cibles en double.")
    target_keep = targets.loc[:, ["ID", "FREQ", "CM", "CHARGE"]]
    merged = features.merge(target_keep, on="ID", how="outer", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise BaselineError(
            "Les ID des variables et des cibles ne correspondent pas."
        )
    return merged.drop(columns="_merge")


def _as_category(series: pd.Series) -> pd.Series:
    """Texte stable pour l'encodage. La zone n'est pas une grandeur."""
    return series.astype("string").fillna("manquant")


def _sum_numbers(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Somme des colonnes numériques. Un manquant vaut zéro."""
    parts = [_as_number(frame[name]) for name in columns]
    return pd.concat(parts, axis=1).fillna(0.0).sum(axis=1)


def _sum_flags(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Somme d'indicateurs 0/1 ou O/N. Un manquant vaut zéro."""
    parts = [_as_flag(frame[name]) for name in columns]
    return pd.concat(parts, axis=1).fillna(0.0).sum(axis=1)


def _as_number(series: pd.Series) -> pd.Series:
    """Convertit un montant. Le suffixe ``+`` d'une classe ouverte est ignoré."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype("string").str.strip()
    text = text.str.replace(r"\+$", "", regex=True)
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _as_flag(series: pd.Series) -> pd.Series:
    """Indicateur : 1 si la valeur numérique est 1, ou si le texte est ``O``."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype("string").str.strip().str.upper()
    return text.map({"O": 1.0, "N": 0.0, "1": 1.0, "0": 0.0})
