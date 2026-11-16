"""Exploration du train : zéros, exposition, millésimes, zones, blocs redondants."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

_KEY_COLUMNS = ("ID", "AN_EXERC", "ZONE", "ANNEE_ASSURANCE")
# MMAX et MSOM avant MM : un nom ne doit porter qu'une statistique.
_WEATHER_MARKERS = (("_MMAX_", "max"), ("_MSOM_", "sum"), ("_MM_", "mean"))


class EdaError(ValueError):
    """Une table d'exploration n'a pas le grain attendu."""


@dataclass(frozen=True)
class ClaimAbsence:
    """Rareté des sinistres incendie.

    Attributes
    ----------
    n_rows : int
        Nombre de contrats-exercices.
    n_without_claim : int
        Lignes avec ``FREQ`` égal à 0.
    share_without_claim : float
        Part de ces lignes.
    n_with_claim : int
        Lignes avec ``FREQ`` strictement positif.
    n_negative_frequency : int
        Lignes avec ``FREQ`` négatif.
    n_zero_charge : int
        Lignes dont ``CHARGE`` vaut 0.
    n_negative_charge : int
        Lignes dont ``CHARGE`` est négative.
    n_zero_cost_with_claim : int
        Sinistres (``FREQ`` > 0) dont le coût moyen vaut 0.
    """

    n_rows: int
    n_without_claim: int
    share_without_claim: float
    n_with_claim: int
    n_negative_frequency: int
    n_zero_charge: int
    n_negative_charge: int
    n_zero_cost_with_claim: int


@dataclass(frozen=True)
class ExposureProfile:
    """Exposition ``ANNEE_ASSURANCE`` et sinistres qu'elle porte.

    Attributes
    ----------
    n_rows : int
        Nombre de lignes.
    total_exposure : float
        Somme des années assurées.
    total_claim_count : float
        Somme de ``FREQ × ANNEE_ASSURANCE``, le nombre de sinistres.
    claims_per_exposure_year : float
        Nombre de sinistres par année d'exposition.
    share_partial : float
        Part des expositions strictement inférieures à 1.
    share_full_year : float
        Part des expositions exactement égales à 1.
    share_longer : float
        Part des expositions strictement supérieures à 1.
    min_exposure : float
        Exposition minimale.
    median_exposure : float
        Exposition médiane.
    mean_exposure : float
        Exposition moyenne.
    max_exposure : float
        Exposition maximale.
    """

    n_rows: int
    total_exposure: float
    total_claim_count: float
    claims_per_exposure_year: float
    share_partial: float
    share_full_year: float
    share_longer: float
    min_exposure: float
    median_exposure: float
    mean_exposure: float
    max_exposure: float


def eda_columns(column_names: list[str]) -> list[str]:
    """Sélectionne les colonnes lues pour l'exploration.

    Identifiant, exercice, zone, exposition, puis les blocs de capitaux,
    de surfaces et de météo (moyenne, maximum, somme). Le reste du
    fichier, dont les cibles, n'est pas relu ici.

    Parameters
    ----------
    column_names : list of str
        Noms de colonnes du fichier de variables, dans l'ordre du fichier.

    Returns
    -------
    list of str
        Sous-ensemble, dans le même ordre.
    """
    selected: list[str] = []
    for name in column_names:
        if name in _KEY_COLUMNS or name.startswith(("KAPITAL", "SURFACE")):
            selected.append(name)
        elif _weather_kind(name) is not None:
            selected.append(name)
    return selected


def claim_absence(targets: pd.DataFrame) -> ClaimAbsence:
    """Compte les contrats sans sinistre et les charges nulles ou négatives.

    Parameters
    ----------
    targets : DataFrame
        Cibles, avec ``FREQ``, ``CM`` et ``CHARGE``.

    Returns
    -------
    ClaimAbsence
        Effectifs et part sans sinistre.

    Raises
    ------
    EdaError
        Si une colonne manque ou contient une valeur manquante.
    """
    _require(targets, ["FREQ", "CM", "CHARGE"])
    _reject_missing(targets, ["FREQ", "CM", "CHARGE"])
    frequency = targets["FREQ"]
    charge = targets["CHARGE"]
    n_rows = len(targets)
    n_without = int(frequency.eq(0).sum())
    return ClaimAbsence(
        n_rows=n_rows,
        n_without_claim=n_without,
        share_without_claim=n_without / n_rows if n_rows else float("nan"),
        n_with_claim=int(frequency.gt(0).sum()),
        n_negative_frequency=int(frequency.lt(0).sum()),
        n_zero_charge=int(charge.eq(0).sum()),
        n_negative_charge=int(charge.lt(0).sum()),
        n_zero_cost_with_claim=int((frequency.gt(0) & targets["CM"].eq(0)).sum()),
    )


def exposure_profile(targets: pd.DataFrame) -> ExposureProfile:
    """Résume l'exposition et le nombre de sinistres qu'elle porte.

    Une exposition « pleine » est exactement 1, telle qu'elle est stockée.
    Le nombre de sinistres est ``FREQ × ANNEE_ASSURANCE``.

    Parameters
    ----------
    targets : DataFrame
        Cibles, avec ``FREQ`` et ``ANNEE_ASSURANCE``.

    Returns
    -------
    ExposureProfile
        Sommes, parts et extrêmes.

    Raises
    ------
    EdaError
        Si une colonne manque, si une valeur manque, ou si l'exposition
        totale est nulle.
    """
    _require(targets, ["FREQ", "ANNEE_ASSURANCE"])
    _reject_missing(targets, ["FREQ", "ANNEE_ASSURANCE"])
    exposure = targets["ANNEE_ASSURANCE"]
    total_exposure = float(exposure.sum())
    if total_exposure == 0:
        raise EdaError("L'exposition totale est nulle.")
    claim_count = targets["FREQ"] * exposure
    n_rows = len(targets)
    return ExposureProfile(
        n_rows=n_rows,
        total_exposure=total_exposure,
        total_claim_count=float(claim_count.sum()),
        claims_per_exposure_year=float(claim_count.sum() / total_exposure),
        share_partial=float(exposure.lt(1).mean()),
        share_full_year=float(exposure.eq(1).mean()),
        share_longer=float(exposure.gt(1).mean()),
        min_exposure=float(exposure.min()),
        median_exposure=float(exposure.median()),
        mean_exposure=float(exposure.mean()),
        max_exposure=float(exposure.max()),
    )


def exposure_gap(features: pd.DataFrame, targets: pd.DataFrame) -> float:
    """Écart absolu maximum entre les deux copies de l'exposition.

    ``ANNEE_ASSURANCE`` est dans le fichier de variables et dans les
    cibles. Elles doivent dire la même chose, ligne à ligne, via ``ID``.

    Parameters
    ----------
    features : DataFrame
        Variables, avec ``ID`` et ``ANNEE_ASSURANCE``.
    targets : DataFrame
        Cibles, avec les mêmes colonnes.

    Returns
    -------
    float
        Écart absolu maximum. ``nan`` si une exposition jointe manque.

    Raises
    ------
    EdaError
        Si une colonne manque, ou si ``ID`` n'est pas unique.
    """
    _require(features, ["ID", "ANNEE_ASSURANCE"])
    _require(targets, ["ID", "ANNEE_ASSURANCE"])
    # Renommées avant la jointure : les deux colonnes portent le même nom.
    left = features[["ID", "ANNEE_ASSURANCE"]].rename(
        columns={"ANNEE_ASSURANCE": "exposure_features"}
    )
    right = targets[["ID", "ANNEE_ASSURANCE"]].rename(
        columns={"ANNEE_ASSURANCE": "exposure_targets"}
    )
    merged = _merge_on_id(left, right)
    gap = (merged["exposure_features"] - merged["exposure_targets"]).abs()
    if gap.isna().any():
        return float("nan")
    return float(gap.max())


def summarize_exercise_years(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Effectif, exposition et sinistralité par millésime ``AN_EXERC``.

    L'ordre est celui du numéro dans le libellé (``ANNEE2`` avant
    ``ANNEE10``). Ce n'est pas une chronologie : les libellés sont
    anonymes.

    Parameters
    ----------
    features : DataFrame
        Variables, avec ``ID`` et ``AN_EXERC``.
    targets : DataFrame
        Cibles jointes par ``ID``.

    Returns
    -------
    DataFrame
        Une ligne par millésime : ``exercise``, ``n_contracts``,
        ``exposure``, ``n_without_claim``, ``n_with_claim``,
        ``share_without_claim``, ``share_with_claim``, ``mean_charge``,
        ``total_charge``.

    Raises
    ------
    EdaError
        Si la jointure par ``ID`` n'est pas bijective.
    """
    merged = _join_targets(features, targets, ["AN_EXERC"])
    grouped = merged.groupby("AN_EXERC", sort=False)
    summary = grouped.agg(
        n_contracts=("ID", "size"),
        exposure=("ANNEE_ASSURANCE", "sum"),
        n_without_claim=("FREQ", lambda series: int(series.eq(0).sum())),
        n_with_claim=("FREQ", lambda series: int(series.gt(0).sum())),
        mean_charge=("CHARGE", "mean"),
        total_charge=("CHARGE", "sum"),
    ).reset_index()
    summary = summary.rename(columns={"AN_EXERC": "exercise"})
    summary["share_without_claim"] = (
        summary["n_without_claim"] / summary["n_contracts"]
    )
    summary["share_with_claim"] = summary["n_with_claim"] / summary["n_contracts"]
    summary = summary.sort_values(
        "exercise", key=lambda series: series.map(_exercise_order)
    )
    return summary.reset_index(drop=True)


def summarize_zones(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Effectif et taux de contrats sinistrés par ``ZONE``.

    Parameters
    ----------
    features : DataFrame
        Variables, avec ``ID`` et ``ZONE``.
    targets : DataFrame
        Cibles jointes par ``ID``.

    Returns
    -------
    DataFrame
        Une ligne par zone : ``zone``, ``n_contracts``, ``exposure``,
        ``n_with_claim``, ``claim_rate``, ``mean_charge``.

    Raises
    ------
    EdaError
        Si la jointure par ``ID`` n'est pas bijective.
    """
    merged = _join_targets(features, targets, ["ZONE"])
    grouped = merged.groupby("ZONE", sort=False)
    summary = grouped.agg(
        n_contracts=("ID", "size"),
        exposure=("ANNEE_ASSURANCE", "sum"),
        n_with_claim=("FREQ", lambda series: int(series.gt(0).sum())),
        mean_charge=("CHARGE", "mean"),
    ).reset_index()
    summary = summary.rename(columns={"ZONE": "zone"})
    summary["claim_rate"] = summary["n_with_claim"] / summary["n_contracts"]
    summary = summary.sort_values("zone")
    return summary.reset_index(drop=True)


def constant_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Colonnes numériques qui ne prennent qu'une valeur observée.

    Une colonne entière de zéros est constante : elle n'apporte pas
    une mesure. Les textes qui ne se convertissent pas en nombre sont
    ignorés.

    Parameters
    ----------
    frame : DataFrame
        Table qui contient ``columns``.
    columns : list of str
        Colonnes à examiner.

    Returns
    -------
    DataFrame
        ``column``, ``value``, ``n_observed``, ``n_missing``.
    """
    _require(frame, columns)
    rows: list[dict[str, object]] = []
    for name in columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        observed = values.dropna()
        if observed.empty:
            continue
        unique = observed.unique()
        if len(unique) == 1:
            rows.append(
                {
                    "column": name,
                    "value": float(unique[0]),
                    "n_observed": int(observed.shape[0]),
                    "n_missing": int(values.isna().sum()),
                }
            )
    return pd.DataFrame(rows, columns=["column", "value", "n_observed", "n_missing"])


def duplicate_pairs(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    min_equal_share: float = 0.999,
) -> pd.DataFrame:
    """Paires numériques égales sur presque toutes les valeurs co-observées.

    Deux colonnes constantes ne sont pas un doublon : ce sont des
    colonnes vides, relevées par ``constant_columns``. On exige aussi
    que l'égalité tienne sur les lignes actives (au moins une des deux
    valeurs non nulle), pour ne pas prendre deux colonnes de zéros
    éparses pour la même mesure.

    Parameters
    ----------
    frame : DataFrame
        Table qui contient ``columns``.
    columns : list of str
        Bloc à comparer, par exemple les surfaces.
    min_equal_share : float, optional
        Part minimale de valeurs égales, sur toutes les lignes
        co-observées et sur les lignes actives. Défaut : 0,999.

    Returns
    -------
    DataFrame
        ``left``, ``right``, ``n_both``, ``n_differ``, ``share_equal``,
        ``n_active``, ``share_equal_active``.
    """
    _require(frame, columns)
    numeric = {
        name: pd.to_numeric(frame[name], errors="coerce") for name in columns
    }
    rows: list[dict[str, object]] = []
    for index, left_name in enumerate(columns):
        left = numeric[left_name]
        if left.nunique(dropna=True) <= 1:
            continue
        for right_name in columns[index + 1 :]:
            right = numeric[right_name]
            if right.nunique(dropna=True) <= 1:
                continue
            both = left.notna() & right.notna()
            n_both = int(both.sum())
            if n_both == 0:
                continue
            n_differ = int((left[both].to_numpy() != right[both].to_numpy()).sum())
            share_equal = 1 - n_differ / n_both
            active = both & (left.ne(0) | right.ne(0))
            n_active = int(active.sum())
            if n_active == 0:
                continue
            n_differ_active = int(
                (left[active].to_numpy() != right[active].to_numpy()).sum()
            )
            share_active = 1 - n_differ_active / n_active
            if share_equal >= min_equal_share and share_active >= min_equal_share:
                rows.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "n_both": n_both,
                        "n_differ": n_differ,
                        "share_equal": share_equal,
                        "n_active": n_active,
                        "share_equal_active": share_active,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "left",
            "right",
            "n_both",
            "n_differ",
            "share_equal",
            "n_active",
            "share_equal_active",
        ],
    )


def matching_nonzero_support(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Paires dont les lignes non nulles sont exactement les mêmes.

    Sert à voir un indicateur 0/1 qui répète un montant : les deux
    colonnes s'allument sur les mêmes contrats, sans être égales.

    Parameters
    ----------
    frame : DataFrame
        Table qui contient ``columns``.
    columns : list of str
        Bloc numérique, par exemple les capitaux.

    Returns
    -------
    DataFrame
        ``left``, ``right``, ``n_nonzero``.
    """
    _require(frame, columns)
    positive: dict[str, pd.Series] = {}
    for name in columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        mask = values.notna() & values.ne(0)
        if int(mask.sum()) > 0:
            positive[name] = mask
    names = list(positive)
    rows: list[dict[str, object]] = []
    for index, left_name in enumerate(names):
        left_mask = positive[left_name]
        for right_name in names[index + 1 :]:
            if left_mask.equals(positive[right_name]):
                rows.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "n_nonzero": int(left_mask.sum()),
                    }
                )
    return pd.DataFrame(rows, columns=["left", "right", "n_nonzero"])


def ordinal_rank(series: pd.Series) -> pd.Series:
    """Extrait le rang ``01`` à ``04`` d'un libellé météo.

    Les fichiers codent la moyenne, le maximum et la somme en paliers
    du type ``02. <= 5``. Les seuils changent d'une statistique à
    l'autre : seul le rang initial permet de les comparer.

    Parameters
    ----------
    series : Series
        Libellés, ou valeurs manquantes.

    Returns
    -------
    Series
        Rang numérique. Les textes sans préfixe chiffré deviennent
        manquants.
    """
    text = series.astype("string")
    return pd.to_numeric(text.str.slice(0, 2), errors="coerce")


def weather_rank_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    """Corrélation des rangs entre moyenne, maximum et somme.

    Une somme redondante a le même rang que la moyenne sur chaque
    ligne où les deux sont remplies.

    Parameters
    ----------
    frame : DataFrame
        Table dont certains noms suivent ``<grandeur>_MM_``,
        ``_MMAX_`` ou ``_MSOM_``.

    Returns
    -------
    DataFrame
        Une ligne par grandeur qui a au moins deux statistiques :
        ``stem``, ``n_compared``, ``missing_share``, ``corr_mean_max``,
        ``corr_mean_sum``, ``corr_max_sum``, ``same_rank_mean_sum``.
    """
    groups: dict[str, dict[str, str]] = {}
    for name in frame.columns:
        parsed = _weather_kind(str(name))
        if parsed is None:
            continue
        stem, kind = parsed
        groups.setdefault(stem, {})[kind] = str(name)

    rows: list[dict[str, object]] = []
    for stem, parts in sorted(groups.items()):
        if len(parts) < 2:
            continue
        ranks = {kind: ordinal_rank(frame[name]) for kind, name in parts.items()}
        # La moyenne sert de référence : c'est elle que la somme recopie.
        reference_kind = "mean" if "mean" in ranks else next(iter(ranks))
        reference = ranks[reference_kind]
        compared = reference.notna()
        for rank in ranks.values():
            compared = compared & rank.notna()
        record: dict[str, object] = {
            "stem": stem,
            "n_compared": int(compared.sum()),
            "missing_share": float(reference.isna().mean()),
            "corr_mean_max": _rank_corr(ranks, compared, "mean", "max"),
            "corr_mean_sum": _rank_corr(ranks, compared, "mean", "sum"),
            "corr_max_sum": _rank_corr(ranks, compared, "max", "sum"),
            "same_rank_mean_sum": _same_rank(ranks, compared, "mean", "sum"),
        }
        rows.append(record)
    return pd.DataFrame(
        rows,
        columns=[
            "stem",
            "n_compared",
            "missing_share",
            "corr_mean_max",
            "corr_mean_sum",
            "corr_max_sum",
            "same_rank_mean_sum",
        ],
    )


def _weather_kind(name: str) -> tuple[str, str] | None:
    """Retourne la grandeur et la statistique, ou ``None``."""
    for marker, kind in _WEATHER_MARKERS:
        if marker in name:
            stem, _rest = name.split(marker, 1)
            return stem, kind
    return None


def _rank_corr(
    ranks: dict[str, pd.Series],
    compared: pd.Series,
    left: str,
    right: str,
) -> float:
    """Corrélation de Pearson des rangs, ou ``nan`` si une statistique manque."""
    if left not in ranks or right not in ranks or int(compared.sum()) < 2:
        return float("nan")
    return float(ranks[left][compared].corr(ranks[right][compared]))


def _same_rank(
    ranks: dict[str, pd.Series],
    compared: pd.Series,
    left: str,
    right: str,
) -> float:
    """Part des lignes où les deux rangs sont égaux."""
    if left not in ranks or right not in ranks or int(compared.sum()) == 0:
        return float("nan")
    same = ranks[left][compared].to_numpy() == ranks[right][compared].to_numpy()
    return float(np.mean(same))


def _exercise_order(label: object) -> int:
    """Numéro porté par un libellé ``ANNEE3``. Les autres libellés passent après."""
    digits = "".join(character for character in str(label) if character.isdigit())
    if not digits:
        return 10**9
    return int(digits)


def _join_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Joint les variables demandées aux cibles, une ligne pour une ligne."""
    _require(features, ["ID", *feature_columns])
    _require(targets, ["ID", "FREQ", "CHARGE", "ANNEE_ASSURANCE"])
    _reject_missing(targets, ["FREQ", "CHARGE", "ANNEE_ASSURANCE"])
    return _merge_on_id(features[["ID", *feature_columns]], targets)


def _merge_on_id(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Jointure interne bijective sur ``ID``."""
    if not left["ID"].is_unique or not right["ID"].is_unique:
        raise EdaError("ID en double : la jointure ne doit pas multiplier les lignes.")
    merged = left.merge(right, on="ID", how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise EdaError("Les ID des deux tables ne correspondent pas.")
    return merged


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    """Lève si une colonne demandée est absente."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise EdaError("Colonnes absentes : " + ", ".join(missing) + ".")


def _reject_missing(frame: pd.DataFrame, columns: list[str]) -> None:
    """Lève si une colonne de contrôle contient un manquant."""
    for name in columns:
        if frame[name].isna().any():
            raise EdaError(f"{name} contient des valeurs manquantes.")
