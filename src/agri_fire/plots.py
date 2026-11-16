"""Graphes d'exploration, fond crème, sans cadre blanc."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Même crème que le thème des présentations (--bg dans portfolio.css).
CREAM = "#f3eee6"
INK = "#1c1610"
MUTED = "#5c564c"
ACCENT = "#8a6a12"

_LABELS = {
    "fr": {
        "claim_share": "Contrats avec un sinistre (%)",
        "zone_count": "Contrats dans la zone",
        "zone_small": "Moins de 500 contrats",
        "zone_large": "500 contrats ou plus",
        "exposure": "Années assurées",
        "contracts": "Contrats",
        "weather_x": "Corrélation des rangs avec la moyenne",
        "weather_sum": "Somme",
        "weather_max": "Maximum",
    },
    "en": {
        "claim_share": "Contracts with a claim (%)",
        "zone_count": "Contracts in the zone",
        "zone_small": "Fewer than 500 contracts",
        "zone_large": "500 contracts or more",
        "exposure": "Years insured",
        "contracts": "Contracts",
        "weather_x": "Rank correlation with the mean",
        "weather_sum": "Sum",
        "weather_max": "Maximum",
    },
}


def save_claim_share_by_year(
    summary: pd.DataFrame,
    path: Path | str,
    *,
    language: str = "fr",
) -> Path:
    """Barres de la part de contrats sinistrés par millésime.

    L'axe part de zéro : les barres restent basses, le millésime ne
    change pas l'ordre de grandeur.

    Parameters
    ----------
    summary : DataFrame
        Sortie de ``summarize_exercise_years``.
    path : Path or str
        Fichier PNG.
    language : str, optional
        ``fr`` ou ``en``, pour le libellé d'axe. Défaut : ``fr``.

    Returns
    -------
    Path
        Chemin écrit.
    """
    _require_columns(summary, ["exercise", "share_with_claim"])
    labels = _labels(language)
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    _style(ax)
    ax.bar(
        summary["exercise"].astype(str),
        summary["share_with_claim"] * 100,
        color=ACCENT,
        width=0.72,
    )
    ax.set_ylim(0, 1.6)
    ax.set_ylabel(labels["claim_share"])
    ax.yaxis.set_major_formatter(_percent_formatter(language))
    ax.tick_params(axis="x", labelrotation=0)
    return _save(fig, path)


def save_year_volume(
    summary: pd.DataFrame,
    path: Path | str,
    *,
    language: str = "fr",
) -> Path:
    """Effectif par millésime, et part sinistrée sur un second axe.

    Parameters
    ----------
    summary : DataFrame
        Sortie de ``summarize_exercise_years``.
    path : Path or str
        Fichier PNG.
    language : str, optional
        ``fr`` ou ``en``. Défaut : ``fr``.

    Returns
    -------
    Path
        Chemin écrit.
    """
    _require_columns(summary, ["exercise", "n_contracts", "share_with_claim"])
    labels = _labels(language)
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    _style(ax)
    positions = range(len(summary))
    ax.bar(positions, summary["n_contracts"], color=ACCENT, width=0.72)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(summary["exercise"].astype(str))
    ax.set_ylabel(labels["contracts"])
    ax.yaxis.set_major_formatter(_count_formatter(language))
    twin = ax.twinx()
    twin.plot(
        list(positions),
        summary["share_with_claim"] * 100,
        color=INK,
        marker="o",
        linewidth=2,
    )
    twin.set_ylim(0, 1.6)
    twin.set_ylabel(labels["claim_share"], color=INK)
    twin.tick_params(colors=INK, labelsize=12)
    twin.yaxis.set_major_formatter(_percent_formatter(language))
    twin.set_facecolor(CREAM)
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_visible(False)
    twin.spines["bottom"].set_visible(False)
    twin.spines["right"].set_color(MUTED)
    return _save(fig, path)


def save_zone_rates(
    summary: pd.DataFrame,
    path: Path | str,
    *,
    language: str = "fr",
    small_below: int = 500,
) -> Path:
    """Taux de sinistre selon l'effectif de la zone.

    Les zones sous ``small_below`` contrats sont en gris : leur taux
    bouge avec quelques contrats.

    Parameters
    ----------
    summary : DataFrame
        Sortie de ``summarize_zones``.
    path : Path or str
        Fichier PNG.
    language : str, optional
        ``fr`` ou ``en``. Défaut : ``fr``.
    small_below : int, optional
        Seuil d'effectif en dessous duquel le point est gris.
        Défaut : 500.

    Returns
    -------
    Path
        Chemin écrit.
    """
    _require_columns(summary, ["n_contracts", "claim_rate"])
    labels = _labels(language)
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    _style(ax)
    small = summary["n_contracts"] < small_below
    ax.scatter(
        summary.loc[~small, "n_contracts"],
        summary.loc[~small, "claim_rate"] * 100,
        s=42,
        color=ACCENT,
        label=labels["zone_large"],
        zorder=3,
    )
    ax.scatter(
        summary.loc[small, "n_contracts"],
        summary.loc[small, "claim_rate"] * 100,
        s=42,
        color=MUTED,
        label=labels["zone_small"],
        zorder=3,
    )
    ax.set_ylim(0, max(2.8, float((summary["claim_rate"] * 100).max()) * 1.15))
    ax.set_xlabel(labels["zone_count"])
    ax.set_ylabel(labels["claim_share"])
    ax.yaxis.set_major_formatter(_percent_formatter(language))
    ax.xaxis.set_major_formatter(_count_formatter(language))
    legend = ax.legend(frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)
    return _save(fig, path)


def save_exposure_histogram(
    exposure: pd.Series,
    path: Path | str,
    *,
    language: str = "fr",
) -> Path:
    """Histogramme de l'exposition, de 0 à 2 ans.

    Parameters
    ----------
    exposure : Series
        ``ANNEE_ASSURANCE``, une valeur par contrat.
    path : Path or str
        Fichier PNG.
    language : str, optional
        ``fr`` ou ``en``. Défaut : ``fr``.

    Returns
    -------
    Path
        Chemin écrit.
    """
    labels = _labels(language)
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    _style(ax)
    ax.hist(
        exposure,
        bins=40,
        range=(0, 2),
        color=ACCENT,
        edgecolor=CREAM,
        linewidth=0.4,
    )
    ax.set_xlim(0, 2)
    ax.set_xlabel(labels["exposure"])
    ax.set_ylabel(labels["contracts"])
    ax.yaxis.set_major_formatter(_count_formatter(language))
    ax.xaxis.set_major_formatter(_percent_formatter(language, decimals=1))
    return _save(fig, path)


def save_weather_rank_strip(
    agreement: pd.DataFrame,
    path: Path | str,
    *,
    language: str = "fr",
) -> Path:
    """Rangs : la somme contre la moyenne, et le maximum contre la moyenne.

    Parameters
    ----------
    agreement : DataFrame
        Sortie de ``weather_rank_agreement``.
    path : Path or str
        Fichier PNG.
    language : str, optional
        ``fr`` ou ``en``. Défaut : ``fr``.

    Returns
    -------
    Path
        Chemin écrit.
    """
    _require_columns(agreement, ["corr_mean_sum", "corr_mean_max"])
    labels = _labels(language)
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    _style(ax)
    sums = agreement["corr_mean_sum"].dropna()
    maxima = agreement["corr_mean_max"].dropna()
    ax.scatter(
        sums,
        [1] * len(sums),
        s=70,
        color=ACCENT,
        zorder=3,
        label=labels["weather_sum"],
    )
    ax.scatter(
        maxima,
        [0] * len(maxima),
        s=36,
        color=INK,
        alpha=0.85,
        zorder=3,
        label=labels["weather_max"],
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([labels["weather_max"], labels["weather_sum"]])
    ax.set_xlabel(labels["weather_x"])
    ax.xaxis.set_major_formatter(_percent_formatter(language, decimals=1))
    if len(sums):
        ax.annotate(
            f"{len(sums)}/{len(sums)}",
            xy=(float(sums.min()), 1),
            xytext=(8, 12),
            textcoords="offset points",
            color=INK,
            fontsize=13,
        )
    return _save(fig, path)


def _labels(language: str) -> dict[str, str]:
    """Libellés d'un graphe. ``fr`` et ``en`` seulement."""
    if language not in _LABELS:
        raise ValueError("language doit être 'fr' ou 'en'.")
    return _LABELS[language]


def _style(ax: plt.Axes) -> None:
    """Fond crème, axes encrés, pas de cadre supérieur."""
    ax.set_facecolor(CREAM)
    ax.figure.set_facecolor(CREAM)
    ax.tick_params(colors=INK, labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)


def _percent_formatter(language: str, decimals: int = 1) -> FuncFormatter:
    """Nombres à virgule en français, à point en anglais."""

    def _format(value: float, _position: object) -> str:
        text = f"{value:.{decimals}f}"
        if language == "fr":
            return text.replace(".", ",")
        return text

    return FuncFormatter(_format)


def _count_formatter(language: str) -> FuncFormatter:
    """Séparateur de milliers : espace en français, virgule en anglais."""

    def _format(value: float, _position: object) -> str:
        text = f"{int(round(value)):,}"
        if language == "fr":
            return text.replace(",", " ")
        return text

    return FuncFormatter(_format)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Lève si le tableau de graphe n'a pas les colonnes prévues."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError("Colonnes absentes : " + ", ".join(missing) + ".")


def _save(fig: plt.Figure, path: Path | str) -> Path:
    """Écrit le PNG sur fond crème et ferme la figure."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        destination,
        dpi=160,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return destination
