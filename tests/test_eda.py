"""Exploration sur un petit jeu fabriqué.

Les CSV du challenge ne sont pas lus ici.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from agri_fire.eda import (
    EdaError,
    claim_absence,
    constant_columns,
    duplicate_pairs,
    eda_columns,
    exposure_gap,
    exposure_profile,
    matching_nonzero_support,
    summarize_exercise_years,
    summarize_zones,
    weather_rank_agreement,
)
from agri_fire.plots import CREAM, save_claim_share_by_year


def _targets() -> pd.DataFrame:
    """Cinq contrats : un sans sinistre, un coût nul, un coût négatif."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "FREQ": [0.0, 1.0, 2.0, 1.0, 0.0],
            "CM": [0.0, 0.0, 10.0, -4.0, 0.0],
            "ANNEE_ASSURANCE": [1.0, 0.5, 1.0, 2.0, 0.25],
            "CHARGE": [0.0, 0.0, 20.0, -8.0, 0.0],
        }
    )


def _features() -> pd.DataFrame:
    """Même exposition que les cibles, deux millésimes, trois zones."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "AN_EXERC": ["ANNEE2", "ANNEE1", "ANNEE2", "ANNEE10", "ANNEE1"],
            "ZONE": [1, 1, 2, 2, 3],
            "ANNEE_ASSURANCE": [1.0, 0.5, 1.0, 2.0, 0.25],
        }
    )


def test_eda_columns_keeps_the_blocks_only() -> None:
    """Identifiant, zone, capitaux, surfaces et météo. Pas l'activité."""
    names = [
        "ID",
        "ACTIVIT2",
        "KAPITAL1",
        "SURFACE2",
        "NBJTX25_MM_A",
        "NBJTX25_MMAX_A",
        "ZONE",
        "AN_EXERC",
        "ANNEE_ASSURANCE",
        "FREQ",
    ]
    assert eda_columns(names) == [
        "ID",
        "KAPITAL1",
        "SURFACE2",
        "NBJTX25_MM_A",
        "NBJTX25_MMAX_A",
        "ZONE",
        "AN_EXERC",
        "ANNEE_ASSURANCE",
    ]


def test_claim_absence_counts_zeros_and_negatives() -> None:
    """FREQ nulle, charge nulle malgré un sinistre, charge négative."""
    result = claim_absence(_targets())

    assert result.n_rows == 5
    assert result.n_without_claim == 2
    assert result.share_without_claim == pytest.approx(0.4)
    assert result.n_with_claim == 3
    assert result.n_negative_frequency == 0
    assert result.n_zero_charge == 3
    assert result.n_negative_charge == 1
    assert result.n_zero_cost_with_claim == 1


def test_exposure_profile_splits_partial_full_and_longer() -> None:
    """Parts d'exposition, et sinistres par année couverte."""
    result = exposure_profile(_targets())

    assert result.share_partial == pytest.approx(2 / 5)
    assert result.share_full_year == pytest.approx(2 / 5)
    assert result.share_longer == pytest.approx(1 / 5)
    assert result.total_exposure == pytest.approx(4.75)
    # 0 + 0,5 + 2 + 2 + 0 = 4,5 sinistres
    assert result.total_claim_count == pytest.approx(4.5)
    assert result.claims_per_exposure_year == pytest.approx(4.5 / 4.75)
    assert result.min_exposure == pytest.approx(0.25)
    assert result.max_exposure == pytest.approx(2.0)


def test_exposure_gap_is_zero_when_the_copies_match() -> None:
    """Les deux fichiers portent la même exposition."""
    assert exposure_gap(_features(), _targets()) == 0.0


def test_exposure_gap_measures_a_difference() -> None:
    """Un écart d'un an sur une ligne remonte dans le maximum."""
    features = _features()
    features.loc[0, "ANNEE_ASSURANCE"] = 2.0
    assert exposure_gap(features, _targets()) == pytest.approx(1.0)


def test_years_follow_the_number_in_the_label() -> None:
    """ANNEE10 vient après ANNEE2, et le taux se calcule dans le groupe."""
    summary = summarize_exercise_years(_features(), _targets())

    assert summary["exercise"].tolist() == ["ANNEE1", "ANNEE2", "ANNEE10"]
    annee1 = summary.loc[summary["exercise"].eq("ANNEE1")].iloc[0]
    assert annee1["n_contracts"] == 2
    assert annee1["n_with_claim"] == 1
    assert annee1["share_with_claim"] == pytest.approx(0.5)
    assert annee1["exposure"] == pytest.approx(0.75)


def test_zones_count_contracts_and_claim_rate() -> None:
    """Trois zones, le taux est la part de FREQ positive."""
    summary = summarize_zones(_features(), _targets())

    assert summary["zone"].tolist() == [1, 2, 3]
    assert summary["n_contracts"].sum() == 5
    zone1 = summary.loc[summary["zone"].eq(1)].iloc[0]
    assert zone1["n_with_claim"] == 1
    assert zone1["claim_rate"] == pytest.approx(0.5)


def test_duplicate_id_is_refused() -> None:
    """Un ID répété casserait les effectifs."""
    features = _features()
    features.loc[1, "ID"] = 1

    with pytest.raises(EdaError, match="ID en double"):
        summarize_zones(features, _targets())


def test_missing_column_is_refused() -> None:
    """Une cible absente ne produit pas un taux silencieux."""
    with pytest.raises(EdaError, match="FREQ"):
        claim_absence(pd.DataFrame({"CM": [0.0], "CHARGE": [0.0]}))


def test_duplicate_pairs_keep_copies_and_drop_zero_columns() -> None:
    """Une copie active est retenue. Deux colonnes de zéros ne le sont pas."""
    frame = pd.DataFrame(
        {
            "SURFACE1": [10.0, 20.0, 30.0, 40.0],
            "SURFACE2": [10.0, 20.0, 30.0, 40.0],
            "SURFACE3": [10.0, 20.0, 0.0, 1.0],
            "SURFACE14": [0.0, 0.0, 0.0, 0.0],
            "SURFACE15": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pairs = duplicate_pairs(frame, list(frame.columns))

    assert pairs[["left", "right"]].values.tolist() == [["SURFACE1", "SURFACE2"]]
    constants = constant_columns(frame, ["SURFACE14", "SURFACE15", "SURFACE1"])
    assert constants["column"].tolist() == ["SURFACE14", "SURFACE15"]
    assert constants["value"].tolist() == [0.0, 0.0]


def test_matching_support_links_a_flag_to_its_amount() -> None:
    """L'indicateur et le montant s'allument sur les mêmes lignes."""
    frame = pd.DataFrame(
        {
            "KAPITAL1": [0, 1, 0, 1],
            "KAPITAL13": [0, 5000, 0, 10000],
            "KAPITAL2": [0, 0, 1, 0],
        }
    )
    pairs = matching_nonzero_support(frame, list(frame.columns))

    assert pairs[["left", "right"]].values.tolist() == [["KAPITAL1", "KAPITAL13"]]
    assert pairs.loc[0, "n_nonzero"] == 2


def test_weather_sum_repeats_the_mean_rank() -> None:
    """Même rang pour la somme et la moyenne, pas pour le maximum."""
    frame = pd.DataFrame(
        {
            "NBJTX25_MM_A": ["01. <= 4", "02. <= 5", "03. <= 7", "04. >= 7"],
            "NBJTX25_MMAX_A": ["04. >= 23", "01. <= 12", "02. <= 18", "03. <= 23"],
            "NBJTX25_MSOM_A": ["01. <= 44", "02. <= 65", "03. <= 88", "04. >= 88"],
            "ACTIVIT2": ["a", "b", "c", "d"],
        }
    )
    agreement = weather_rank_agreement(frame)

    assert agreement["stem"].tolist() == ["NBJTX25"]
    row = agreement.iloc[0]
    assert row["n_compared"] == 4
    assert row["corr_mean_sum"] == pytest.approx(1.0)
    assert row["same_rank_mean_sum"] == pytest.approx(1.0)
    assert row["corr_mean_max"] < 0.5


def test_claim_share_chart_is_cream_not_white(tmp_path: Path) -> None:
    """Le PNG a le fond crème du thème, pas un cadre blanc."""
    summary = summarize_exercise_years(_features(), _targets())
    path = save_claim_share_by_year(summary, tmp_path / "annees.png", language="fr")

    image = plt.imread(path)
    red, green, blue = (int(channel, 16) / 255 for channel in (CREAM[1:3], CREAM[3:5], CREAM[5:7]))
    corner = image[0, 0, :3]
    assert corner[0] == pytest.approx(red, abs=0.03)
    assert corner[1] == pytest.approx(green, abs=0.03)
    assert corner[2] == pytest.approx(blue, abs=0.03)
    # Le blanc pur serait (1, 1, 1). Le coin reste en dessous.
    assert float(corner.sum()) < 2.95
