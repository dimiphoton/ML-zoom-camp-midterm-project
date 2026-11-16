"""L'artefact de la ridge retenue se recharge et prédit une charge.

Les CSV du challenge ne sont pas lus ici. Le script est lancé sur un
petit jeu fabriqué.
"""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_fire.features.baseline import assemble_baseline_frame, baseline_source_columns
from agri_fire.models.artifact import (
    RETAINED_ALPHA_COST,
    RETAINED_ALPHA_FREQUENCY,
    load_model,
)
from agri_fire.models.selection import SelectionError

ROOT = Path(__file__).resolve().parents[1]


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


def test_train_script_saves_an_artifact_that_returns_charge(tmp_path: Path) -> None:
    """train.py écrit un pickle. Rechargé, il renvoie une CHARGE par ligne."""
    features, targets = _paired()
    train_path = tmp_path / "train.csv"
    target_path = tmp_path / "cibles.csv"
    output = tmp_path / "ridge.pkl"
    features.to_csv(train_path, index=False)
    targets.to_csv(target_path, index=False)

    # Le pipe Windows n'est pas en UTF-8 tout seul : on force les deux bouts.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train.py"),
            "--train",
            str(train_path),
            "--targets",
            str(target_path),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )

    assert output.is_file()
    assert "ridge enregistrée" in completed.stdout
    loaded = load_model(output)
    frame = assemble_baseline_frame(features, targets)
    charge = loaded.predict_charge(frame)

    assert loaded.kind == "ridge"
    assert loaded.settings["alpha_frequency"] == RETAINED_ALPHA_FREQUENCY
    assert loaded.settings["alpha_cost"] == RETAINED_ALPHA_COST
    assert loaded.n_frequency_rows == len(frame)
    assert loaded.n_cost_rows == 3
    assert loaded.n_negative_cost_excluded == 1
    assert charge.shape == (len(frame),)
    assert np.isfinite(charge).all()
    exposure = frame["ANNEE_ASSURANCE"].to_numpy(dtype=float)
    np.testing.assert_allclose(
        charge,
        loaded.predict_frequency(frame) * loaded.predict_cost(frame) * exposure,
    )


def test_load_model_rejects_another_object(tmp_path: Path) -> None:
    """Un pickle qui n'est pas la ridge retenue ne se charge pas."""
    path = tmp_path / "autre.pkl"
    with path.open("wb") as handle:
        pickle.dump({"kind": "ridge"}, handle)

    with pytest.raises(SelectionError):
        load_model(path)


def test_train_script_uses_the_package_logic() -> None:
    """Le script appelle le package. Il ne réécrit pas la ridge."""
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    assert "train_retained_files" in source
    assert "save_model" in source
    assert "Ridge(" not in source
    assert "OneHotEncoder" not in source
    assert "test_input" not in source
    assert "95_852" not in source
