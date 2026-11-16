"""Le service renvoie FREQ, CM et CHARGE à partir des colonnes du modèle.

Le CSV du challenge n'est pas lu. La ridge est ajustée sur un petit
tableau fabriqué, puis servie par le client de test de Flask.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_fire.models.artifact import save_model
from agri_fire.models.selection import fit_ridge
from agri_fire.service import PredictionError, contracts_frame, predict_contracts
from predict import create_app

ROOT = Path(__file__).resolve().parents[1]

_COLUMNS = [
    "ACTIVIT2",
    "VOCATION",
    "ZONE",
    "ANNEE_ASSURANCE",
    "surface_sum",
    "capital_amount",
    "capital_flags",
    "FREQ",
    "CM",
    "CHARGE",
]


def _frame() -> pd.DataFrame:
    """Quatre contrats, dont deux sinistres à coût positif ou nul."""
    return pd.DataFrame(
        [
            ["ACT1", "VOC1", "1", 1.0, 10.0, 100.0, 1.0, 0.0, 0.0, 0.0],
            ["ACT1", "VOC1", "1", 0.5, 20.0, 50.0, 0.0, 2.0, 0.0, 0.0],
            ["ACT5", "VOC6", "2", 1.0, 30.0, 400.0, 3.0, 1.0, 10.0, 10.0],
            ["ACT5", "VOC6", "2", 2.0, 5.0, 80.0, 2.0, 1.0, 30.0, 60.0],
        ],
        columns=_COLUMNS,
    )


def _contract(**overrides: object) -> dict:
    """Un contrat complet. Les clés passées en argument remplacent le reste."""
    record = {
        "ACTIVIT2": "ACT1",
        "VOCATION": "VOC1",
        "ZONE": "1",
        "ANNEE_ASSURANCE": 1.0,
        "surface_sum": 10.0,
        "capital_amount": 100.0,
        "capital_flags": 1.0,
    }
    record.update(overrides)
    return record


def _fitted(tmp_path: Path):
    """Ridge aux pénalités retenues, écrite dans un pickle temporaire."""
    model = fit_ridge(_frame(), 10_000.0, 100.0)
    path = tmp_path / "ridge.pkl"
    save_model(model, path)
    return model, path


def test_predict_route_returns_frequency_cost_and_charge(tmp_path: Path) -> None:
    """POST /predict renvoie les trois nombres, égaux à ceux du modèle."""
    model, path = _fitted(tmp_path)
    payload = _contract(ZONE=1, ANNEE_ASSURANCE=1.5, surface_sum=12)
    response = create_app(path).test_client().post("/predict", json=payload)

    assert response.status_code == 200
    body = response.get_json()
    assert list(body) == ["FREQ", "CM", "CHARGE"]
    frame = contracts_frame([payload])
    assert body["FREQ"] == pytest.approx(float(model.predict_frequency(frame)[0]))
    assert body["CM"] == pytest.approx(float(model.predict_cost(frame)[0]))
    assert body["CHARGE"] == pytest.approx(float(model.predict_charge(frame)[0]))
    assert body["CHARGE"] == pytest.approx(body["FREQ"] * body["CM"] * 1.5)


def test_zone_as_number_matches_zone_as_text(tmp_path: Path) -> None:
    """La zone 1 et la zone « 1 » passent par la même modalité."""
    _model, path = _fitted(tmp_path)
    client = create_app(path).test_client()
    as_number = client.post("/predict", json=_contract(ZONE=1)).get_json()
    as_text = client.post("/predict", json=_contract(ZONE="1")).get_json()
    assert as_number == as_text


def test_a_list_of_contracts_comes_back_in_the_same_order(tmp_path: Path) -> None:
    """Une liste en entrée donne une liste, contrat par contrat."""
    _model, path = _fitted(tmp_path)
    first = _contract(ANNEE_ASSURANCE=1.0)
    second = _contract(ACTIVIT2="ACT5", VOCATION="VOC6", ZONE="2", ANNEE_ASSURANCE=2.0)
    response = create_app(path).test_client().post("/predict", json=[first, second])

    assert response.status_code == 200
    body = response.get_json()
    direct = predict_contracts(fit_ridge(_frame(), 10_000.0, 100.0), [first, second])
    assert len(body) == 2
    assert body[0]["CHARGE"] == pytest.approx(direct[0]["CHARGE"])
    assert body[1]["CHARGE"] == pytest.approx(direct[1]["CHARGE"])
    assert body[0]["CHARGE"] != pytest.approx(body[1]["CHARGE"])


def test_missing_column_is_rejected(tmp_path: Path) -> None:
    """Une colonne du modèle absente ne produit pas de prime."""
    _model, path = _fitted(tmp_path)
    payload = _contract()
    del payload["surface_sum"]
    response = create_app(path).test_client().post("/predict", json=payload)

    assert response.status_code == 400
    assert "surface_sum" in response.get_json()["error"]


def test_negative_exposure_is_rejected() -> None:
    """Une durée assurée négative n'est pas un contrat."""
    with pytest.raises(PredictionError):
        contracts_frame([_contract(ANNEE_ASSURANCE=-0.2)])


def test_unknown_activity_still_returns_a_finite_charge(tmp_path: Path) -> None:
    """Une modalité absente de l'apprentissage retombe sur des zéros."""
    _model, path = _fitted(tmp_path)
    response = create_app(path).test_client().post(
        "/predict", json=_contract(ACTIVIT2="INCONNUE")
    )

    assert response.status_code == 200
    charge = response.get_json()["CHARGE"]
    assert np.isfinite(charge)


def test_predict_script_uses_the_package() -> None:
    """Le script charge l'artefact et appelle le package. Pas une seconde formule."""
    source = (ROOT / "predict.py").read_text(encoding="utf-8")
    assert "load_model" in source
    assert "predict_contracts" in source
    assert "Flask" in source
    assert "Ridge(" not in source
    assert "9696" in source


def test_dockerfile_runs_the_service() -> None:
    """L'image copie l'artefact et lance predict.py."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "predict.py" in text
    assert "ridge.pkl" in text
    assert "9696" in text
