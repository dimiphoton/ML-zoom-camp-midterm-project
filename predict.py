"""Sert la ridge retenue derrière une route Flask.

Charge ``models/ridge.pkl``. Une requête ``POST /predict`` envoie les
sept colonnes du modèle et reçoit ``FREQ``, ``CM`` et ``CHARGE``.

À lancer depuis la racine du dépôt, une fois l'artefact écrit :

    python predict.py

Le service écoute sur le port 9696. ``MODEL_PATH`` remplace le chemin
du pickle si besoin.
"""

import os
from pathlib import Path

from flask import Flask, jsonify, request

from agri_fire.models.artifact import load_model
from agri_fire.service import PredictionError, predict_contracts

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "models" / "ridge.pkl"
# 0.0.0.0 : dans Docker, localhost ne verrait que l'intérieur du conteneur.
HOST = "0.0.0.0"
PORT = 9696
_BODY_ERROR = "Le corps doit être un contrat ou une liste de contrats."


def create_app(model_path: Path | str | None = None) -> Flask:
    """Construit l'application autour de l'artefact déjà ajusté.

    Parameters
    ----------
    model_path : Path or str, optional
        Fichier pickle. Par défaut ``models/ridge.pkl``, ou ``MODEL_PATH``.

    Returns
    -------
    Flask
        Application avec la route ``POST /predict``.
    """
    path = _resolve_model_path(model_path)
    model = load_model(path)
    application = Flask("agri-fire")
    application.json.sort_keys = False

    @application.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        single, records = _as_records(payload)
        if records is None:
            return jsonify(error=_BODY_ERROR), 400
        try:
            predicted = predict_contracts(model, records)
        except PredictionError as exc:
            return jsonify(error=str(exc)), 400
        body = predicted[0] if single else predicted
        return jsonify(body)

    return application


def main() -> None:
    """Charge l'artefact et écoute sur le port 9696."""
    path = _resolve_model_path(None)
    if not path.is_file():
        raise SystemExit(
            f"Artefact introuvable : {path}. Lancez python train.py avant le service."
        )
    application = create_app(path)
    application.run(host=HOST, port=PORT, debug=False, use_reloader=False)


def _resolve_model_path(model_path: Path | str | None) -> Path:
    """Chemin explicite, sinon la variable d'environnement, sinon le défaut."""
    if model_path is not None:
        return Path(model_path)
    override = os.environ.get("MODEL_PATH")
    if override:
        return Path(override)
    return ARTIFACT


def _as_records(payload: object) -> tuple[bool, list[dict] | None]:
    """Un objet JSON est un contrat. Une liste en contient plusieurs."""
    if isinstance(payload, dict):
        return True, [payload]
    if isinstance(payload, list):
        return False, payload
    return False, None


if __name__ == "__main__":
    main()
