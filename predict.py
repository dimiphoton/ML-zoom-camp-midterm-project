"""Sert la ridge retenue derrière une route Flask.

Charge ``models/ridge.pkl``. Une requête ``POST /predict`` envoie les
sept colonnes du modèle et reçoit ``FREQ``, ``CM`` et ``CHARGE``.

À lancer depuis la racine du dépôt, une fois l'artefact écrit :

    python predict.py

En local, le service écoute sur le port 9696. Render envoie le trafic
sur la variable d'environnement ``PORT`` (souvent 10000). ``MODEL_PATH``
remplace le chemin du pickle si besoin.
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
DEFAULT_PORT = 9696
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
    """Charge l'artefact et écoute sur ``PORT``, ou 9696 en local."""
    path = _resolve_model_path(None)
    if not path.is_file():
        raise SystemExit(
            f"Artefact introuvable : {path}. Lancez python train.py avant le service."
        )
    application = create_app(path)
    application.run(host=HOST, port=listen_port(), debug=False, use_reloader=False)


def listen_port() -> int:
    """Port d'écoute : ``PORT`` si l'hébergeur le fournit, sinon 9696.

    Returns
    -------
    int
        Port entre 1 et 65535.

    Raises
    ------
    SystemExit
        ``PORT`` n'est pas un entier de cet intervalle.
    """
    raw = os.environ.get("PORT")
    if raw is None or raw.strip() == "":
        return DEFAULT_PORT
    try:
        port = int(raw.strip())
    except ValueError as exc:
        raise SystemExit(f"PORT invalide : {raw}") from exc
    if not 1 <= port <= 65535:
        raise SystemExit(f"PORT invalide : {raw}")
    return port


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
