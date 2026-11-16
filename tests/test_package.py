"""Vérifie que le package renommé s'importe."""

import agri_fire
from agri_fire.cli import main


def test_package_imports() -> None:
    """Le package agri_fire est importable."""
    assert agri_fire.__name__ == "agri_fire"


def test_cli_entrypoint_is_callable() -> None:
    """Le point d'entrée en ligne de commande est une fonction."""
    assert callable(main)
