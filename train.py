"""Entraîne la ridge retenue et l'enregistre.

À lancer depuis n'importe quel dossier :

    python train.py

Lit les deux CSV d'apprentissage dans ``data/raw/``. N'ouvre pas le
fichier de test. N'ajuste pas les pénalités : elles viennent du hold-out
déjà fait. Écrit ``models/ridge.pkl``.
"""

import argparse
from pathlib import Path

from agri_fire.models.artifact import save_model, train_retained_files

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "data" / "raw" / "train_input_Z61KlZo.csv"
TARGETS = ROOT / "data" / "raw" / "train_output_DzPxaPY.csv"
ARTIFACT = ROOT / "models" / "ridge.pkl"


def main(argv: list[str] | None = None) -> Path:
    """Ajuste la ridge retenue sur tout le train et écrit l'artefact.

    Parameters
    ----------
    argv : list of str, optional
        Arguments de la ligne de commande. ``None`` lit ``sys.argv``.

    Returns
    -------
    Path
        Fichier pickle écrit.
    """
    parser = argparse.ArgumentParser(
        description="Entraîne la ridge retenue et enregistre l'artefact."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=TRAIN,
        help="CSV des variables d'apprentissage.",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=TARGETS,
        help="CSV des cibles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT,
        help="Fichier pickle à écrire.",
    )
    args = parser.parse_args(argv)
    # Tout le train : le hold-out a déjà choisi les pénalités.
    model = train_retained_files(args.train, args.targets)
    path = save_model(model, args.output)
    print(f"ridge enregistrée : {path}", flush=True)
    print(
        f"lignes fréquence {model.n_frequency_rows}, "
        f"coût {model.n_cost_rows}, "
        f"coûts négatifs exclus {model.n_negative_cost_excluded}",
        flush=True,
    )
    print(
        f"pénalité fréquence {model.settings['alpha_frequency']:g}, "
        f"coût {model.settings['alpha_cost']:g}",
        flush=True,
    )
    return path


if __name__ == "__main__":
    main()
