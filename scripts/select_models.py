"""Compare la droite, les arbres, la baseline et le plancher.

À lancer depuis n'importe quel dossier :

    python scripts/select_models.py

Lit les deux CSV d'apprentissage dans ``data/raw/``. N'ouvre pas le
fichier de test. Écrit les deux graphes de la comparaison dans
``pictures/presentations/``.
"""

from pathlib import Path

import pandas as pd

from agri_fire.models.selection import evaluate_selection_files, format_selection
from agri_fire.plots import save_calibration_chart, save_error_gap_chart

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "raw" / "train_input_Z61KlZo.csv"
TARGETS = ROOT / "data" / "raw" / "train_output_DzPxaPY.csv"
PICTURES = ROOT / "pictures" / "presentations"


def main() -> None:
    """Ajuste, compare, affiche le résumé et écrit les PNG."""
    result = evaluate_selection_files(
        str(TRAIN),
        str(TARGETS),
        n_jobs=-1,
        verbose=True,
    )
    print(format_selection(result), flush=True)
    print("--- essais ---", flush=True)
    print(result.trials.to_string(index=False), flush=True)
    print("--- calibration ---", flush=True)
    print(result.calibration.to_string(index=False), flush=True)

    scores = pd.DataFrame(
        {
            "key": [item.name for item in result.scores],
            "rmse": [item.rmse for item in result.scores],
        }
    )
    for language in ("fr", "en"):
        save_error_gap_chart(
            scores,
            PICTURES / f"ecart-modeles-{language}.png",
            language=language,
            chosen=result.chosen,
        )
        save_calibration_chart(
            result.calibration,
            PICTURES / f"calibration-charge-{language}.png",
            language=language,
            chosen=result.chosen,
        )
    print("graphes écrits", flush=True)


if __name__ == "__main__":
    main()
