"""Entraînement et inférence des modèles."""

from agri_fire.models.baseline import (
    SEED,
    HoldoutScore,
    cost_training_rows,
    evaluate_baseline,
    fit_baseline,
    rmse,
    split_rows,
)
from agri_fire.models.selection import (
    SelectionResult,
    TreeParams,
    evaluate_selection_files,
    fit_ridge,
    fit_trees,
    select_models,
)

__all__ = [
    "SEED",
    "HoldoutScore",
    "SelectionResult",
    "TreeParams",
    "cost_training_rows",
    "evaluate_baseline",
    "evaluate_selection_files",
    "fit_baseline",
    "fit_ridge",
    "fit_trees",
    "rmse",
    "select_models",
    "split_rows",
]
