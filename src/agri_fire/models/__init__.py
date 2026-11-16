"""Entraînement et inférence des modèles."""

from agri_fire.models.artifact import (
    RETAINED_ALPHA_COST,
    RETAINED_ALPHA_FREQUENCY,
    fit_retained_model,
    load_model,
    save_model,
    train_retained_files,
)
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
    "RETAINED_ALPHA_COST",
    "RETAINED_ALPHA_FREQUENCY",
    "SEED",
    "HoldoutScore",
    "SelectionResult",
    "TreeParams",
    "cost_training_rows",
    "evaluate_baseline",
    "evaluate_selection_files",
    "fit_baseline",
    "fit_retained_model",
    "fit_ridge",
    "fit_trees",
    "load_model",
    "rmse",
    "save_model",
    "select_models",
    "split_rows",
    "train_retained_files",
]
