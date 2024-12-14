import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from .utils import ConvergenceHistory, rmse, whether_to_stop


class GradientBoostingMSE:
    const_prediction: float = None

    def __init__(
            self,
            n_estimators: int,
            tree_params: dict[str, Any] | None = None,
            learning_rate=0.1,
    ) -> None:
        """
        Initializes the GradientBoostingMSE model.

        This is a handmade gradient boosting regressor that trains a sequence of
        short decision trees to correct the errors of each other's predictions.
        It employs scikit-learn's `DecisionTreeRegressor` under the hood.

        Args:
            n_estimators (int): Number of trees to boost each other.
            tree_params (dict[str, Any] | None, optional): Parameters for the
                decision trees. Defaults to None.
            learning_rate (float, optional): Scaling factor for the "gradient"
                step (the weight applied to each tree prediction). Defaults to
                0.1.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if tree_params is None:
            tree_params = {}
        if tree_params.get("max_features") is None:
            tree_params["max_features"] = 1 / 3
        self.tree_params = tree_params
        self.forest = [
            DecisionTreeRegressor(**tree_params) for _ in range(n_estimators)
        ]

    def fit(
            self,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            X_val: npt.NDArray[np.float64] | None = None,
            y_val: npt.NDArray[np.float64] | None = None,
            trace: bool | None = None,
            patience: int | None = None,
    ) -> ConvergenceHistory | None:
        """
        Trains an ensemble of trees on the provided data.

        Args:
            X (npt.NDArray[np.float64]): Objects features matrix, array of shape
                (n_objects, n_features).
            y (npt.NDArray[np.float64]): Regression labels, array of shape
                (n_objects,).
            X_val (npt.NDArray[np.float64] | None, optional): Validation set of
                objects, array of shape (n_val_objects, n_features). Defaults
                to None.
            y_val (npt.NDArray[np.float64] | None, optional): Validation set
                of labels, array of shape (n_val_objects,). Defaults to None.
            trace (bool | None, optional): Whether to calculate RMSLE while
                training. True by default if validation data is provided.
                Defaults to None.
            patience (int | None, optional): Number of training steps without
                decreasing the train loss (or validation if provided), after
                which to stop training. Defaults to None.

        Returns:
            ConvergenceHistory | None: Instance of `ConvergenceHistory`
            if `trace=True` or if validation data is provided.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if y_val is not None and isinstance(y_val, pd.Series):
            y_val = y_val.values

        if trace is None:
            trace = X_val is not None and y_val is not None

        history = {'train': [], 'val': []} if trace else None

        y_pred = np.zeros_like(y)

        for i in range(self.n_estimators):
            residuals = y - y_pred

            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_bootstrap = X[indices]
            residuals_bootstrap = residuals[indices]

            tree = self.forest[i]
            tree.fit(X_bootstrap, residuals_bootstrap)

            y_pred += self.learning_rate * tree.predict(X)

            if trace:
                train_loss = rmse(y, y_pred)
                history['train'].append(train_loss)

                if X_val is not None and y_val is not None:
                    val_pred = self.predict(X_val, i)
                    val_loss = rmse(y_val, val_pred)
                    history['val'].append(val_loss)

                    if patience is not None and whether_to_stop(history,
                                                                patience):
                        break

        return history

    def predict(self, X: np.ndarray, n: int | None) -> np.ndarray:
        """
        Makes predictions with the ensemble of trees.

        All the trees make sequential predictions.

        Args:
            X (npt.NDArray[np.float64]): Objects' features matrix, array of
                shape (n_objects, n_features).

        Returns:
            npt.NDArray[np.float64]: Predicted values, array of shape
            (n_objects,).
        """
        if n is None:
            n = self.n_estimators
        y_pred = np.zeros(X.shape[0])
        for tree in self.forest[:n + 1]:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def dump(self, dirpath: str) -> None:
        """
        Saves the model to the specified directory.

        Args:
            dirpath (str): Path to the directory where the model will be saved.
        """
        path = Path(dirpath)
        # path.mkdir(parents=True)

        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "const_prediction": self.const_prediction,
        }
        with (path / "params.json").open("w") as file:
            json.dump(params, file, indent=4)

        trees_path = path / "trees"
        trees_path.mkdir()
        for i, tree in enumerate(self.forest):
            joblib.dump(tree, trees_path / f"tree_{i:04d}.joblib")

    @classmethod
    def load(cls, dirpath: str) -> "GradientBoostingMSE":
        """
        Loads the model from the specified directory.

        Args:
            dirpath (str): Path to the directory where the model is saved.

        Returns:
            GradientBoostingMSE: An instance of the GradientBoostingMSE model.
        """
        with (Path(dirpath) / "params.json").open() as file:
            params = json.load(file)
        instance = cls(params["n_estimators"],
                       learning_rate=params["learning_rate"])

        trees_path = Path(dirpath) / "trees"

        instance.forest = [
            joblib.load(trees_path / f"tree_{i:04d}.joblib")
            for i in range(params["n_estimators"])
        ]
        instance.const_prediction = params["const_prediction"]

        return instance
