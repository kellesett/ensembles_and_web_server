import json
from time import perf_counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import NotFittedError

from .utils import ConvergenceHistory, whether_to_stop, rmse


class RandomForestMSE:
    def __init__(
        self, n_estimators: int, tree_params: dict[str, Any] | None = None
    ) -> None:
        """
        Handmade random forest regressor.

        Classic ML algorithm that trains a set of independent tall decision trees and averages its predictions. Employs scikit-learn `DecisionTreeRegressor` under the hood.

        Args:
            n_estimators (int): Number of trees in the forest.
            tree_params (dict[str, Any] | None, optional): Parameters for sklearn trees. Defaults to None.
        """
        self.n_estimators = n_estimators
        if tree_params is None:
            tree_params = {}
        self.forest = [
            DecisionTreeRegressor(**tree_params) for _ in range(n_estimators)
        ]
        self.fitted_estimators = 0

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
        Train an ensemble of trees on the provided data.

        Args:
            X (npt.NDArray[np.float64]): Objects features matrix, array of shape (n_objects, n_features).
            y (npt.NDArray[np.float64]): Regression labels, array of shape (n_objects,).
            X_val (npt.NDArray[np.float64] | None, optional): Validation set of objects, array of shape (n_val_objects, n_features). Defaults to None.
            y_val (npt.NDArray[np.float64] | None, optional): Validation set of labels, array of shape (n_val_objects,). Defaults to None.
            trace (bool | None, optional): Whether to calculate rmse while training. True by default if validation data is provided. Defaults to None.
            patience (int | None, optional): Number of training steps without decreasing the train loss (or validation if provided), after which to stop training. Defaults to None.

        Returns:
            ConvergenceHistory | None: Instance of `ConvergenceHistory` if `trace=True` or if validation data is provided.
        """
        np.random.seed(42)

        if y_val is not None:
            trace = True
        elif trace is None:
            trace = False

        times = list()
        history = ConvergenceHistory(train=[], val=[])
        pred = np.zeros((X.shape[0]))

        if y_val is not None:
            val_pred = np.zeros((X_val.shape[0]))
        for epoch, estimator in enumerate(self.forest):
            idx = np.random.choice(
                np.arange(y.shape[0]), y.shape[0], replace=True)

            start = perf_counter()
            estimator.fit(X[idx], y[idx])
            self.fitted_estimators += 1

            pred += estimator.predict(X)
            times.append(perf_counter() - start)
            history['train'].append(rmse(y, pred / self.fitted_estimators))

            if y_val is not None:
                val_pred += estimator.predict(X_val)
                history['val'].append(
                    rmse(y_val, val_pred / self.fitted_estimators))

            if patience is not None and whether_to_stop(history, patience):
                break

        if trace:
            return history, times
        else:
            return None

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Make prediction with ensemble of trees.

        All the trees make their own predictions which then are averaged.

        Args:
            X (npt.NDArray[np.float64]): Objects' features matrix, array of shape (n_objects, n_features).

        Returns:
            npt.NDArray[np.float64]: Predicted values, array of shape (n_objects,).
        """
        if self.fitted_estimators == 0:
            raise NotFittedError

        predictions = np.empty((X.shape[0], self.fitted_estimators))
        for i in range(self.fitted_estimators):
            predictions[:, i] = self.forest[i].predict(X)

        return np.mean(predictions, axis=1)

    def dump(self, dirpath: str) -> None:
        """
        Save the trained model to the specified directory.

        Args:
            dirpath (str): Path to the directory where the model will be saved.
        """
        path = Path(dirpath)
        path.mkdir(parents=True)

        params = {
            "n_estimators": self.n_estimators,
            "fitted_estimators": self.fitted_estimators
        }
        with (path / "params.json").open("w") as file:
            json.dump(params, file, indent=4)

        trees_path = path / "trees"
        trees_path.mkdir()
        for i, tree in enumerate(self.forest):
            joblib.dump(tree, trees_path / f"tree_{i:04d}.joblib")

    @classmethod
    def load(cls, dirpath: str) -> "RandomForestMSE":
        """
        Load a trained model from the specified directory.

        Args:
            dirpath (str): Path to the directory where the model is saved.

        Returns:
            RandomForestMSE: An instance of the loaded model.
        """
        with (Path(dirpath) / "params.json").open() as file:
            params = json.load(file)
        instance = cls(params["n_estimators"])

        trees_path = Path(dirpath) / "trees"

        instance.forest = [
            joblib.load(trees_path / f"tree_{i:04d}.joblib")
            for i in range(params["n_estimators"])
        ]
        instance.fitted_estimators = params["fitted_estimators"]

        return instance
