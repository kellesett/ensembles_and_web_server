from typing import Any

import numpy.typing as npt
import requests

from ensembles.backend.schemas import ExperimentConfig, ConvergenceHistoryResponse


class Client:
    def __init__(self, base_url: str) -> None:
        """
        Initializes the Client with a base URL for the API.

        Args:
            base_url (str): The base URL of the API.
        """

        self.base_url = base_url
        self.session = requests.Session()

    def get_names(self) -> list[str]:
        """
        Retrieves the names of all existing experiments.

        Returns:
            list[str]: A list of experiment names.
        """

        response = self.session.get(f"{self.base_url}/existing_experiments/")
        response.raise_for_status()
        return response.json()["experiment_names"]

    def register_experiment(self, experiment_config, train_file) -> None:
        """
        Registers a new experiment with the given configuration and training data.

        Args:
            experiment_config (Any): The configuration for the experiment.
            train_file (Any): The training data file.
        """

        response = self.session.post(
            f"{self.base_url}/register_experiment/",
            data={"experiment_config": experiment_config.model_dump_json()},
            files={"train_file": train_file}
        )
        response.raise_for_status()

    def load_experiment_config(self, experiment_name) -> dict[str, Any]:
        """
        Loads the configuration of an existing experiment.

        Args:
            experiment_name (Any): The name of the experiment.

        Returns:
            ExperimentConfig: The configuration of the experiment.
        """

        response = self.session.get(
            f"{self.base_url}/load_experiment_config/",
            params={'experiment_name': experiment_name}
        )
        response.raise_for_status()
        return ExperimentConfig(**response.json())

    def is_training_needed(self, experiment_name) -> bool:
        """
        Request info about was the model ever trained.

        Args:
            experiment_name (Any): The name of the experiment.
        
        Returns:
            bool: indicator was the model ever trained.
        """

        response = self.session.get(
            f"{self.base_url}/needs_training/",
            params={'experiment_name': experiment_name}
        )
        response.raise_for_status()
        return response.json()['flag']

    def train_model(self, experiment_name) -> None:
        """
        Trains the model for the specified experiment.

        Args:
            experiment_name (Any): The name of the experiment.
        """

        response = self.session.put(
            f"{self.base_url}/train_model/",
            params={'experiment_name': experiment_name}
        )
        response.raise_for_status()

    def get_convergence_history(self, experiment_name) -> ConvergenceHistoryResponse:
        """
        Retrieves the convergence history of the specified experiment.

        Args:
            experiment_name (Any): The name of the experiment.

        Returns:
            ConvergenceHistory: The convergence history of the experiment.
        """

        response = self.session.get(
            f"{self.base_url}/get_convergence_history/",
            params={'experiment_name': experiment_name}
        )
        response.raise_for_status()
        return ConvergenceHistoryResponse(**response.json())

    def predict(self, experiment_name, test_file) -> npt.NDArray[Any]:
        """
        Makes predictions using the trained model of the specified experiment.

        Args:
            experiment_name (Any): The name of the experiment.
            test_file (Any): The test data file.

        Returns:
            npt.NDArray[Any]: The predictions made by the model.
        """

        response = self.session.get(
            f"{self.base_url}/predict/",
            params={"experiment_name": experiment_name},
            files={"test_file": test_file}
        )
        response.raise_for_status()
        return response.json()["predicted_values"]
