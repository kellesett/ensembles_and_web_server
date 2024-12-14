import json
import os

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Query

from .schemas import (
    ExperimentConfig,
    ExistingExperimentsResponse,
    MessageResponse,
    BoolResponse,
    ConvergenceHistoryResponse
)

from ensembles import RandomForestMSE, GradientBoostingMSE
from ensembles.utils import ConvergenceHistory
import pandas as pd

from sklearn.model_selection import train_test_split

app = FastAPI()


def get_runs_dir() -> Path:
    return Path.cwd() / "runs"


@app.get("/existing_experiments/")
async def existing_experiments() -> ExistingExperimentsResponse:
    """
    Get information about existing experiments.

    This endpoint scans the directory where experiments are stored and returns a list of
    existing experiments along with their absolute paths. Each experiment is stored as
    a directory in the host filesystem.

    Returns:
        ExistingExperimentsResponse: A response containing the location of the experiments
        directory, absolute paths of the experiment directories, and the names of the experiments.
    """
    path = get_runs_dir()
    response = ExistingExperimentsResponse(location=path)
    if not path.exists():
        return response
    response.abs_paths = [obj for obj in path.iterdir() if obj.is_dir()]
    response.experiment_names = [filepath.stem for filepath in response.abs_paths]
    return response


@app.post("/register_experiment/")
async def register_experiment(experiment_config: str = Form(...),
                              train_file: UploadFile = File(...)) -> MessageResponse:
    
    experiment_config = ExperimentConfig(**json.loads(experiment_config))
    path = Path(os.sep.join(["runs", f"{experiment_config.name}"]))
    if path.exists():
        raise ValueError('WrOng NaMe BrOtHer')
    path.mkdir(mode=0o777, parents=True)

    config_path = path.joinpath('config.json')
    config_path.write_text(experiment_config.model_dump_json())
    config_path.chmod(0o777)
    train_file_path = path.joinpath('train_file.csv')
    train_file_path.write_bytes(train_file.file.read())
    train_file_path.chmod(0o777)

    response = MessageResponse(
        message='OK'
    )
    return response


@app.get("/load_experiment_config/")
async def existing_experiments(experiment_name: str = Query(...)) -> ExperimentConfig:
    path = Path(os.sep.join(["runs", experiment_name, 'config.json']))
    experiment_config = json.loads(path.read_text())
    response = ExperimentConfig(**experiment_config)
    return response


@app.get("/needs_training/")
async def existing_experiments(experiment_name: str = Query(...)) -> BoolResponse:
    path = Path(os.sep.join(["runs", experiment_name, 'convergence_history.json']))
    return BoolResponse(response=not path.exists())


@app.put("/train_model/")
async def register_experiment(experiment_name: str = Query(...)) -> MessageResponse:
    path = os.sep.join(["runs", experiment_name, "train_file.csv"])
    df = pd.read_csv(path)

    path = Path(path)
    path = path.with_name("config.json")
    config = ExperimentConfig(**json.loads(path.read_text()))

    models = {
        "Random Forest": RandomForestMSE,
        "Gradient Boosting": GradientBoostingMSE
    }
    model_type = models[config.ml_model]
    if config.max_features == 'all':
        config.max_features = None
    tree_params = {
        "max_depth": config.max_depth,
        "max_features": config.max_features
    }

    model = model_type(
        config.n_estimators,
        tree_params=tree_params
    )

    target = df[config.target_column]
    X_train, X_val, y_train, y_val = train_test_split(
        df.drop(config.target_column, axis=1).to_numpy(),
        target.to_numpy(),
        test_size=0.3,
        random_state=52
    )
    history = model.fit(X_train, y_train, X_val, y_val, trace=True)[0]
    model.dump(os.sep.join(["runs", experiment_name, "model"]))

    history_path = path.with_name("convergence_history.json")
    history_path.write_text(json.dumps(history, indent=4))

    response = MessageResponse(
        message='OK'
    )
    return response


@app.get("/get_convergence_history/")
async def existing_experiments(experiment_name: str = Query(...)) -> ConvergenceHistoryResponse:
    path = Path(os.sep.join(["runs", experiment_name, 'convergence_history.json']))
    response = ConvergenceHistoryResponse(**json.loads(path.read_text()))
    return response
