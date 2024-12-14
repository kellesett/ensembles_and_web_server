import json
import os

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Query

from .schemas import (
    ExperimentConfig,
    ExistingExperimentsResponse,
    MessageResponse,
    BoolResponse
)

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
    path.mkdir(mode=0o777)

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
    """
    Get information about existing experiments.

    This endpoint scans the directory where experiments are stored and returns a list of
    existing experiments along with their absolute paths. Each experiment is stored as
    a directory in the host filesystem.

    Returns:
        ExistingExperimentsResponse: A response containing the location of the experiments
        directory, absolute paths of the experiment directories, and the names of the experiments.
    """
    path = Path(os.sep.join(["runs", experiment_name, 'config.json']))
    experiment_config = json.loads(path.read_text())
    response = ExperimentConfig(**experiment_config)
    return response


@app.get("/needs_training/")
async def existing_experiments(experiment_name: str = Query(...)) -> BoolResponse:
    """
    Get information about existing experiments.

    This endpoint scans the directory where experiments are stored and returns a list of
    existing experiments along with their absolute paths. Each experiment is stored as
    a directory in the host filesystem.

    Returns:
        ExistingExperimentsResponse: A response containing the location of the experiments
        directory, absolute paths of the experiment directories, and the names of the experiments.
    """
    path = Path(os.sep.join(["runs", experiment_name, 'params.json']))
    return BoolResponse(response=path.exists())
