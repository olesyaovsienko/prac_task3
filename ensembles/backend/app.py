import json
import os
import shutil
from pathlib import Path
from typing import Annotated, Dict, List, Union

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from sklearn.model_selection import train_test_split

from ensembles.boosting import GradientBoostingMSE
from ensembles.random_forest import RandomForestMSE

from .schemas import (ConvergenceHistoryResponse, ExistingExperimentsResponse,
                      ExperimentConfig, TrainModelRequest)

app = FastAPI()


def get_runs_dir() -> Path:
    """
    Returns the directory path where experiment runs are stored.

    Returns:
        Path: The path to the 'runs' directory.
    """
    return Path.cwd() / "runs"


def convert_to_json_serializable(
        conv_history: ConvergenceHistoryResponse) -> Dict[str, List[float]]:
    """
    Converts the convergence history to a JSON serializable format.

    Args:
        conv_history (ConvergenceHistoryResponse): The convergence history response object.

    Returns:
        Dict[str, List[float]]: A dictionary with 'train' and 'val' keys containing lists of floats.
    """
    return {
        "train": list(map(float, conv_history["train"])),
        "val": list(map(float, conv_history.get("val", [])))
    }


@app.get("/existing_experiments/", response_model=ExistingExperimentsResponse)
async def existing_experiments() -> ExistingExperimentsResponse:
    """
    Retrieves a list of existing experiment directories.

    Returns:
        ExistingExperimentsResponse: An object containing the location, absolute paths,
                                     and experiment names of existing experiments.
    """
    path = get_runs_dir()
    response = ExistingExperimentsResponse(location=path)
    if not path.exists():
        return response
    response.abs_paths = [obj for obj in path.iterdir() if obj.is_dir()]
    response.experiment_names = [filepath.stem for filepath in
                                 response.abs_paths]
    return response


@app.post("/register_experiment/")
async def register_experiment(
        name: Annotated[str, Form(...)],
        ml_model: Annotated[str, Form(...)],
        n_estimators: Annotated[int, Form(...)],
        max_depth: Annotated[int, Form(...)],
        max_features: Annotated[str, Form(...)],
        target_column: Annotated[str, Form(...)],
        train_file: Annotated[UploadFile, File(...)]
) -> Dict[str, str]:
    """
    Registers a new experiment and saves the configuration and training file.

    Args:
        name (str): The name of the experiment.
        ml_model (str): The machine learning model to use.
        n_estimators (int): The number of estimators for the model.
        max_depth (int): The maximum depth of the trees.
        max_features (str): The maximum number of features to consider.
        target_column (str): The target column in the training data.
        train_file (UploadFile): The training data file.

    Returns:
        Dict[str, str]: A dictionary with a success message.
    """
    experiment_dir = get_runs_dir() / name
    if not experiment_dir.exists():
        os.makedirs(experiment_dir)

    config = ExperimentConfig(
        name=name,
        ml_model=ml_model,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        target_column=target_column
    )

    config_path = experiment_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config.model_dump_json())

    train_file_path = experiment_dir / train_file.filename
    with open(train_file_path, "wb") as buffer:
        shutil.copyfileobj(train_file.file, buffer)

    return {"message": "Experiment registered successfully"}


@app.get("/load_experiment_config/")
async def load_experiment_config(name: str) -> Dict[
        str, Union[str, int, float]]:
    """
    Loads the configuration for a given experiment.

    Args:
        name (str): The name of the experiment.

    Returns:
        Dict[str, Union[str, int, float]]: The configuration of the experiment.
    """
    experiment_dir = get_runs_dir() / name
    config_path = experiment_dir / "config.json"

    with open(config_path, encoding="utf-8") as f:
        config_data = f.read()

    config = ExperimentConfig.model_validate_json(config_data)
    return dict(config)


@app.get("/needs_training/")
async def needs_training(experiment_name: str) -> Dict[str, bool]:
    """
    Checks if an experiment needs training.

    Args:
        experiment_name (str): The name of the experiment.

    Returns:
        Dict[str, bool]: A dictionary indicating whether the experiment needs training.
    """
    experiment_dir = get_runs_dir() / experiment_name
    model_path = experiment_dir / "trees"
    if not model_path.exists():
        return {"response": True}
    return {"response": False}


@app.post("/train_model/", response_model=Dict[str, str])
async def train_model(request: TrainModelRequest) -> Dict[str, str]:
    """
    Trains a model for a given experiment based on the configuration.

    Args:
        request (TrainModelRequest): The request object containing the experiment name.

    Returns:
        Dict[str, str]: A dictionary with a success message.
    """
    experiment_name = request.name
    experiment_dir = get_runs_dir() / experiment_name
    config_path = experiment_dir / "config.json"

    with open(config_path, encoding="utf-8") as f:
        config_json = f.read()
        config = ExperimentConfig.model_validate_json(config_json)

    name = config.name
    ml_model = config.ml_model
    n_estimators = config.n_estimators
    max_depth = config.max_depth
    max_features = config.max_features
    target_column = config.target_column
    tree_params = {"max_depth": max_depth, "max_features": max_features}

    experiment_dir = get_runs_dir() / name
    train_file_path = next(experiment_dir.glob("*.csv"))
    data = pd.read_csv(train_file_path)

    X_full = data.drop(columns=[target_column])
    y_full = data[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full,
                                                      test_size=0.2,
                                                      random_state=42)
    model = RandomForestMSE(n_estimators,
                            tree_params) if ml_model == "Random Forest" else GradientBoostingMSE(
        n_estimators, tree_params)
    conv_history = model.fit(X_train, y_train, X_val, y_val, trace=True)

    history_serializable = convert_to_json_serializable(conv_history)

    history_path = experiment_dir / "convergence_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_serializable, f)

    model.dump(str(experiment_dir))

    return {"message": "Model trained successfully"}


@app.post("/predict/", response_model=Dict[str, List[float]])
async def predict(
        name: Annotated[str, Form(...)],
        test_file: Annotated[UploadFile, File(...)]
) -> Dict[str, List[float]]:
    """
    Makes predictions using a trained model for a given experiment.

    Args:
        name (str): The name of the experiment.
        test_file (UploadFile): The test data file.

    Returns:
        Dict[str, List[float]]: A dictionary with the list of predictions.
    """
    experiment_dir = get_runs_dir() / name
    with open(experiment_dir / "config.json", encoding="utf-8") as f:
        config = ExperimentConfig.model_validate_json(f.read())
        if config.ml_model == "Random Forest":
            model = RandomForestMSE.load(str(experiment_dir))
        else:
            model = GradientBoostingMSE.load(str(experiment_dir))

    test_file_path = experiment_dir / test_file.filename
    with open(test_file_path, "wb") as buffer:
        shutil.copyfileobj(test_file.file, buffer)
    X_test = pd.read_csv(test_file_path)

    predictions = model.predict(X_test, X_test.shape[0])
    return {"predictions": list(predictions)}


@app.get("/convergence_history/", response_model=Dict[str, List[float]])
async def convergence_history(name: str) -> Union[
        Dict[str, List[float]], Dict[str, str]]:
    """
    Retrieves the convergence history for a given experiment.

    Args:
        name (str): The name of the experiment.

    Returns:
        Union[Dict[str, List[float]], Dict[str, str]]: The convergence history or an error message.
    """
    experiment_dir = get_runs_dir() / name
    history_path = experiment_dir / "convergence_history.json"
    if not history_path.exists():
        return {"error": "Convergence history not found"}

    with open(history_path, encoding="utf-8") as f:
        history_data = json.load(f)
    return history_data
