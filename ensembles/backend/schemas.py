from pathlib import Path

from pydantic import BaseModel, field_validator


class ExistingExperimentsResponse(BaseModel):
    """
    Response model for existing experiments.

    Attributes:
        location (Path): The directory path where the experiments are stored.
        experiment_names (list[str]): A list of names of the existing experiments. Defaults to an empty list.
        abs_paths (list[Path]): A list of absolute paths to the experiment directories. Defaults to an empty list.
    """

    location: Path
    experiment_names: list[str] = []
    abs_paths: list[Path] = []


class ConvergenceHistoryResponse(BaseModel):

    train: list[float]
    val: list[float] | None = None


class ExperimentConfig(BaseModel):
    name: str
    ml_model: str
    n_estimators: int
    max_depth: int
    max_features: int | float | str
    target_column: str

    @field_validator('max_features')
    def validate_max_features(cls, value):
        """
        Validator for the max_features parameter.
        :param value:
        :return:
        """

        if isinstance(value, str):
            if value.isdigit():
                return int(value)
            try:
                return float(value)
            except ValueError:
                if value not in {'sqrt', 'log2'}:
                    if value == "all":
                        return 1.0
                    raise ValueError(
                        f"Invalid string value for max_features: {value}")
        elif isinstance(value, (int, float)):
            return value
        else:
            raise ValueError(f"Invalid type for max_features: {type(value)}")
        return value



class TrainModelRequest(BaseModel):
    name: str
