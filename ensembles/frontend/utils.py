import pandas as pd
import plotly.express as px

from ensembles.backend.schemas import ConvergenceHistoryResponse


def plot_learning_curves(convergence_history: ConvergenceHistoryResponse):
    """
    Plots the learning curves for the training and validation datasets
    based on the provided convergence history.

    Args:
        convergence_history (ConvergenceHistoryResponse): The convergence
        history containing the RMSLE values for training and validation
        datasets across iterations.

    Returns:
        plotly.graph_objects.Figure: A Plotly line plot showing the RMSLE
        for training and validation datasets across iterations.
    """
    df = pd.DataFrame(convergence_history.model_dump())
    df_melted = df.reset_index().melt(
        id_vars=["index"],
        value_vars=["train", "val"],
        var_name="Dataset",
        value_name="RMSLE",
    )
    train_loss = min(convergence_history.train)
    val_loss = min(convergence_history.val)

    return px.line(
        df_melted,
        x="index",
        y="RMSLE",
        color="Dataset",
        labels={"index": "Iterations", "RMSLE": "RMSLE"},
        title=f"RMSLE: train [{train_loss:.4f}] | validation [{val_loss:.4f}]")
