import pandas as pd
import plotly.express as px

from ensembles.backend.schemas import ConvergenceHistoryResponse


def plot_learning_curves(convergence_history: ConvergenceHistoryResponse):
    df = pd.DataFrame(convergence_history.model_dump())
    df_melted = df.reset_index().melt(
        id_vars=["index"],
        value_vars=["train", "val"],
        var_name="Dataset",
        value_name="RMSE",
    )
    train_loss = min(convergence_history.train)
    val_loss = min(convergence_history.val)

    return px.line(
        df_melted,
        x="index",
        y="RMSE",
        color="Dataset",
        labels={"index": "Iterations", "RMSE": "RMSE"},
        title=f"RMSE: train [{train_loss:.4f}] | validation [{val_loss:.4f}]",
    )
