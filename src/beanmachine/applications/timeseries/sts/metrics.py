import torch


def mape(y_pred, y_true) -> torch.Tensor:
    """
    Mean Absolute Percentage Error for the predictions `y_pred` when compared
    to the baseline given by `y_true`.

    :param y_pred: predictions from the model.
    :param y_true: true observations.
    :return: torch scalar denoting the MAPE.
    """
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
