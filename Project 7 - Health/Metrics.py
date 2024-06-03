import torch
import numpy as np

def MAPE(y_pred, y_true, mode):
    if mode == "torch":
        mape = torch.mean(torch.abs((y_true - y_pred) / y_true))
        if mape > 100:
            return torch.median(torch.abs((y_true - y_pred) / y_true))
        else:
            return mape
    elif mode == "numpy":
        mape = np.mean(np.abs((y_true - y_pred) / y_true))
        if mape > 100:
            return np.median(np.abs((y_true - y_pred) / y_true))
        else:
            return mape
    else:
        raise TypeError("Mode should be 'torch' or 'numpy'.")
    
def SMAPE(y_pred, y_true, mode):
    if mode == "torch":
        smape = torch.mean(torch.abs((y_true - y_pred) / ((torch.abs(y_true)+torch.abs(y_pred))/2)))
        if smape > 100:
            return torch.median(torch.abs((y_true - y_pred) / y_true))
        else:
            return smape
    elif mode == "numpy":
        smape = np.mean(np.abs((y_true - y_pred) / ((np.abs(y_true)+torch.abs(y_pred))/2)))
        if smape > 100:
            return np.median(np.abs((y_true - y_pred) / y_true))
        else:
            return smape
    else:
        raise TypeError("Mode should be 'torch' or 'numpy'.")
    
def MAE(y_pred, y_true, mode):
    if mode == "torch":
        mae = torch.mean(torch.abs(y_true - y_pred))
        return mae
    elif mode == "numpy":
        mae = np.mean(np.abs(y_true - y_pred))
        return mae
    else:
        raise TypeError("Mode should be 'torch' or 'numpy'.")

def MSE(y_pred, y_true, mode):
    if mode == "torch":
        mse = torch.mean((y_true - y_pred)**2)
        return mse
    elif mode == "numpy":
        mse = np.mean((y_true - y_pred)**2)
        return mse
    else:
        raise TypeError("Mode should be 'torch' or 'numpy'.")

def RMSE(y_pred, y_true, mode):
    if mode == "torch":
        rmse = torch.sqrt(torch.mean((y_true - y_pred)**2))
    elif mode == "numpy":
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    else:
        raise TypeError("Mode should be 'torch' or 'numpy'.")

def r2_score(y_pred, y_true, mode):
    """
    Calculate the R-squared (coefficient of determination) for a regression model.

    Args:
        y_true (torch.Tensor): The true target values.
        y_pred (torch.Tensor): The predicted target values.

    Returns:
        torch.Tensor: The R-squared value.
    """
    if mode == "torch":
        mean = torch.mean(y_true)
        # total sum of squares
        tss = torch.sum((y_true - mean) ** 2)
        # residual sum of squares
        rss = torch.sum((y_true - y_pred) ** 2)
        # coefficient of determination
        r2 = 1 - (rss / tss)
        return r2
    elif mode == "numpy":
        mean = np.mean(y_true)
        # total sum of squares
        tss = np.sum((y_true - mean) ** 2)
        # residual sum of squares
        rss = np.sum((y_true - y_pred) ** 2)
        # coefficient of determination
        r2 = 1 - (rss / tss)
        return r2
    else:
        raise TypeError("Mode should be 'torch' or 'numpy'.")