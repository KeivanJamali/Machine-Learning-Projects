import torch
# from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str):
    model.train()
    train_loss, train_acc = 0, 0
    true_predict = {"true": [], "predict": []}

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += r2_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        true_predict["true"].append(y.to("cpu"))
        true_predict["predict"].append(y_pred.to("cpu"))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc, true_predict


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str):
    model.eval()
    test_loss, test_acc = 0, 0
    true_predict = {"true": [], "predict": []}

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device),
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y)
            test_acc += r2_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            true_predict["true"].append(y.to("cpu"))
            true_predict["predict"].append(y_pred.to("cpu"))

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc, true_predict


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          experiment_name: str,
          model_name: str,
          early_stop_patience: int,
          device: str):
    best_loss = float("inf")
    early_stop = 0
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    train_true_predict = {"true": [], "predict": []}
    test_true_predict = {"true": [], "predict": []}

    model.to(device)
    writer = create_writer(experiment_name=experiment_name, model_name=model_name, extra=str(epochs) + "epoch")
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, true_predict = train_step(model=model,
                                                         dataloader=train_dataloader,
                                                         loss_fn=loss_fn,
                                                         optimizer=optimizer,
                                                         device=device)
        train_true_predict["true"].extend(true_predict["true"])
        train_true_predict["predict"].extend(true_predict["predict"])

        test_loss, test_acc, true_predict = test_step(model=model,
                                                      dataloader=test_dataloader,
                                                      loss_fn=loss_fn,
                                                      device=device)
        test_true_predict["true"].extend(true_predict["true"])
        test_true_predict["predict"].extend(true_predict["predict"])
        print(
            f"Epoch {epoch} | train: Loss {train_loss:.6f} Accuracy {train_acc:.2f} | validation: Loss {test_loss:.6f} Accuracy {test_acc:.2f}")

        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)
        results["test_acc"].append(test_acc)
        results["test_loss"].append(test_loss)

        if writer:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"Train_loss": train_loss,
                                                "Validation_Loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"Train_accuracy": train_acc,
                                                "Validation_accuracy": test_acc},
                               global_step=epoch)
            writer.close()

        if test_loss <= best_loss:
            best_loss = test_loss
            early_stop = 0
        else:
            early_stop += 1
        if early_stop_patience and early_stop >= early_stop_patience:
            print(f"Early_Stop_at_ {epoch} Epoch")
            break
    return results, {"train": train_true_predict, "test": test_true_predict}


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None):
    """Creates torch.utils.tensorboard.writer.SummaryWriter() instance tracking to a specific directory."""
    from datetime import datetime
    import os
    # get timestamp of current date in reverse order : YYYY_MM_DD | datetime.now().strftime("%Y-%m-%d-%H-%M-%S") |
    timestamp = datetime.now().strftime("%Y-%m-%d-%H")

    # create log directory
    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    print(f"[INFO] create SummaryWriter saving to {log_dir}")
    return SummaryWriter(log_dir=log_dir)
