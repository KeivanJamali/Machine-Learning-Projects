import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str) -> tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_logit = model(X)
        loss = loss_fn(y_logit, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_logit)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: str) -> tuple[float, float]:
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            val_loss += loss_fn(y_logit, y).item()
            val_pred_labels = y_logit.argmax(dim=1)
            val_acc += ((val_pred_labels == y).sum().item() / len(y_logit))

    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          experiment_name: str,
          model_name: str,
          early_stop_patience: int = None,
          device: str = "cpu") -> dict[str, list]:
    best_loss = float("inf")  # for early stopping
    early_stop = 0
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []}
    model.to(device)

    writer = create_writer(experiment_name=experiment_name,
                           model_name=model_name, extra=str(epochs)+"epoch")

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        val_loss, val_acc = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fn,
                                     device=device)

        print(
            f"Epoch {epoch} | train: Loss {train_loss:.6f} Accuracy {train_acc:.2f} | validation: Loss {val_loss:.6f} Accuracy {val_acc:.2f}")

        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)
        results["val_acc"].append(val_acc)
        results["val_loss"].append(val_loss)
        if writer:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"Train_loss": train_loss,
                                                "Validation_Loss": val_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"Train_accuracy": train_acc,
                                                "Validation_accuracy": val_acc},
                               global_step=epoch)
            writer.add_graph(model=model,
                             input_to_model=torch.randn(32, 6).to(device))
            writer.close()
        # early stopping
        if val_loss <= best_loss:
            best_loss = val_loss
            early_stop = 0
        else:
            early_stop += 1
        if early_stop_patience and early_stop >= early_stop_patience:
            print(f"Early_Stop_at_ {epoch} Epoch")
            break

    return results


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
