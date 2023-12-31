import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
from datetime import datetime
import os


class Machine_Engine:
    def __init__(self, model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader):
        """
        Initializes the class object with the given model, dataloaders, and device.

        Parameters:
            model (torch.nn.Module): The model to be used for training and evaluation.
            train_dataloader (torch.utils.data.DataLoader): The dataloader for the training data.
            val_dataloader (torch.utils.data.DataLoader): The dataloader for the validation data.
            test_dataloader (torch.utils.data.DataLoader): The dataloader for the test data.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = None
        self.train_true_predict_list, self.val_true_predict_list, self.test_true_predict_list = None, None, None

    def _train_step(self, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple:
        """
        Trains the model for one step.

        Args:
            loss_fn (torch.nn.Module): The loss function used for training.
            optimizer (torch.optim.Optimizer): The optimizer used for training.

        Returns:
            tuple: A tuple containing the train loss, train accuracy, and a dictionary
                   containing the true and predicted values.
                   - val_loss (float): The average train loss for one epoch.
                   - val_acc (float): The average train accuracy for one epoch.
                   - true_predict_list (dict): A dictionary containing two lists: "true" and "predict".
                                              The "true" list contains the true labels of the train data,
                                              and the "predict" list contains the predicted labels.        """
        self.model.train()
        train_loss, train_acc = 0, 0
        true_predict_list = {"true": [], "predict": []}

        for batch, (x, y) in enumerate(self.train_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            # train_acc += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
            true_predict_list["true"].append(y.squeeze().detach().cpu().numpy())
            true_predict_list["predict"].append(y_pred.squeeze().detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(self.train_dataloader)
        # train_acc /= len(self.train_dataloader)
        return train_loss, train_acc, true_predict_list

    def _val_step(self, loss_fn: torch.nn.Module) -> tuple:
        """
        Calculate the validation loss and accuracy for the current model.

        Parameters:
            loss_fn (torch.nn.Module): The loss function used for calculating the loss.

        Returns:
            tuple: A tuple containing the validation loss, validation accuracy, and a dictionary
                   containing the true and predicted values.
                   - val_loss (float): The average validation loss for one epoch.
                   - val_acc (float): The average validation accuracy for one epoch.
                   - true_predict_list (dict): A dictionary containing two lists: "true" and "predict".
                                              The "true" list contains the true labels of the validation data,
                                              and the "predict" list contains the predicted labels.
        """
        self.model.eval()
        val_loss, val_acc = 0, 0
        true_predict_list = {"true": [], "predict": []}

        with torch.inference_mode():
            for x, y in self.val_dataloader:
                x, y = x.to(self.device), y.to(self.device),
                y_pred = self.model(x)
                val_loss += loss_fn(y_pred, y)
                # val_acc += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
                true_predict_list["true"].append(y.detach().cpu().numpy())
                true_predict_list["predict"].append(y_pred.detach().cpu().numpy())

            val_loss /= len(self.val_dataloader)
            # val_acc /= len(self.val_dataloader)
            return val_loss, val_acc, true_predict_list

    def train(self,
              model_name: str,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              epochs_num: int,
              writer: bool = False,
              device: str = "cuda" if torch.cuda.is_available() else "cpu",
              early_stop_patience: int = None) -> dict:
        """
        Trains the model for a specified number of epochs.

        Args:
            model_name (str): The name of the model.
            loss_fn (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            epochs_num (int): The number of epochs to train.
            writer (bool, optional): If True, creates a Tensorboard writer. Defaults to False.
            device (str, optional): The device to use for training. Defaults to "cuda" if available, otherwise "cpu".
            early_stop_patience (int, optional): The number of epochs to wait for early stopping. Defaults to None, means no stop.

        Returns:
            dict: A dictionary containing training and validation loss and accuracy.

        Raises:
            EnvironmentError: If no writer is found.
        """
        best_loss = float("inf")
        early_stop = 0
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        train_true_predict_list = {"true": [], "predict": []}
        val_true_predict_list = {"true": [], "predict": []}

        try:
            if writer:
                writer = Machine_Engine._create_writer(model_name=model_name, epochs=str(epochs_num))
        except:
            raise EnvironmentError("No writer found. Please check your torch.utils.tensorboard.SummaryWriter()")

        self.device = device
        self.model.to(device)
        for epoch in tqdm(range(epochs_num)):
            train_loss, train_acc, true_predict_list = self._train_step(loss_fn=loss_fn, optimizer=optimizer)
            train_true_predict_list["true"].extend(true_predict_list["true"])
            train_true_predict_list["predict"].extend(true_predict_list["predict"])

            val_loss, val_acc, true_predict_list = self._val_step(loss_fn=loss_fn)
            val_true_predict_list["true"].extend(true_predict_list["true"])
            val_true_predict_list["predict"].extend(true_predict_list["predict"])
            print(
                f"Epoch {epoch} | train: Loss {train_loss:.6f} Accuracy {train_acc:.4f} | validation: Loss {val_loss:.6f} Accuracy {val_acc:.4f}")

            results["train_acc"].append(train_acc)
            results["train_loss"].append(train_loss.item())
            results["val_acc"].append(val_acc)
            results["val_loss"].append(val_loss.item())

            if writer:
                writer.add_scalars(main_tag="Loss",
                                   tag_scalar_dict={"Train_loss": train_loss,
                                                    "Validation_Loss": val_loss},
                                   global_step=epoch)
                writer.add_scalars(main_tag="Accuracy",
                                   tag_scalar_dict={"Train_accuracy": train_acc,
                                                    "Validation_accuracy": val_acc},
                                   global_step=epoch)
                writer.close()

            if val_loss <= best_loss:
                best_loss = val_loss
                early_stop = 0
            else:
                early_stop += 1
            if early_stop_patience and early_stop >= early_stop_patience:
                print(f"Early_Stop_at {epoch} Epoch")
                break
        self.train_true_predict_list = train_true_predict_list
        self.val_true_predict_list = val_true_predict_list
        return results

    def test(self, loss_fn: torch.nn.Module) -> tuple:
        """
        Calculate the test loss and accuracy of the model.

        Parameters:
            loss_fn (torch.nn.Module): The loss function used for calculating the test loss.

        Returns:
            tuple: A tuple containing the test loss (float), test accuracy (float), and a dictionary
            containing the true and predicted values for each test sample.
        """
        print("[!!!IMPORTANT NOTE!!!]")
        print("The test_function provided here is intended solely for the final model analysis and reporting purposes.")
        print("Please refrain from using it as a general-purpose function in your own projects. Always refer to")
        print("the appropriate train and validation data for developing and fine-tuning your own models.")
        self.model.eval()
        test_loss, test_acc = 0, 0
        true_predict_list = {"true": [], "predict": []}

        with torch.inference_mode():
            for x, y in self.test_dataloader:
                x, y = x.to(self.device), y.to(self.device),
                y_pred = self.model(x)
                test_loss += loss_fn(y_pred, y)
                # test_acc += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
                true_predict_list["true"].append(y.detach().cpu().numpy())
                true_predict_list["predict"].append(y_pred.detach().cpu().numpy())

            test_loss /= len(self.val_dataloader)
            # test_acc /= len(self.val_dataloader)
            return test_loss, test_acc, true_predict_list

    @staticmethod
    def _create_writer(model_name: str, epochs: str) -> SummaryWriter:
        """Creates torch.utils.tensorboard.writer.SummaryWriter() instance tracking to a specific directory."""
        # get timestamp of current date in reverse order : YYYY_MM_DD | datetime.now().strftime("%Y-%m-%d-%H-%M-%S") |
        timestamp = datetime.now().strftime("%Y-%m-%d-%H")
        log_dir = os.path.join("runs", model_name, f"epochs_{epochs}", timestamp)
        print(f"[INFO] create SummaryWriter saving to {log_dir}")
        return SummaryWriter(log_dir=log_dir)

    def predict(self, x: np.ndarray, main_scaler, scaler_input=None) -> pd.DataFrame:
        """
        Predicts the output for the given input using the trained model.

        Parameters:
            x (np.ndarray): The input data to be predicted.
            main_scaler: The scaler used for inverse transformation of the predicted output.
            scaler_input: The optional scaler used for scaling the input data.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted output.
        """
        from DataLoader import AOV_Dataloader
        x = pd.DataFrame(x, columns=AOV_Dataloader.features)
        if scaler_input:
            x_scaled = scaler_input.transform(x)
        else:
            x_scaled = x.values.copy()
        x_scaled = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(dim=0)
        y_scaled = self.model(x_scaled.to(self.device)).cpu().detach().numpy()

        x_predict = x.values[-1].reshape(1, -1)
        y_predict = main_scaler.inverse_transform(np.array(y_scaled).reshape(1, -1))

        predict = pd.DataFrame(np.concatenate((x_predict, y_predict), axis=1),
                               columns=AOV_Dataloader.columns)
        return predict
