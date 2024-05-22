import os
import torch
import Information

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from tqdm.auto import tqdm
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass

class Machine_Engine:
    def __init__(self, model: torch.nn.Module,
                 train_data: torch.utils.data.DataLoader,
                 val_data: torch.utils.data.DataLoader,
                 test_data: torch.utils.data.DataLoader):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.device = None
        self.prediction = None
        self.writer = None
        self.early_stop_patience = None
        self.results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.train_true_predict_list = {"input": [], "true": [], "predict": []}
        self.val_true_predict_list = {"input": [], "true": [], "predict": []}
        self.test_true_predict_list = {"input": [], "true": [], "predict": []}

    def fit_to_train(self, loss_fn: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     epochs_num: int,
                     writer: bool = False,
                     device: str = "cuda" if torch.cuda.is_available() else "cpu",
                     early_stop_patience: int = None,
                     resolution: int = 1) -> dict:
        """
        Trains the model for a specified number of epochs.

        Args:
            loss_fn (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            epochs_num (int): The number of epochs to train.
            writer (bool, optional): If True, creates a Tensorboard writer. Defaults to False.
            device (str, optional): The device to use for training. Defaults to "cuda" if available, otherwise "cpu".
            early_stop_patience (int, optional): The number of epochs to wait for early stopping. Defaults to None, means no stop.
            resolution (int, optional): epoch / resolution = every writing. Defaults to 1.

        Returns:
            dict: A dictionary containing training and validation loss and accuracy.

        Raises:
            EnvironmentError: If no writer is found.
        """

        torch.manual_seed(Information.random_state_train)
        self.device = device
        self.early_stop_patience = early_stop_patience

        self.writer = self._create_writer(Information.model_name, Information.model_architecture,
                                          Information.model_version) if writer else None
        # Train The model.
        self._train(loss_fn, optimizer, epochs_num, writer, device, resolution)

        return self.results

    def _train(self, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               epochs_num: int,
               writer: bool,
               device,
               resolution: int = 1) -> dict:

        best_loss = float("inf")
        early_stop = 0

        self.model.to(device)
        for epoch in tqdm(range(1, epochs_num + 1)):
            train_loss, train_acc, addition_train = self._train_step(loss_fn=loss_fn, optimizer=optimizer)
            val_loss, val_acc, addition_val = self._val_step(loss_fn=loss_fn)
            self._add_to_true_predict(addition_train, addition_val)
            self._add_to_results(train_loss, train_acc, val_loss, val_acc)
            self._print_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
            if writer and epoch % resolution == 0:
                self._writer_step(train_loss, train_acc, val_loss, val_acc, epoch)
            if self.early_stop_patience:
                best_loss, early_stop = self._early_stop_fn(best_loss=best_loss, loss=val_loss, early_stop=early_stop)
            # use if necessary.
            # more = self._early_stop_purpose(treshold_loss=0.001, treshold_acc=0.9, loss=val_loss, acc=val_acc)

            if not best_loss:
                break

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
                                              and the "predict" list contains the predicted labels.."""
        self.model.train()
        train_loss, train_acc = 0, 0
        true_predict_list = {"input": [], "true": [], "predict": []}

        for batch, (x, y) in enumerate(self.train_data):
            x, y = x.unsqueeze(1).to(self.device), y.unsqueeze(1).to(self.device)
            y_logit = self.model(x)
            loss = loss_fn(y_logit, y)
            train_loss += loss.item()
            train_acc += self.r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_logit.detach().cpu().numpy())
            true_predict_list["input"].append(x.detach().cpu().numpy())
            true_predict_list["true"].append(y.detach().cpu().numpy())
            true_predict_list["predict"].append(y_logit.squeeze().detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(self.train_data)
        train_acc /= len(self.train_data)
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
        true_predict_list = {"input": [], "true": [], "predict": []}

        with torch.inference_mode():
            for x, y in self.val_data:
                x, y = x.unsqueeze(1).to(self.device), y.unsqueeze(1).to(self.device)
                y_logit = self.model(x)
                val_loss += loss_fn(y_logit, y).item()
                val_acc += self.r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_logit.detach().cpu().numpy())
                true_predict_list["input"].append(x.detach().cpu().numpy())
                true_predict_list["true"].append(y.detach().cpu().numpy())
                true_predict_list["predict"].append(y_logit.squeeze().detach().cpu().numpy())

            val_loss /= len(self.val_data)
            val_acc /= len(self.val_data)
            return val_loss, val_acc, true_predict_list

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
        try:
            len(self.test_data)
        except:
            print("[INFO] There is no test in your data")
            raise ValueError
        self.model.eval()
        test_loss, test_acc = 0, 0
        true_predict_list = {"input":[], "true": [], "predict": []}

        with torch.inference_mode():
            for x, y in self.test_data:
                x, y = x.to(self.device).unsqueeze(1), y.to(self.device).unsqueeze(1)
                y_logit = self.model(x)
                test_loss += loss_fn(y_logit, y).item()
                test_acc += self.r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_logit.detach().cpu().numpy())
                true_predict_list["input"].append(x.detach().cpu().numpy())
                true_predict_list["true"].append(y.detach().cpu().numpy())
                true_predict_list["predict"].append(y_logit.detach().cpu().numpy())

            test_loss /= len(self.test_data)
            test_acc /= len(self.test_data)
            return test_loss, test_acc, true_predict_list

    def _writer_step(self, train_loss, train_acc, val_loss, val_acc, epoch):
        self.writer.add_scalars(main_tag="Loss",
                                tag_scalar_dict={"Train_loss": train_loss,
                                                 "Validation_Loss": val_loss},
                                global_step=epoch)
        self.writer.add_scalars(main_tag="Accuracy",
                                tag_scalar_dict={"Train_accuracy": train_acc,
                                                 "Validation_accuracy": val_acc},
                                global_step=epoch)
        self.writer.close()

    def _add_to_true_predict(self, addition_to_train, addition_to_val):
        self.train_true_predict_list["true"].append(addition_to_train["true"])
        self.train_true_predict_list["predict"].append(addition_to_train["predict"])
        self.train_true_predict_list["input"].append(addition_to_train["input"])
        self.val_true_predict_list["true"].append(addition_to_val["true"])
        self.val_true_predict_list["predict"].append(addition_to_val["predict"])
        self.val_true_predict_list["input"].append(addition_to_val["input"])

    def _add_to_results(self, train_loss, train_acc, val_loss, val_acc):
        self.results["train_acc"].append(train_acc)
        self.results["train_loss"].append(train_loss)
        self.results["val_acc"].append(val_acc)
        self.results["val_loss"].append(val_loss)

    def _early_stop_fn(self, best_loss, loss, early_stop):
        if loss <= best_loss:
            best_loss = loss
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= self.early_stop_patience:
            print(f"[INFO] Early Stopped!!!")
            return False, False
        return best_loss, early_stop

    def predict(self, x: np.ndarray, y_scaler=None, x_scaler=None) -> pd.DataFrame:
        """
        Predicts the output for the given data using the trained model.

        Parameters:
            x (np.ndarray): The data to be predicted.
            y_scaler: The scaler used for inverse transformation of the predicted output.
            x_scaler: The optional scaler used for scaling the data.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted output.
        """
        if x_scaler:
            x = x_scaler.transform(x)

        x = torch.tensor(x, dtype=torch.float).unsqueeze(dim=0).to(self.device)
        y_logit = self.model(x).cpu().detach().numpy()

        x_predict = x
        y_predict = y_scaler.inverse_transform(y_logit) if y_scaler else y_logit

        if self.prediction is None:
            self.prediction = pd.DataFrame(np.concatenate((x_predict, y_predict), axis=1))
        else:
            self.prediction = pd.concat([self.prediction, pd.DataFrame(np.concatenate((x_predict, y_predict), axis=1))],
                                        axis=0)

        return self.prediction

    @staticmethod
    def _create_writer(model_name: str, model_architecture: str, model_version: str):
        """Creates torch.utils.tensorboard.writer.SummaryWriter() instance tracking to a specific directory."""
        try:
            # get timestamp of current date in reverse order : YYYY_MM_DD | datetime.now().strftime("%Y-%m-%d-%H-%M-%S") |
            timestamp = datetime.now().strftime("%Y-%m-%d-%H")
            log_dir = os.path.join("runs", model_name, model_architecture, model_version, timestamp)
            print(f"[INFO] create SummaryWriter saving to {log_dir}")
            return SummaryWriter(log_dir=log_dir)

        except:
            raise EnvironmentError("No writer found. Please check your torch.utils.tensorboard.SummaryWriter()")

    @staticmethod
    def _print_epoch(epoch, train_loss, train_acc, val_loss, val_acc):
        print(
            f"Epoch {epoch} | train: Loss {train_loss:.6f} Accuracy {train_acc:.4f} | validation: Loss {val_loss:.6f} Accuracy {val_acc:.4f}")

    @staticmethod
    def _early_stop_purpose(treshold_loss, treshold_acc, loss, acc):
        if loss < treshold_loss and acc > treshold_acc:
            print(f"[INFO] _early_stop_purpose: great!!!")
            return False

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the R-squared (coefficient of determination) for a regression model.

        Args:
            y_true (np.ndarray): The true target values.
            y_pred (np.ndarray): The predicted target values.

        Returns:
            float: The R-squared value.
        """
        mean = np.mean(y_true)
        # total sum of squares
        tss = np.sum((y_true - mean) ** 2)
        # residual sum of squares
        rss = np.sum((y_true - y_pred) ** 2)
        # coefficient of determination
        r2 = 1 - (rss / tss)
        return r2

    def plot_loss(self, save=True):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.results["train_loss"])), self.results["train_loss"], label="train")
        plt.plot(range(len(self.results["val_loss"])), self.results["val_loss"], label="val")
        plt.legend()
        plt.title("Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        if save:
            plt.savefig(f"plot/{Information.model_name}_loss.png")
        plt.show()

    def plot_acc(self, save=True):
        plt.figure(figsize=(10, 5))
        v_a = self.results["val_acc"]
        v_a = [100 * i if i > 0 else 0 for i in v_a]
        t_a = self.results["train_acc"]
        t_a = [100 * i if i > 0 else 0 for i in t_a]
        plt.plot(range(len(self.result["train_acc"])), t_a, label="train")
        plt.plot(range(len(self.result["val_acc"])), v_a, label="val")
        plt.legend()
        plt.title("Accuracy")
        plt.ylabel("Accuracy(%)")
        plt.xlabel("Epoch")
        if save:
            plt.savefig(f"plot/{Information.model_name}_accuracy.png")
        plt.show()

    def plot_predict_real(self, save=True):
        y1 = [item for sublist in self.train_true_predict_list["predict"][-1] for item in sublist]
        y2 = [item for sublist in self.val_true_predict_list["predict"][-1] for item in sublist]

        r1 = [item for sublist in self.train_true_predict_list["true"][-1] for item in sublist]
        r2 = [item for sublist in self.val_true_predict_list["true"][-1] for item in sublist]

        x1 = [i for i in range(len(r1))]
        x2 = [i for i in range(len(r1), len(r1) + len(r2))]

        plt.figure(figsize=(10, 5))
        plt.scatter(x1, y1, label="Train Output", c="green", marker=".")
        plt.scatter(x2, y2, label="predicted Output", c="red", marker=".")
        plt.scatter(x1 + x2, r1 + r2, label="Actual Output", c="Blue", marker=".")
        plt.ylabel(f"{Information.model_name} Output")
        plt.xlabel("Data")
        plt.title("Actual Values VS Predicted Value")
        plt.legend()
        if save:
            plt.savefig(f"plot/{Information.model_name}.png")
        plt.show()

    def save(self):
        save_path = f"model/{Information.model_architecture}/{Information.model_name}_{Information.model_architecture}_{Information.model_version}.pt"
        torch.save(self.model.state_dict(), save_path)
