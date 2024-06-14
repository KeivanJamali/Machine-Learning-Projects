import os
import torch
import time
import Information
import DataLoader
import Metrics

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from tqdm.auto import tqdm
from datetime import datetime
from IPython.display import clear_output

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass


class Machine_Engine:
    def __init__(self, model: torch.nn.Module,
                 dataloader: DataLoader):
        self.model = model
        self.dataloader = dataloader
        self.scalars = self.dataloader.scaler_y
        self.step_to_print = None
        self.device = None
        self.prediction = None
        self.writer = None
        self.early_stop_patience = None
        self.virtualize = None
        self.results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.train_true_predict_list = {"input": [], "true": [], "predict": []}
        self.val_true_predict_list = {"input": [], "true": [], "predict": []}
        self.test_true_predict_list = {"input": [], "true": [], "predict": []}

    def setting(self, writer, epoch, resolution, train_loss, train_acc, val_loss, val_acc, best_loss,
                early_stop) -> bool:
        if writer and epoch % resolution == 0:
            self._writer_step(train_loss, train_acc, val_loss, val_acc, epoch)
        if self.early_stop_patience:
            best_loss, early_stop = self._early_stop_fn(best_loss=best_loss, loss=train_loss, early_stop=early_stop)

        if epoch % self.step_to_print == 0:
            if self.virtualize:
                # time.sleep(0.5)
                clear_output(wait=True)
                self.plot_predict_real(False, train_loss, train_acc, val_loss, val_acc, epoch - 1)
                plt.clf()
            else:
                self._print_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

        # use if necessary. 94_600
        more = self._early_stop_purpose(treshold_loss=98_500, treshold_acc=0.6, loss=val_loss, acc=train_acc)
        if not more:
            self.plot_predict_real(False, train_loss, train_acc, val_loss, val_acc, epoch - 1)
            self._print_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
            return False

        if not best_loss:
            return False

        return True

    def fit(self, loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            #  optimizer1: torch.optim.Optimizer,
            epochs_num: int,
            writer: bool = False,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            early_stop_patience: int = None,
            resolution: int = 1,
            virtualize: bool = False,
            step_to_print: int = 100) -> dict:
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
            virtualize (bool, optional): show plots instead of numbers. Default is False.
            step_to_print (int, optional): clear the interval between each print or plot. Default is 100 interval.

        Returns:
            dict: A dictionary containing training and validation loss and accuracy.

        Raises:
            EnvironmentError: If no writer is found.
        """

        torch.manual_seed(Information.random_state_train)
        self.device = device
        self.early_stop_patience = early_stop_patience
        self.virtualize = virtualize
        self.step_to_print = step_to_print

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
            val_loss, val_acc, addition_val = self._val_step()
            self._add_to_true_predict(addition_train, addition_val)
            self._add_to_results(train_loss, train_acc, val_loss, val_acc)
            check = self.setting(writer=writer,
                                 epoch=epoch,
                                 resolution=resolution,
                                 train_loss=train_loss,
                                 train_acc=train_acc,
                                 val_loss=val_loss,
                                 val_acc=val_acc,
                                 best_loss=best_loss,
                                 early_stop=early_stop)
            if not check:
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
        for batch, (x, y) in enumerate(self.dataloader.train_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            y_logit = self.model(x)
            loss = loss_fn(y_logit, y)

            for i in range(len(self.scalars) - 1, -1, -1):
                y_logit_inv = self._inverse_scale(y_logit, self.scalars[i], "torch")
                y_inv = self._inverse_scale(y, self.scalars[i], "torch")

            loss_real = Metrics.MSE(y_logit_inv, y_inv, mode="numpy")
            train_loss += loss_real
            train_acc += Metrics.r2_score(y_pred=y_logit, y_true=y, mode="torch").item()
            true_predict_list["input"].append(x.detach().cpu().numpy())
            true_predict_list["true"].append(y.detach().cpu().numpy())
            true_predict_list["predict"].append(y_logit.squeeze().detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(self.dataloader.train_dataloader)
        train_acc /= len(self.dataloader.train_dataloader)
        return train_loss, train_acc, true_predict_list

    def _val_step(self) -> tuple:
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
            for x, y in self.dataloader.val_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_logit = self.model(x)

                for i in range(len(self.scalars) - 1, -1, -1):
                    y_logit_inv = self._inverse_scale(y_logit, self.scalars[i], "torch")
                    y_inv = self._inverse_scale(y, self.scalars[i], "torch")

                val_loss += Metrics.MSE(y_logit_inv, y_inv, mode="numpy")
                val_acc += Metrics.r2_score(y_pred=y_logit, y_true=y, mode="torch").item()
                true_predict_list["input"].append(x.detach().cpu().numpy())
                true_predict_list["true"].append(y.detach().cpu().numpy())
                true_predict_list["predict"].append(y_logit.squeeze().detach().cpu().numpy())

            val_loss /= len(self.dataloader.val_dataloader)
            val_acc /= len(self.dataloader.val_dataloader)
            return val_loss, val_acc, true_predict_list

    def test(self) -> tuple:
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
            len(self.dataloader.test_dataloader)
        except:
            print("[INFO] There is no test in your data")
            raise ValueError
        self.model.eval()
        test_loss, test_acc = 0, 0
        true_predict_list = {"input": [], "true": [], "predict": []}

        with torch.inference_mode():
            for x, y in self.dataloader.test_dataloader:
                x, y = x.to(self.device).unsqueeze(1), y.to(self.device).unsqueeze(1)
                y_logit = self.model(x) # Wrong. We have input data here that does not exist in real life.
                test_loss += Metrics.MSE(y_logit, y, mode="numpy")
                test_acc += Metrics.r2_score(y_pred=y_logit.detach().cpu().numpy(), y_true=y.detach().cpu().numpy(),
                                             mode="numpy")
                true_predict_list["input"].append(x.detach().cpu().numpy())
                true_predict_list["true"].append(y.detach().cpu().numpy())
                true_predict_list["predict"].append(y_logit.detach().cpu().numpy())

            test_loss /= len(self.dataloader.test_dataloader)
            test_acc /= len(self.dataloader.test_dataloader)
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

    def predict(self, x: np.ndarray, y_scaler, x_scaler=None, n: int = 1, p: int = 0) -> pd.DataFrame:
        """
        Predicts the output for the given data using the trained model.

        Parameters:
            x (np.ndarray): The data to be predicted.
            y_scaler: The scaler used for inverse transformation of the predicted output.
            x_scaler: The optional scaler used for scaling the data.
            n: how many prediction do you want?
            p: only parameter for function.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted output.
        """
        if p == n:
            return self.prediction
        x = pd.DataFrame(x, columns=Information.features)
        if x_scaler:
            x_scaled = x_scaler.transform(x)
        else:
            x_scaled = x.values.copy()
        x_scaled = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(dim=0)
        x1 = x_scaled[:, :, 0].unsqueeze(dim=2)
        y_logit = self.model(x1)
        x2 = torch.cat((y_logit, x_scaled[:, -1, 1:]), dim=1)
        y_scaled = self.model1(x2.to(self.device)).cpu().detach().numpy()
        y_scaled = pd.DataFrame(np.array(y_scaled).reshape(1, -1), columns=Information.target)

        x_predict = x.values[-1].reshape(1, -1)
        y_predict = y_scaler.inverse_transform(y_scaled)
        if self.prediction is None:
            self.prediction = pd.DataFrame(np.concatenate((x_predict, y_predict), axis=1),
                                           columns=Information.columns)
        else:
            self.prediction = pd.concat([self.prediction, pd.DataFrame(np.concatenate((x_predict, y_predict), axis=1),
                                                                       columns=Information.columns)], axis=0)
        new_x = np.concatenate((x.values[1:], np.concatenate((y_predict, x_predict[:, 1:]), axis=1)), axis=0)
        self.predict(new_x, y_scaler, x_scaler, n, p + 1)
        return self.predict

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
        plt.plot(range(len(self.results["train_acc"])), t_a, label="train")
        plt.plot(range(len(self.results["val_acc"])), v_a, label="val")
        plt.legend()
        plt.title("Accuracy")
        plt.ylabel("Accuracy(%)")
        plt.xlabel("Epoch")
        if save:
            plt.savefig(f"plot/{Information.model_name}_accuracy.png")
        plt.show()

    def plot_predict_real(self, save, train_loss, train_acc, val_lss, val_acc, epoch):
        y1 = [item for sublist in self.train_true_predict_list["predict"][epoch] for item in sublist]
        y2 = [item for sublist in self.val_true_predict_list["predict"][epoch] for item in sublist]

        r1 = [item for sublist in self.train_true_predict_list["true"][epoch] for item in sublist]
        r2 = [item for sublist in self.val_true_predict_list["true"][epoch] for item in sublist]

        x1 = self.dataloader.train_data.index[Information.sequence:]
        x2 = self.dataloader.val_data.index[Information.sequence:]

        plt.figure(figsize=(10, 5))
        plt.plot(x1, y1, label="Train Output", c="green", marker=".")
        plt.plot(x2, y2, label="predicted Output", c="red", marker=".")
        plt.plot(x1.append(x2), r1 + r2, label="Actual Output", c="Blue", marker=".")
        plt.xticks(ticks=x1.append(x2)[::3], rotation=45)
        plt.ylabel(f"{Information.model_name} Output")
        plt.xlabel("Data")
        plt.title(f"Train:{train_loss:.4f}|{train_acc:.2f}, Val:{val_lss:.4f}|{val_acc:.2f},  epoch:{epoch + 1}")
        plt.legend()
        if save:
            plt.savefig(f"plot/{Information.model_name}.png")
        plt.show()

    def save(self):
        save_path = f"model/{Information.model_architecture}/{Information.model_name}_{Information.model_architecture}_{Information.model_version}.pt"
        torch.save(self.model.state_dict(), save_path)

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
    def _print_epoch(epoch, train_loss, train_acc, val_loss, val_acc) -> None:
        print(
            f"Epoch {epoch} | train: Loss {train_loss:.6f} Accuracy {train_acc:.4f} | validation: Loss {val_loss:.6f} Accuracy {val_acc:.4f}")

    @staticmethod
    def _early_stop_purpose(treshold_loss, treshold_acc, loss, acc) -> bool:
        if loss < treshold_loss and acc > treshold_acc:
            print(f"[INFO] _early_stop_purpose: great!!!")
            return False
        return True

    @staticmethod
    def _inverse_scale(x, scaler, mode):
        if mode == "torch":
            return scaler.inverse_transform(x.detach().cpu().numpy())
        elif mode == "numpy":
            return scaler.inverse_transform(x)

    def adjusted_r2_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n = torch.tensor(len(y_true))
        k = 0
        k = torch.sum(torch.tensor([torch.numel(params) for params in self.model.parameters()]))
        r2 = 1 - (1 - self.r2_score(y_pred=y_pred, y_true=y_true)) * (n - 1) / (n - k - 1)
        return r2
