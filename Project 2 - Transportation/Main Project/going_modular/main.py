import data_setup as dl
import Gravity_Model_engine as Js
import Neural_Network_engine as ML
import random
import numpy as np
import os
import pandas as pd
import torch
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--scale", help="if you want your scaled data through model.", action="store_true")
parser.add_argument("--new_data", help="if you want to model on new random data.", action="store_true")
parser.add_argument("--zone", help="which zone are you looking at?", type=str,
                    choices=["SiouxFalls", "Anaheim", "Chicago_Sketch", "Chicago_Regional", "GoldCoast"])
parser.add_argument("--seed", help="what is the random you want to model on it?", type=int, default=-379)
parser.add_argument("--missing_rates", help="what are the list for missing rates?", type=str, default="")
parser.add_argument("--drop_last", help="want to drop the last item of batch?", action="store_true")
parser.add_argument("-g", "--gravity_model",
                    help="write a list to specify which approach of gravity models you want to be done?", type=str,
                    default="")
parser.add_argument("-n", "--neural_network",
                    help="specify hyper parameters of neural_network in order to: hidden_units, batch_size, epochs, learning_rate",
                    type=str, default="")
parser.add_argument("--bin", help="set the scale of histogram.[start, stop, step] you only set start.", type=float,
                    default=0.5)
args = parser.parse_args()


def f(x):
    return x[0]


scale = args.scale
make_data = args.new_data
gravity_model = args.gravity_model.split(",")
if args.neural_network:
    hyper_parameters = args.neural_network.split(",")
    neural_network = True
else:
    hyper_parameters = None
    neural_network = False
zone = args.zone
mis = [float(i) for i in args.missing_rates.split(",")]
if args.seed == -379:  # default of seed
    seed = random.choice(range(1000))
else:
    seed = args.seed
bin_ = [-args.bin, args.bin + (args.bin / 10), args.bin / 10]

if scale:
    if make_data:
        data = dl.Dataloader(data_file_name=f"raw_data/{zone}/{zone}_OD.dat",
                             coordinate_file=f"raw_data/{zone}/{zone}_node.csv")
        data.scaled_data_preparation(f"data/{zone}/Scaled/random_{seed}", missing_rates=mis, random_state=seed)
    print("GMS")
    if "guess" in gravity_model:
        model_guess = Js.GM()
        model_guess.pass_data_from_folder_scaled(folder=f"data/{zone}/Scaled/random-{seed}", missing_rate=mis,
                                                 approach="guess",
                                                 scale_time_period=5,
                                                 saved_plot_folder=f"Plots/GM_plots/{zone}_plots/random_{seed}/guess",
                                                 saved_data_folder=f"Results/GM_Results/{zone}_results/random_{seed}",
                                                 bins=np.arange(float(bin_[0]), float(bin_[1]), float(bin_[2])))
    if "exp" in gravity_model:
        model_exp = Js.GM()
        model_exp.pass_data_from_folder_scaled(folder=f"data/{zone}/Scaled/random_{seed}", missing_rate=mis,
                                               approach="exp",
                                               saved_plot_folder=f"Plots/GM_plots/{zone}_plots/random_{seed}/exp",
                                               saved_data_folder=f"Results/GM_Results/{zone}_results/random_{seed}",
                                               bins=np.arange(float(bin_[0]), float(bin_[1]), float(bin_[2])))
    if "power" in gravity_model:
        model_power = Js.GM()
        model_power.pass_data_from_folder_scaled(folder=f"data/{zone}/Scaled/random_{seed}",
                                                 missing_rate=mis, approach="power",
                                                 saved_plot_folder=f"Plots/GM_plots/{zone}_plots/random_{seed}/power",
                                                 saved_data_folder=f"Results/GM_Results/{zone}_results/random_{seed}",
                                                 bins=np.arange(float(bin_[0]), float(bin_[1]), float(bin_[2])))
    if "tanner" in gravity_model:
        model_tanner = Js.GM()
        model_tanner.pass_data_from_folder_scaled(folder=f"data/{zone}/Scaled/random_{seed}", missing_rate=mis,
                                                  approach="tanner",
                                                  saved_plot_folder=f"Plots/GM_plots/{zone}_plots/random_{seed}/tanner",
                                                  saved_data_folder=f"Results/GM_Results/{zone}_results/random_{seed}",
                                                  bins=np.arange(float(bin_[0]), float(bin_[1]), float(bin_[2])))
    print("NNS")
    if neural_network:
        val_result, test_result, data_nums = [], [], []
        for iteration in mis:
            temp = 0
            folder = f"data/{zone}/Scaled/random_{seed}/{int(iteration * 100)}"
            data = ML.DataLoader_Me(folder, iteration, batch_size=int(hyper_parameters[1]), device="cpu",
                                    dr=args.drop_last)
            train, val, test = data.train, data.val, data.test

            input_size = 3
            hidden_units = int(hyper_parameters[0])
            output_size = 1
            epochs = int(hyper_parameters[2])
            learning_rate = float(hyper_parameters[3])

            model = ML.FlowPredict(input_shape=input_size, hidden_units=hidden_units, output_shape=output_size)
            count_epoch, loss_values, val_loss_values = ML.train_model(model, train, val, epochs, learning_rate)
            [val_rmse, val_mae, val_r2], data_val = ML.evaluate_model(model, val)
            [test_rmse, test_mae, test_r2], data_test = ML.test_model(model, test)
            val_result.append([np.array(val_rmse), np.array(val_mae), val_r2])
            test_result.append([np.array(test_rmse), np.array(test_mae), test_r2])

            print(f"R2 is in mis of {iteration} == {test_r2}")
            ML.plot_fn(data_test, save=True, show=False, model=f"{zone}_{seed}_{iteration}",
                       bins=np.arange(float(bin_[0]), float(bin_[1]), float(bin_[2])))

            if not os.path.exists(f"Results/NN_Results/{zone}_results/random_{seed}"):
                os.makedirs(f"Results/NN_Results/{zone}_results/random_{seed}")
            data_test = np.array(data_test)
            if not os.path.exists(f"Results/NN_Results/{zone}_results/random_{seed}/data_NN_model.csv"):
                data_test = pd.DataFrame({"Predict1.0": data_test[0].squeeze(), "Real1.0": data_test[1].squeeze()})
                data_test.to_csv(f"Results/NN_Results/{zone}_results/random_{seed}/data_NN_model.csv")
            else:
                data_last = pd.read_csv(f"Results/NN_Results/{zone}_results/random_{seed}/data_NN_model.csv")
                data_last = data_last.iloc[:, 1:]
                data_last[f"Predict{iteration*10:.1f}"] = data_test[0].squeeze()[0:len(data_last.index)]
                data_last[f"Real{iteration*10:.1f}"] = data_test[1].squeeze()[0:len(data_last.index)]
                data_last.to_csv(f"Results/NN_Results/{zone}_results/random_{seed}/data_NN_model.csv")

            with torch.inference_mode():
                plt.clf()
                plt.plot(count_epoch, loss_values, c="b", label="Train")
                plt.plot(count_epoch, val_loss_values, c="r", label="val")
                plt.legend()
                plt.show()
                if not os.path.exists(f"Models/{zone}"):
                    os.makedirs(f"Models/{zone}")
                torch.save(model, f"Models/{zone}/{zone}{seed}_{int(iteration * 100)}_model_nn.pth")

        if not os.path.exists(f"Results/NN_Results/{zone}_results/random_{seed}/NN_model.csv"):
            test_result = pd.DataFrame(test_result, columns=["RMSE", "MAE", "R^2"])
            test_result.to_csv(f"Results/NN_Results/{zone}_results/random_{seed}/NN_model.csv")
        else:
            data_last = pd.read_csv(f"Results/NN_Results/{zone}_results/random_{seed}/NN_model.csv")
            data_last = data_last.iloc[:, 1:]
            test_result = pd.DataFrame(test_result, columns=["RMSE", "MAE", "R^2"])
            test_result = pd.concat([test_result, data_last], ignore_index=True)
            test_result.to_csv(f"Results/NN_Results/{zone}_results/random_{seed}/NN_model.csv")


else:
    if make_data:
        data = dl.Dataloader(data_file_name=f"raw_data/{zone}/{zone}_OD.dat",
                             coordinate_file=f"raw_data/{zone}/{zone}_node.csv")
        data.not_scaled_data_preparation(main_folder=f"data/{zone}/Not_Scaled/random_{seed}", missing_rates=mis,
                                         random_state=seed)

    if "guess" in gravity_model:
        model_guess = Js.GM()
        model_guess.pass_data_from_folder_not_scaled(folder=f"data/{zone}/Not_Scaled/random_{seed}", missing_rate=mis,
                                                     approach="guess", scale_time_period=5,
                                                     saved_plot_folder=f"Plots/GM_plots/{zone}_plots/random_{seed}/guess",
                                                     saved_data_folder=f"Results/GM_Results/{zone}_results/random_{seed}",
                                                     )
    if "exp" in gravity_model:
        model_exp = Js.GM()
        model_exp.pass_data_from_folder_not_scaled(folder=f"data/{zone}/Not_Scaled/random_{seed}",
                                                   missing_rate=mis,
                                                   approach="exp",
                                                   saved_plot_folder=f"Plots/GM_plots/{zone}_plots/random_{seed}/exp",
                                                   saved_data_folder=f"Results/GM_Results/{zone}_results/random_{seed}")
    if "power" in gravity_model:
        model_power = Js.GM()
        model_power.pass_data_from_folder_not_scaled(folder=f"data/{zone}/Not_Scaled/random_{seed}",
                                                     missing_rate=mis,
                                                     approach="power",
                                                     saved_plot_folder=f"Plots/GM_plots/{zone}_plots/random_{seed}/power",
                                                     saved_data_folder=f"Results/GM_Results/{zone}_results/random_{seed}")
    if "tanner" in gravity_model:
        model_tanner = Js.GM()
        model_tanner.pass_data_from_folder_not_scaled(folder=f"data/{zone}/Not_Scaled/random_{seed}", missing_rate=mis,
                                                      approach="tanner",
                                                      saved_plot_folder=f"Plots/GM_plots/{zone}_plots/random_{seed}/tanner",
                                                      saved_data_folder=f"Results/GM_Results/{zone}_results/random_{seed}")

    if neural_network:
        val_result, test_result, data_nums = [], [], []
        for iteration in mis:
            temp = 0
            folder = f"data/{zone}/Not_Scaled/random_{seed}/{int(iteration * 100)}"
            data = ML.DataLoader_Me(folder, iteration, batch_size=int(hyper_parameters[1]), device="cpu",
                                    dr=args.drop_last)
            train, val, test = data.train, data.val, data.test

            input_size = 3
            hidden_units = int(hyper_parameters[0])
            output_size = 1
            epochs = int(hyper_parameters[2])
            learning_rate = float(hyper_parameters[3])

            model = ML.FlowPredict(input_shape=input_size, hidden_units=hidden_units, output_shape=output_size)
            count_epoch, loss_values, val_loss_values = ML.train_model(model, train, val, epochs, learning_rate)
            [val_rmse, val_mae, val_r2], data_val = ML.evaluate_model(model, val)
            [test_rmse, test_mae, test_r2], data_test = ML.test_model(model, test)
            val_result.append([np.array(val_rmse), np.array(val_mae), val_r2])
            test_result.append([np.array(test_rmse), np.array(test_mae), test_r2])

            print(f"R2 is in mis of {iteration} == {test_r2}")
            ML.plot_fn(data_test, save=True, show=False, model=f"{zone}_{seed}_{iteration}",
                       bins=np.arange(float(bin_[0]), float(bin_[1]), float(bin_[2])))

            if not os.path.exists(f"Results/NN_Results/{zone}_results/random_{seed}"):
                os.makedirs(f"Results/NN_Results/{zone}_results/random_{seed}")
            data_test = np.array(data_test)
            if not os.path.exists(f"Results/NN_Results/{zone}_results/random_{seed}/data_NN_model.csv"):
                data_test = pd.DataFrame({"Predict1.0": data_test[0].squeeze(), "Real1.0": data_test[1].squeeze()})
                data_test.to_csv(f"Results/NN_Results/{zone}_results/random_{seed}/data_NN_model.csv")
            else:
                data_last = pd.read_csv(f"Results/NN_Results/{zone}_results/random_{seed}/data_NN_model.csv")
                data_last = data_last.iloc[:, 1:]
                data_last[f"Predict{iteration*10:.1f}"] = data_test[0].squeeze()
                data_last[f"Real{iteration*10:.1f}"] = data_test[1].squeeze()
                data_last.to_csv(f"Results/NN_Results/{zone}_results/random_{seed}/data_NN_model.csv")

            with torch.inference_mode():
                # plt.clf()
                # plt.plot(count_epoch, loss_values, c="b", label="Train")
                # plt.plot(count_epoch, val_loss_values, c="r", label="val")
                # plt.legend()
                # plt.show()
                if not os.path.exists(f"Models/{zone}"):
                    os.makedirs(f"Models/{zone}")
                torch.save(model, f"Models/{zone}/{zone}{seed}_{int(iteration * 100)}_model_nn.pth")

        if not os.path.exists(f"Results/NN_Results/{zone}_results/random_{seed}/NN_model.csv"):
            test_result = pd.DataFrame(test_result, columns=["RMSE", "MAE", "R^2"])
            test_result.to_csv(f"Results/NN_Results/{zone}_results/random_{seed}/NN_model.csv")
        else:
            data_last = pd.read_csv(f"Results/NN_Results/{zone}_results/random_{seed}/NN_model.csv")
            data_last = data_last.iloc[:, 1:]
            test_result = pd.DataFrame(test_result, columns=["RMSE", "MAE", "R^2"])
            test_result = pd.concat([test_result, data_last], ignore_index=True)
            test_result.to_csv(f"Results/NN_Results/{zone}_results/random_{seed}/NN_model.csv")
