import pandas as pd


def prepare(zones: list, approaches_list: list, Ml: list, rand: list, miss_rate: list) -> None:
    """
    This function will prepare your data in an Excel files.
    :param zones: name of the zones you want to make preparation on them
    :param approaches_list: what are the approaches you are forwarding to
    :param Ml: what kind of strategy you want to get data from
    :param rand: what are the seeds you get data from
    :param miss_rate: what missing rates you want to processes with
    """
    for ml in Ml:
        if ml == "NN":
            approaches = [i for i in approaches_list if i in ["model"]]
        else:
            approaches = [i for i in approaches_list if i in ["exp", "power", "tanner", "guess"]]
        for zone in zones:
            print(f"we are in {zone}")
            for approach in approaches:
                print(approach)
                data = []
                for i in rand:
                    if ml == "NN":
                        data_temp = pd.read_csv(f"Results/{ml}_Results/{zone}_results/random_{i}/{ml}_{approach}.csv")
                        data_temp = data_temp.iloc[:, 1:]
                        data.append(data_temp)
                    else:
                        data_temp = pd.read_csv(f"Results/{ml}_Results/{zone}_results/random_{i}/{ml}_{approach}.csv")
                        data_temp = data_temp.iloc[:, 1:]
                        data.append(data_temp)

                r2 = pd.DataFrame(0, index=range(len(miss_rate)), columns=["r2"])
                mae = pd.DataFrame(0, index=range(len(miss_rate)), columns=["mae"])
                rmse = pd.DataFrame(0, index=range(len(miss_rate)), columns=["rmse"])
                for i in range(len(rand)):
                    print(i)
                    rmse.iloc[:, 0] = rmse.iloc[:, 0] + data[i].iloc[:, 0]
                    mae.iloc[:, 0] = mae.iloc[:, 0] + data[i].iloc[:, 1]
                    r2.iloc[:, 0] = r2.iloc[:, 0] + data[i].iloc[:, 2]

                rmse /= len(rand)
                mae /= len(rand)
                r2 /= len(rand)
                result = pd.DataFrame({"RMSE": rmse.iloc[:, 0], "MAE": mae.iloc[:, 0], "R2": r2.iloc[:, 0]})
                result.to_csv(f"Results/{ml}_Results/{ml}_{zone}_{approach}_results.csv")


def make_excel(places: list, types: list, miss_rate: list):
    for file_name in places:
        data_rmse = pd.DataFrame(index=types, columns=range(1, len(miss_rate)+1))
        data_mae = pd.DataFrame(index=types, columns=range(1, len(miss_rate)+1))
        data_r2 = pd.DataFrame(index=types, columns=range(1, len(miss_rate)+1))
        for model in ["exp", "power"]:
            if f"GM_{model}" in types:
                data_gm = pd.read_csv(f"Results/GM_Results/GM_{file_name}_{model}_results.csv")
                data_gm = data_gm.iloc[:, 1:]
                data_rmse.loc[f"GM_{model}", range(1, len(miss_rate)+1)] = data_gm.iloc[:, 0].values
                data_mae.loc[f"GM_{model}", range(1, len(miss_rate)+1)] = data_gm.iloc[:, 1].values
                data_r2.loc[f"GM_{model}", range(1, len(miss_rate)+1)] = data_gm.iloc[:, 2].values

        if "Neural_Net" in types:
            data_nn = pd.read_csv(f"Results/NN_Results/NN_{file_name}_model_results.csv")
            data_nn = data_nn.iloc[:, 1:]
            data_rmse.loc["Neural_Net", range(1, len(miss_rate)+1)] = data_nn.iloc[:, 0].values
            data_mae.loc["Neural_Net", range(1, len(miss_rate)+1)] = data_nn.iloc[:, 1].values
            data_r2.loc["Neural_Net", range(1, len(miss_rate)+1)] = data_nn.iloc[:, 2].values

        if "Graph_Neural_Net" in types:
            pass

        excel_writer = pd.ExcelWriter(f"{file_name}.xlsx")
        data_rmse.to_excel(excel_writer, sheet_name='RMSE', index=True)
        data_mae.to_excel(excel_writer, sheet_name='MAE', index=True)
        data_r2.to_excel(excel_writer, sheet_name='R2', index=True)
        excel_writer.close()
