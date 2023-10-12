import pandas as pd
import os

if not os.path.exists("NN_Results"):
    os.makedirs("NN_Results")

names = ["Anaheim", "SiouxFall", "Chicago_Sketch"]

approaches = ["model"]
for name in names:
    for approach in approaches:
        data = []
        for i in range(10):
            data_temp = pd.read_csv(f"data/Result{i + 1}/{name}_data/NN_{approach}.csv")
            data_temp = data_temp.iloc[:, 1:]
            data.append(data_temp)

        r2 = pd.DataFrame(0, index=range(9), columns=["r2"])
        mae = pd.DataFrame(0, index=range(9), columns=["mae"])
        rmse = pd.DataFrame(0, index=range(9), columns=["rmse"])
        for i in range(10):
            print(i)
            rmse.iloc[:, 0] = rmse.iloc[:, 0] + data[i].iloc[:, 0]
            mae.iloc[:, 0] = mae.iloc[:, 0] + data[i].iloc[:, 1]
            r2.iloc[:, 0] = r2.iloc[:, 0] + data[i].iloc[:, 2]

        rmse /= 10
        mae /= 10
        r2 /= 10
        result = pd.DataFrame({"RMSE": rmse.iloc[:, 0], "MAE": mae.iloc[:, 0], "R2": r2.iloc[:, 0]})
        result.to_csv(f"NN_Results/NN_{name}_{approach}_results.csv")
