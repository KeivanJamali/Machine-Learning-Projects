import pandas as pd

for zone in ["SiouxFalls", "Anaheim", "Chicago"]:
# for zone in ["SiouxFalls"]:
    data_GAN = pd.read_csv(f"{zone}/data_GAN.csv")
    data_GNN = pd.read_csv(f"{zone}/data_GNN.csv")

    data_GM = pd.read_csv(f"{zone}/data_GM_exp.csv")
    data_GM = data_GM[["Predict5.0", "Real5.0"]]

    data_NN = pd.read_csv(f"{zone}/data_NN_model.csv")
    data_NN = data_NN[["Predict5.0", "Real5.0"]]

    real = []
    predict_GAN = []
    predict_GNN = []
    bad = []

    for i in range(len(data_GM)):
        do = False
        real.append(data_GM.iloc[i, 1])
        for j in range(len(data_GAN)):
            if (data_GAN.iloc[j, 0]-data_GM.iloc[i, 1]) < 0.000000000000000000001:
                predict_GAN.append(data_GAN.iloc[j, 1])
                predict_GNN.append(data_GNN.iloc[j, 1])
                do = True
                data_GAN.drop(j)
                break
        if not do:
            predict_GAN.append(0)
            predict_GNN.append(0)
            print(len(bad))

            bad.append(0)

    data = pd.DataFrame(
        {"y true": real, "y pred (GM)": data_GM.iloc[:, 0], "y pred (NN)": data_NN.iloc[:, 0],
         "y pred (GCN)": predict_GNN,
         "y pred (GAT)": predict_GAN})
    data.to_csv(f"final/data_{zone}.csv")
    print(len(bad))
