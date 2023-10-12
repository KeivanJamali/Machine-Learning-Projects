from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def reg(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model


def load(folder_name, miss_rate):
    production = pd.read_csv(f"{folder_name}/production.csv")
    production.index = production.iloc[:, 0]
    production = production.iloc[:, 1:]
    attraction = pd.read_csv(f"{folder_name}/attraction.csv")
    attraction.index = attraction.iloc[:, 0]
    attraction = attraction.iloc[:, 1:]
    travel_time = pd.read_csv(f"{folder_name}/travel_time_matrix.csv")
    travel_time.index = travel_time.iloc[:, 0]
    travel_time = travel_time.iloc[:, 1:]
    train_data = pd.read_csv(f"{folder_name}/at_miss{miss_rate}_train_od_matrix.csv")
    train_data.index = train_data.iloc[:, 0]
    train_data = train_data.iloc[:, 1:]
    val_data = pd.read_csv(f"{folder_name}/at_miss{miss_rate}_val_od_matrix.csv", low_memory=False)
    val_data.index = val_data.iloc[:, 0]
    val_data = val_data.iloc[:, 1:]
    test_data = pd.read_csv(f"{folder_name}/at_miss{miss_rate}_test_od_matrix.csv", low_memory=False)
    test_data.index = test_data.iloc[:, 0]
    test_data = test_data.iloc[:, 1:]

    index_len = len(travel_time.index) * len(travel_time.columns)
    attraction = pd.concat([attraction] * len(travel_time.index), axis=0)
    attraction = attraction.reset_index(drop=True)

    production = pd.DataFrame(np.repeat(production.values, len(travel_time.columns), axis=0),
                              columns=production.columns)
    production = production.reset_index(drop=True)
    data = pd.DataFrame({"travel_time": travel_time.values.reshape(index_len),
                         "production": production.iloc[:, 0],
                         "attraction": attraction.iloc[:, 0]}, index=range(index_len))

    y_train = pd.DataFrame(train_data.values.reshape(index_len), index=range(index_len))
    mask_train = ~y_train.isin(["False", "No_connection"])
    test = mask_train
    y_train = y_train[mask_train]
    y_train.dropna(inplace=True)

    y_val = pd.DataFrame(val_data.values.reshape(index_len), index=range(index_len))
    mask_val = ~y_val.isin(["False", "No_connection"])
    y_val = y_val[mask_val]
    y_val.dropna(inplace=True)

    y_test = pd.DataFrame(test_data.values.reshape(index_len), index=range(index_len))
    mask_test = ~y_test.isin(["False", "No_connection"])
    y_test = y_test[mask_test]
    y_test.dropna(inplace=True)

    mask_train = pd.DataFrame({"travel_time": mask_train.iloc[:, 0], "production": mask_train.iloc[:, 0],
                               "attraction": mask_train.iloc[:, 0]})
    x_train = data[mask_train]
    x_train.dropna(inplace=True)

    mask_val = pd.DataFrame(
        {"travel_time": mask_val.iloc[:, 0], "production": mask_val.iloc[:, 0], "attraction": mask_val.iloc[:, 0]})
    x_val = data[mask_val]
    x_val.dropna(inplace=True)

    mask_test = pd.DataFrame({"travel_time": mask_test.iloc[:, 0], "production": mask_test.iloc[:, 0],
                              "attraction": mask_test.iloc[:, 0]})
    x_test = data[mask_test]
    x_test.dropna(inplace=True)

    return x_train, y_train, x_val, y_val, x_test, y_test


def testit(model, x, y):
    y_pred = model.predict(x)
    r2 = r2_score(y_pred, y)
    return r2
