import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ML_toolkit as ml
X = pd.read_csv("cars (dataset for k means).csv")
X.drop(" brand", inplace=True, axis=1)
X.replace(" ", 0, inplace=True)
X = X.astype(float)
X.replace(0, pd.NA, inplace=True)
X.dropna(inplace=True)
# print(X)

# X = ml.scale_data(X)
# X.describe()

model = ml.K_means(X, 4, 100)
model.kmeans()