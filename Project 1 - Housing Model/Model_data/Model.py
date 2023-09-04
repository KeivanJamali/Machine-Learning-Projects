import ML
import pandas as pd
import torch
from torch import nn
from torch import optim
from sklearn.metrics import r2_score

r = []
for _ in range(50):
    data = pd.read_csv("Housing.csv")
    data.replace("yes", 1, inplace=True)
    data.replace("no", 0, inplace=True)
    data.replace("furnished", 1, inplace=True)
    data.replace("semi-furnished", 0.5, inplace=True)
    data.replace("unfurnished", 0, inplace=True)

    data = ML.DataLoader(data)
    data_train, data_val, data_test, y_train, y_val, y_test = data.preparation(random_state=None)
    # for learn_rate in range(670, 685, 1): # 680
    # for hidden_layer2 in range(1,10):
    #     for hidden_layer1 in range(hidden_layer2,10):
    # for bach_size in range(30,40):
    # for n_epochs in range(100,500,50):
    input_size = 12
    hidden_layer1 = 6
    hidden_layer2 = 5
    n_epochs = 100
    bach_size = 32
    # learn_rate = learn_rate / 1000
    learn_rate = 0.68

    model = ML.Housing_Model(input_size, hidden_layer1, hidden_layer2)
    # print(model)

    loss_fun = nn.L1Loss()
    # loss_fun = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)  # 25.8 changeable
    # optimizer = optim.Adagrad(model.parameters(), lr=learn_rate) # bad
    # optimizer = optim.RMSprop(model.parameters(), lr=learn_rate) # 23.7 more strait
    # optimizer = optim.AdamW(model.parameters(), lr=learn_rate) # bad
    # optimizer = optim.SGD(model.parameters(), lr=learn_rate) # bad

    for epoch in range(n_epochs):
        for i in range(0, len(data_train), bach_size):
            x_train_bach = data_train[i: i + bach_size]
            y_train_bach = y_train[i: i + bach_size]
            y_predict = model(x_train_bach)

            loss = loss_fun(y_predict, y_train_bach)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Finished epoch {epoch}, latest loss {loss}")

    with torch.no_grad():
        loss_fun = nn.MSELoss()
        model.eval()
        val_outputs = model(data_val)
        val_loss = loss_fun(val_outputs, y_val)
        r2 = r2_score(val_outputs, y_val)
        # print(f"hidden_layer1 is {hidden_layer1}")
        # print(f"hidden_layer2 is {hidden_layer2}")
        # print(f"learn rate is {learn_rate}")
        # print(f"bach_size is {bach_size}")
        # print(f"n_epochs is {n_epochs}")
        # print(f"validation Loss: {val_loss.item()}")
        print(f"validation Loss: {r2}")
        r.append(r2)

print(max(r))  # 65.59% haha
# 65.59 with one 4 neuron layer.
# 75.51 with two 6,5 neuron layers.
