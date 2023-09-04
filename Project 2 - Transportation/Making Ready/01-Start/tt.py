import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import Toolkit_01 as ml

init_data = pd.read_csv("AdmissionPredict.csv")
data = init_data.copy().dropna()
data_x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

data_train, data_test, y_train, y_test = ml.split_scale(data_x, y, test_size=0.2, random_state=1)


class MLP(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


print(data_x.shape[1])  # it is 7 here
input_size = 7
hidden1_size = 2
hidden2_size = 2
output_size = 1
num_epochs = 100
batch_size = 120

learning_rate = 0.001

temp_x = torch.Tensor(data_train)
temp_y = torch.Tensor(y_train)
train_data = torch.utils.data.TensorDataset(temp_x, temp_y)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

temp_x = torch.Tensor(data_test)
temp_y = torch.Tensor(y_test)
test_data = torch.utils.data.TensorDataset(temp_x, temp_y)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = MLP(input_size, hidden1_size, hidden2_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        output = model(inputs)
        loss = criterion(output, labels.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in test_loader:
            output = model(inputs)
            val_loss += criterion(output, labels.unsqueeze(1)).item() * inputs.size(0)
        val_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")

new_data = torch.Tensor([[340, 120, 5, 4.5, 4.5, 9.6, 1]])
prediction = model(new_data).item()
print(f"Predicted chance of admission: {prediction:.4f}")
new_data = torch.Tensor([[315, 110, 5, 4, 4, 9, 1]])
prediction = model(new_data).item()
print(f"Predicted chance of admission: {prediction:.4f}")
new_data = torch.Tensor([[290, 80, 1, 1, 1, 1, 0]])
prediction = model(new_data).item()
print(f"Predicted chance of admission: {prediction:.4f}")
new_data = torch.Tensor([[290, 80, 1, 1, 1, 1, 0]])
prediction = model(new_data).item()
print(f"Predicted chance of admission: {prediction:.4f}")
