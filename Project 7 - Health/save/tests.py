import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Air Passengers dataset
passengers_data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432]

# Normalize the data between 0 and 1
passengers_data = np.array(passengers_data)
passengers_data = (passengers_data - np.min(passengers_data)) / (np.max(passengers_data) - np.min(passengers_data))
passengers_data = passengers_data.reshape(-1, 1)

# Split the data into training and test sets
train_data = passengers_data[:90]
test_data = passengers_data[90:]

# Define a function to create data sequences and corresponding labels
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        X.append(seq)
        y.append(label)
    return torch.FloatTensor(X), torch.FloatTensor(y)

# Set the sequence length
sequence_length = 10

# Create data sequences and labels for training and test sets
train_X, train_y = create_sequences(train_data, sequence_length)
test_X, test_y = create_sequences(test_data, sequence_length)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Set the hyperparameters
input_size = 1
hidden_size = 32
output_size = 1
num_epochs = 100
learning_rate = 0.01

# Initialize the RNN model
model = RNN(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the RNN model
for epoch in range(num_epochs):
    model.train()
    outputs = model(train_X)
    loss = criterion(outputs, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Evaluate the RNN model
model.eval()
with torch.no_grad():
    train_predictions = model(train_X)
    test_predictions = model(test_X)

# Denormalizethe predicted values
train_predictions = train_predictions.squeeze().numpy() * (np.max(passengers_data) - np.min(passengers_data)) + np.min(passengers_data)
test_predictions = test_predictions.squeeze().numpy() * (np.max(passengers_data) - np.min(passengers_data)) + np.min(passengers_data)

# Denormalize the actual values
train_actual = train_y.squeeze().numpy() * (np.max(passengers_data) - np.min(passengers_data)) + np.min(passengers_data)
test_actual = test_y.squeeze().numpy() * (np.max(passengers_data) - np.min(passengers_data)) + np.min(passengers_data)

# Plot the training data
plt.figure(figsize=(10, 6))
plt.plot(train_actual, label='Actual')
plt.plot(train_predictions, label='Predicted')
plt.title('Air Passengers - Training Data')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend()
plt.show()

# Plot the test data
plt.figure(figsize=(10, 6))
plt.plot(test_actual, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.title('Air Passengers - Test Data')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend()
plt.show()
