import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

vectors = np.load("vectors.npy")
labels = np.load("labels.npy")
original = np.load("original.npy")
lengths = np.load("lengths.npy")
tokens = np.load("tokens.npy")
test_indices = np.load("test_indices.npy")

indices = np.arange(len(labels))
X_train, X_test, y_train, y_test, _, test_indices = train_test_split(vectors, labels, indices, test_size=0.2,
                                                                     shuffle=True, random_state=42)

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        num_layers = 1

        hidden_size = 128

        self.lstm1 = nn.LSTM(100, 128, num_layers, bidirectional=True)

        self.dropout1 = nn.Dropout(p=0.6)

        self.lstm2 = nn.LSTM(256, 128, num_layers, bidirectional=True)

        self.dropout2 = nn.Dropout(p=0.6)

        self.lstm3 = nn.LSTM(256, 128, num_layers, bidirectional=True)

        self.linear = nn.Linear(2 * 128, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)

        x = self.dropout1(x)

        x, _ = self.lstm2(x)

        x = self.dropout2(x)

        x, _ = self.lstm3(x)

        x = self.linear(x)

        return torch.sigmoid(x)


lstm_model = LSTMModel()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

writer = SummaryWriter()

num_epochs = 80

for epoch in range(num_epochs):
    lstm_model.train()
    total_loss = 0.0

    for i, (inputs_batch, targets_batch) in tqdm(enumerate(train_dataloader)):
        # if i % 10 == 0:
        #    print("step ", i)
        # Forward pass
        outputs = lstm_model(inputs_batch)

        # Compute the loss
        loss = criterion(outputs, targets_batch)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the loss for averaging later
        total_loss += loss.item()

    # Average loss for the current epoch
    average_loss = total_loss / (i + 1)

    # Print and log the average loss for the current epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
    writer.add_scalar("Loss/train", average_loss, epoch)

    lstm_model.eval()  # Set the model to evaluation mode
    total_test_loss = 0.0

    with torch.no_grad():
        for i, (inputs_batch, targets_batch) in enumerate(test_dataloader):
            # Forward pass (no need to compute gradients)
            outputs = lstm_model(inputs_batch)

            # Compute the test loss
            loss = criterion(outputs, targets_batch)

            # Accumulate the test loss for averaging later
            total_test_loss += loss.item()

        # Average test loss for the current epoch
        average_test_loss = total_test_loss / (i + 1)

        # Print and log the average test loss for the current epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {average_test_loss:.4f}")
        writer.add_scalar("Loss/test", average_test_loss, epoch)

    if (epoch % 2 == 1):
        checkpoint_filename = f"model_epoch_{epoch + 1}.pth"
        checkpoint_path = f"./checkpoints/{checkpoint_filename}"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': lstm_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': average_test_loss,
        }, checkpoint_path)

writer.close()


def compute_accuracy(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Assuming the model output is between 0 and 1
            outputs = model(inputs)
            predicted_labels = (outputs >= 0.5).float()  # Thresholding at 0.5 for binary classification
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy


accuracy = compute_accuracy(lstm_model, test_dataloader)

print(accuracy)

lengths_test = lengths[test_indices]
print(lengths_test.shape)
MAX = 50000


def get_accuracy_bin_length():
    lstm_model.eval()
    correct_predictions = 0
    total_samples = 0

    bins_f = np.zeros((50,))
    bins_t = np.zeros((50,))
    with torch.no_grad():
        i = 0
        for inputs, labels in tqdm(test_dataloader):
            if i > MAX:
                break
            # Assuming the model output is between 0 and 1
            outputs = lstm_model(inputs)
            predicted_labels = (outputs >= 0.5).float()  # Thresholding at 0.5 for binary classification
            for b in range(len(predicted_labels)):
                l = lengths_test[i] // 5
                if labels[b] != predicted_labels[b]:
                    bins_f[l] += 1
                else:
                    bins_t[l] += 1

                i += 1
    bins_accuracy = np.zeros((50,))
    for i in range(len(bins_accuracy)):
        bins_accuracy[i] = bins_t[i] / (bins_f[i] + bins_t[i])
    return bins_accuracy


# print accuracy in function of tweet length
acc = get_accuracy_bin_length()[:10]
plt.ylabel("Accuracy")
plt.xlabel("Tweet length (in number of words)")
plt.plot(np.arange(9) * 5, acc[:9])
