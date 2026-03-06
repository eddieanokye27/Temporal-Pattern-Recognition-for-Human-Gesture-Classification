import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np

import os

#Q1

# PyTorch dataset for the UWaveGestureLibrary dataset
class UWaveGestureLibraryDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_filepath):
        # dataset_filepath is the full path to a file containing data

        samples_list = []
        labels_list = []

        # Read file line-by-line because data is not a true CSV
        with open(dataset_filepath, "r") as f:
            for line in f:
                line = line.strip()

                # Split features and label using colon
                x_feature, y_feature, z_feature, label_part = line.split(
                    ":",
                )
                x_vals = np.array(x_feature.split(","), dtype=np.float32)
                y_vals = np.array(y_feature.split(","), dtype=np.float32)
                z_vals = np.array(z_feature.split(","), dtype=np.float32)

                # append them along colmun axis
                # [x, y, z]
                features = np.stack([x_vals, y_vals, z_vals], axis=1)

                # Convert label to integer
                label = int(float(label_part))

                samples_list.append(features)
                labels_list.append(label)

        # Convert to tensors
        samples_list = np.array(samples_list)
        self.samples = torch.tensor(samples_list, dtype=torch.float32)
        self.labels = torch.tensor(labels_list, dtype=torch.long)

        # Convert labels from 1–8 to 0–7 if necessary
        if torch.min(self.labels) == 1:
            self.labels = self.labels - 1

        self.total_classes = len(torch.unique(self.labels))

        # Return nothing

    def __len__(self):
        # num_samples is the total number of samples in the dataset
        num_samples = len(self.samples)
        return num_samples

    def __getitem__(self, index):
        # index is the index of the sample to be retrieved

        x = self.samples[index]
        x = x.T  

        class_id = self.labels[index]
        y = torch.zeros(self.total_classes, dtype=torch.float32)
        y[class_id] = 1.0

        # x is one sample of data
        # y is the label associated with the sample
        return x, y
#Q2

# A function that creates a cnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_cnn_model(training_data_filepath):
  # training_data_filepath is the full path to a file containing the training data

  data_obj = UWaveGestureLibraryDataset(training_data_filepath)

  train_count = int(0.8 * data_obj.__len__())
  val_count = data_obj.__len__() - train_count
  train_part, val_part = random_split(data_obj, [train_count, val_count])

  train_batches = DataLoader(train_part, batch_size=32, shuffle=True)
  val_batches = DataLoader(val_part, batch_size=32, shuffle=False)

  seq_len = data_obj.samples.shape[1]
  class_count = data_obj.total_classes
  kernels =[3,3,3]

  #Define CNN architecture
  #Note: Batch normalization is used to stabilize and accelerate training by normalizing the inputs of each layer. In this model, batch normalization is applied after each convolutional layer to improve training stability and performance.
  #Observe this article if you want: https://salsabilabasalamah.medium.com/improving-training-stability-convolutional-neural-networks-and-batch-normalization-synergy-1bb6ad7a6d03
  class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        k1, k2, k3 = kernels
        self.conv1 = nn.Conv1d(3, 32, k1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, k2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, k3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        fake_tensor = torch.zeros(1, 3, seq_len)
        fake_tensor = self.pool1(F.relu(self.bn1(self.conv1(fake_tensor))))
        fake_tensor = self.pool2(F.relu(self.bn2(self.conv2(fake_tensor))))
        fake_tensor = self.pool3(F.relu(self.bn3(self.conv3(fake_tensor))))

        self.flatten_size = fake_tensor.view(1, -1).shape[1]


        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, class_count)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

  model = SimpleCNN()

  loss_function = nn.CrossEntropyLoss()
  optimiser = optim.Adam(model.parameters(), lr=0.001)

  num_epochs = 60

  for _ in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_batches:
      true_classes = torch.argmax(batch_y, dim=1)

      preds = model(batch_x)
      loss = loss_function(preds, true_classes)

      optimiser.zero_grad()
      loss.backward()
      optimiser.step()


  def check_accuracy(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      for batch_x, batch_y in loader:
        true_classes = torch.argmax(batch_y, dim=1)
        preds = model(batch_x)
        predicted_classes = torch.argmax(preds, dim=1)

        correct += (predicted_classes == true_classes).sum().item()
        total += true_classes.size(0)

    return correct / total


  training_performance = check_accuracy(train_batches)
  validation_performance = check_accuracy(val_batches)

  # model is a trained cnn model to predict which class a sequence corresponds to
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance, validation_performance


# A function that creates an rnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_rnn_model(training_data_filepath):
  data_obj = UWaveGestureLibraryDataset(training_data_filepath)

  train_count = int(0.8 * len(data_obj))
  val_count = len(data_obj) - train_count
  train_part, val_part = random_split(data_obj, [train_count, val_count])

  train_batches = DataLoader(train_part, batch_size=32, shuffle=True)
  val_batches = DataLoader(val_part, batch_size=32, shuffle=False)

  seq_len = data_obj.samples.shape[1]
  class_count = data_obj.total_classes

  class SimpleRNN(nn.Module):
    def __init__(self):
      super(SimpleRNN, self).__init__()
      self.layer1 = nn.RNN(3, 16, batch_first=True)

      #needed for consolidation
      self.output = nn.Linear(16, class_count)
    
    def forward(self, x):
      # print(x)
      x, _ = self.layer1(x)
      x = x[:, -1, :]
      x = self.output(x)

      return x

  # training_data_filepath is the full path to a file containing the training data

  # model is a trained rnn model to predict which class a sequence corresponds to
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set

  model = SimpleRNN()

  loss_function = nn.CrossEntropyLoss()
  optimiser = optim.Adam(model.parameters(), lr=0.001)

  num_epochs = 10

  for _ in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_batches:
      true_classes = torch.argmax(batch_y, dim=1)

      batch_x = batch_x.squeeze(1)
      preds = model(batch_x)

      loss = loss_function(preds, true_classes)

      optimiser.zero_grad()
      loss.backward()
      optimiser.step()


  def check_accuracy(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      for batch_x, batch_y in loader:
        true_classes = torch.argmax(batch_y, dim=1)
        preds = model(batch_x)
        predicted_classes = torch.argmax(preds, dim=1)

        correct += (predicted_classes == true_classes).sum().item()
        total += true_classes.size(0)

    return correct / total

  training_performance = check_accuracy(train_batches)
  validation_performance = check_accuracy(val_batches)

  print(training_performance)
  print(validation_performance)

  return model, training_performance, validation_performance

if __name__ == "__main__":
  # build the training file path in a cross-platform way
  train_path = os.path.join(os.getcwd(), "UWaveGestureLibrary_TRAIN.csv")

  # Call the RNN (or CNN) using the constructed path. These should only run
  # when the script is executed directly, not on import.
  u_wave_gesture_library_rnn_model(train_path)
  # u_wave_gesture_library_cnn_model(train_path)
