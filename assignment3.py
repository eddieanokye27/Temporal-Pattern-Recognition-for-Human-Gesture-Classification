import torch
import numpy as np

#Q1

# PyTorch dataset for the UWaveGestureLibrary dataset
class UWaveGestureLibraryDataset(torch.utils.data.Dataset):

  def __init__(self, dataset_filepath):
    # dataset_filepath is the full path to a file containing data

    raw_data = np.loadtxt(dataset_filepath, delimiter=',')

    self.samples = torch.tensor(raw_data[:, :-1], dtype=torch.float32)
    self.labels = torch.tensor(raw_data[:, -1], dtype=torch.long)

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
    
    x = self.samples[index].unsqueeze(0)

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

  import torch.nn as nn
  import torch.optim as optim
  from torch.utils.data import DataLoader, random_split

  data_obj = UWaveGestureLibraryDataset(training_data_filepath)

  train_count = int(0.8 * len(data_obj))
  val_count = len(data_obj) - train_count
  train_part, val_part = random_split(data_obj, [train_count, val_count])

  train_batches = DataLoader(train_part, batch_size=32, shuffle=True)
  val_batches = DataLoader(val_part, batch_size=32, shuffle=False)

  seq_len = data_obj.samples.shape[1]
  class_count = data_obj.total_classes


  class SimpleCNN(nn.Module):
    def __init__(self):
      super(SimpleCNN, self).__init__()

      self.layer1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
      self.pool1 = nn.MaxPool1d(2)

      self.layer2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
      self.pool2 = nn.MaxPool1d(2)

      new_len = seq_len // 4

      self.hidden = nn.Linear(32 * new_len, 128)
      self.output = nn.Linear(128, class_count)

    def forward(self, x):
      x = torch.relu(self.layer1(x))
      x = self.pool1(x)

      x = torch.relu(self.layer2(x))
      x = self.pool2(x)

      x = x.view(x.size(0), -1)
      x = torch.relu(self.hidden(x))
      x = self.output(x)

      return x


  model = SimpleCNN()

  loss_function = nn.CrossEntropyLoss()
  optimiser = optim.Adam(model.parameters(), lr=0.001)

  num_epochs = 10

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
  # training_data_filepath is the full path to a file containing the training data

  # model is a trained rnn model to predict which class a sequence corresponds to
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance, validation_performance
