from assignment3 import UWaveGestureLibraryDataset, u_wave_gesture_library_cnn_model
from torch.utils.data import DataLoader
import torch

def check_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            true_classes = torch.argmax(batch_y, dim=1) if batch_y.ndim > 1 else batch_y
            logits = model(batch_x)
            predicted_classes = torch.argmax(logits, dim=1)

            correct += (predicted_classes == true_classes).sum().item()
            total += true_classes.size(0)

    return correct / total



model, train_acc, val_acc = u_wave_gesture_library_cnn_model(
    "UWaveGestureLibrary_TRAIN.csv"
)

print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Validation Accuracy: {val_acc*100:.2f}%")


# test dataset
test_dataset = UWaveGestureLibraryDataset("UWaveGestureLibrary_TEST.csv")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_acc = check_accuracy(model, test_loader)

print(f"Test Accuracy: {test_acc*100:.2f}%")