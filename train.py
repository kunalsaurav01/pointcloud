import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# -------------------- 1. Define Image Transformations --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] range
])

# -------------------- 2. Load the Large Dataset from Folder --------------------
data_dir = r"C:\Users\Kunal Saurav\Downloads\archive (5)\MY_data"  # Change this to your actual dataset folder

# Automatically assigns labels based on folder names
train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

# DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Print dataset info
print(f"Total Training Images: {len(train_dataset)}")
print(f"Total Test Images: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")


# -------------------- 3. Define a Simple CNN Model --------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjust based on image size
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------- 4. Train the Model --------------------
def train_model(model, train_loader, epochs=5, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}")


# -------------------- 5. Test the Model --------------------
def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# -------------------- 6. Run Training & Testing --------------------
if __name__ == "__main__":
    num_classes = len(train_dataset.classes)
    model = SimpleCNN(num_classes)

    print("Training the model...")
    train_model(model, train_loader, epochs=10, learning_rate=0.001)

    print("Testing the model...")
    test_model(model, test_loader)
