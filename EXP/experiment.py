import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image






# Define the dataset class
class ProductDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        image_path, label = self.file_list[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# Define the model architecture
class ProductDetector(nn.Module):
    def __init__(self, num_classes):
        super(ProductDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Set up the training data
train_data = [
    (r"C:\Users\hp\EXP\a.jpg", 0),
    (r"C:\Users\hp\EXP\b.jpg", 1),
    (r"C:\Users\hp\EXP\c.jpg", 2),
    (r"C:\Users\hp\EXP\d.jpg", 3)
]

# Set up the image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Set up the dataset and data loader
train_dataset = ProductDataset(train_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Initialize the model
model = ProductDetector(num_classes=4)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

# Test the model
test_image = Image.open(r"C:\Users\hp\EXP\a").convert("RGB")
test_image = transform(test_image).unsqueeze(0)
model.eval()
with torch.no_grad():
    outputs = model(test_image)
    _, predicted = torch.max(outputs.data, 1)
    print("Predicted product:", predicted.item())



