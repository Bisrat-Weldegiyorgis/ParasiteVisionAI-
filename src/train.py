from torch.utils.data import DataLoader
from dataset import ParasiteDataset
import torchvision.models as models
import torch.nn as nn
import torch

# Load dataset
dataset = ParasiteDataset("data/parasites.json")
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.label_map))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
