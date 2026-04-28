import torchvision.models as models
import torch.nn as nn

def build_model(num_classes, pretrained=True):
    """Build a ResNet18 model with custom output layer."""
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
