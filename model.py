import torch
import torch.nn as nn
from torchvision.models import resnet18

def build_model(num_classes):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_trained_model(model_path, device):

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    state_dict = checkpoint["model_state"]
    label_map = checkpoint["label_map"]

    model = build_model(len(label_map))
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, label_map
