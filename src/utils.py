import torch

def accuracy(outputs, labels):
    """Compute accuracy for a batch."""
    _, preds = torch.max(outputs, 1)
    return (preds == labels).sum().item() / labels.size(0)

def save_checkpoint(model, path="checkpoint.pth"):
    """Save model weights to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_checkpoint(model, path="checkpoint.pth"):
    """Load model weights from a file."""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model
