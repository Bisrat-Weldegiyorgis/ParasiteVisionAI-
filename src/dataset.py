import json
import cv2
import torch
from torch.utils.data import Dataset

class ParasiteDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.transform = transform

        # Map parasite names to numeric labels
        parasite_names = [entry["parasite_name"] for entry in self.data]
        self.label_map = {name: idx for idx, name in enumerate(sorted(set(parasite_names)))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = entry["image_metadata"]["file_path"]
        label_name = entry["parasite_name"]
        label = self.label_map[label_name]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img, label
