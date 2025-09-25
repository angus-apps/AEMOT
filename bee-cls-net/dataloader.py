import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class BeeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label_dir in ['false', 'true']:
            full_path = os.path.join(root_dir, label_dir)
            for img_name in os.listdir(full_path):
                # Get bee label
                bee_label = 0 if label_dir == 'false' else 1

                # Get velocity values
                img_name_split = img_name[:-4].split("_")
                vx = float(img_name_split[2])
                vy = float(img_name_split[3])

                # Store values
                self.image_paths.append(os.path.join(full_path, img_name))
                self.labels.append(torch.tensor([bee_label, vx, vy]))
                

                # self.image_paths.append(os.path.join(full_path, img_name))
                # # Assign label 0 for 'false' (not a bee) and 1 for 'true' (bee)
                # self.labels.append(0 if label_dir == 'false' else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])


