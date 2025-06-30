import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class UXTD_SampleDataset(Dataset):
    def __init__(self, ult_dir, txt_dir, transform=None):
        self.ult_dir = ult_dir
        self.txt_dir = txt_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(ult_dir) if f.endswith(".png") or f.endswith(".bmp")
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.ult_dir, self.image_files[idx])
        label_path = os.path.join(
            self.txt_dir,
            self.image_files[idx].replace('.png', '.txt').replace('.bmp', '.txt')
        )

        # Load ultrasound image (grayscale)
        image = Image.open(image_path).convert('L')

        # Load label from corresponding .txt file
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = int(f.read().strip().split()[0])  # First token is label
        else:
            label = 0  # fallback label if missing

        if self.transform:
            image = self.transform(image)

        return image, label
