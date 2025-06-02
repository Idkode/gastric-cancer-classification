from torch.utils.data import Dataset
from PIL import Image


class StomachCancerDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.x = images
        self.y = labels
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        label = self.y[idx]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
