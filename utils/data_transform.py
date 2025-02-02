from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from typing import Tuple

class DataTransforms:
    def __init__(self, dataset_path: str, batch_size: int):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        dataset = datasets.ImageFolder(root=dataset_path, transform=transforms.ToTensor())
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.mean, self.std = self.calculate_mean_std()
        self.transform = self.get_transforms()

    def calculate_mean_std(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        mean = 0
        std = 0
        total_images = 0
        for images, _ in self.dataloader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_samples
        mean /= total_images
        std /= total_images
        return mean.numpy(), std.numpy()

    def get_transforms(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def get_transformed_dataloader(self):
        dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
