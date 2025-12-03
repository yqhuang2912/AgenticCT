import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class DegradationDataset(Dataset):
    def __init__(self, dataset_info_file, split='train', degradation_type="ldct", severity="low", train_ratio=0.75):
        dataset_info = pd.read_csv(dataset_info_file)

        # Filter dataset based on severity if needed
        self.dataset_info = dataset_info[dataset_info[f'{degradation_type}_severities'] == severity]

        # shuffle and split the dataset
        self.dataset_info = self.dataset_info.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(len(self.dataset_info) * train_ratio)
        if split == 'train':
            self.data = self.dataset_info.iloc[:train_size]
        else:
            self.data = self.dataset_info.iloc[train_size:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_paths']
        label_path = self.data.iloc[idx]['label_paths']
        image = np.load(image_path) # shape: [1, 1, H, W]
        label = np.load(label_path)
        
        image = torch.from_numpy(image).squeeze(0).float()
        label = torch.from_numpy(label).squeeze(0).float()

        return image, label
    

if __name__ == "__main__":
    import random
    from utils.utils import get_adaptive_window, denormalize_image, visualize_images
    dataset = DegradationDataset('/data/hyq/codes/AgenticCT/src/dataset/data/ldct/dataset_info.csv', split='train')
    print(f"Dataset size: {len(dataset)}")
    idx = random.randint(0, len(dataset) - 1)
    sample_image, sample_label = dataset[idx]
    print(f"Sample image shape: {sample_image.shape}, max: {sample_image.max()}, min: {sample_image.min()}")
    print(f"Sample label shape: {sample_label.shape}, max: {sample_label.max()}, min: {sample_label.min()}")

    min_hu, max_hu = -1024, 3072
    # denormalization example
    sample_image = denormalize_image(sample_image.squeeze().numpy(), min_hu, max_hu)
    sample_label = denormalize_image(sample_label.squeeze().numpy(), min_hu, max_hu)
    print(f"Denormalized sample image max: {sample_image.max()}, min: {sample_image.min()}")
    print(f"Denormalized sample label max: {sample_label.max()}, min: {sample_label.min()}")

    vmin, vmax = get_adaptive_window(sample_image)

    visualize_images(
        [sample_image.squeeze(), sample_label.squeeze()], 
        rows=1, cols=2, savepath="visualization.png", 
        titles=['Degraded Image', 'Label Image'],
        vmin=vmin, vmax=vmax
    )
