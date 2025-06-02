from torch.utils.data import Dataset, DataLoader

def build_loader(dataset: Dataset,
                 batch_size: int = 2,
                 num_workers: int = 0):
    
    return DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2
    )