from _typeshed import Incomplete
from torch.utils.data import DataLoader as DataLoader, Dataset
from torchvision import transforms as transforms

class MyImageDataset(Dataset):
    root_dir: Incomplete
    transform: Incomplete
    image_name: Incomplete
    image_paths: Incomplete
    labels: Incomplete
    label_name: Incomplete
    def __init__(self, root_dir, transform: Incomplete | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx): ...
    def save_label_pare(self) -> None: ...
