import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image
import os

SPLIT_RANDOM_SEED = 42


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_dirname, labels, transform=None):
        super().__init__()
        self._root = root
        self._image_dirname = image_dirname

        self._transform = transform
        if self._transform is None:
            self._transform = v2.Compose([
                torchvision.transforms.ToTensor()])

        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, item):
        label, filename = self._labels.iloc[item]['Category'], self._labels.iloc[item]['Id']
        image = Image.open(os.path.join(self._root, self._image_dirname, filename)).convert('RGB')
        image = self._transform(image)
        return image, label


def get_loaders(root, image_dirname, labels, train_transform, test_size, batch_size):
    # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    # dataset = ImageDataset(root=root, image_dirname=train_dir, labels=labels)
    # loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())
    # N_CHANNELS = 3
    # mean = torch.zeros(N_CHANNELS)
    # std = torch.zeros(N_CHANNELS)
    # for inputs, _labels in tqdm(loader):
    #     for i in range(N_CHANNELS):
    #         mean[i] += inputs[:,i,:,:].mean()
    #         std[i] += inputs[:,i,:,:].std()
    # mean.div_(len(dataset))
    # std.div_(len(dataset))
    mean, std = [0.5692, 0.5448, 0.4934], [0.1823, 0.1810, 0.1854]
    test_transform = v2.Compose([
        torchvision.transforms.ToTensor(),
        v2.Normalize(mean=mean, std=std)])
    train, val = torch.utils.data.random_split(range(len(labels)), [1 - test_size, test_size],
                                               torch.Generator().manual_seed(SPLIT_RANDOM_SEED))
    train_split = ImageDataset(root, image_dirname, labels.iloc[train.indices], transform=train_transform)
    val_split = ImageDataset(root, image_dirname, labels.iloc[val.indices], transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
