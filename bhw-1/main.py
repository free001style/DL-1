import torch
import torchvision
from torchvision.transforms import v2
import pandas as pd
import warnings

from loader import get_loaders
from model import model
from trainer import train
from tester import get_predict

warnings.filterwarnings('ignore')
mean, std = [0.5692, 0.5448, 0.4934], [0.1823, 0.1810, 0.1854]
root = 'dataset'
train_dir = 'trainval'
test_dir = 'test'
labels = pd.read_csv(f'{root}/labels.csv')
NUM_CLASSES = labels['Category'].nunique()
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda')

model = model(NUM_CLASSES)
model.to(device)


def train_model():
    train_transform_1 = v2.Compose([
        v2.RandomChoice([v2.RandomHorizontalFlip(0.5), v2.RandomVerticalFlip(0.5)], p=[0.5, 0.5]),
        v2.RandomGrayscale(p=0.5),
        v2.RandomApply([
            v2.GaussianBlur(kernel_size=5)], p=0.5),
        torchvision.transforms.ToTensor(),
        v2.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomErasing(scale=(0.1, 0.25), ratio=(1.0, 1.0)),
    ])
    batch_size = 200
    train_loader, val_loader = get_loaders(root=root, image_dirname=train_dir, labels=labels,
                                           train_transform=train_transform_1, test_size=0.01, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=1, patience=7)
    num_epochs = 100
    train(model, "first_phase", optimizer, scheduler, criterion, train_loader, val_loader, num_epochs)
    train_transform_2 = v2.Compose([
        v2.RandomHorizontalFlip(0.5),
        v2.RandAugment(),
        torchvision.transforms.ToTensor(),
        v2.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomErasing(scale=(0.1, 0.25), ratio=(1.0, 1.0)),
    ])
    train_loader, val_loader = get_loaders(root=root, image_dirname=train_dir, labels=labels,
                                           train_transform=train_transform_2, test_size=0.01, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=1, patience=7)
    num_epochs = 50
    train(model, "second_phase", optimizer, scheduler, criterion, train_loader, val_loader, num_epochs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
    train(model, "third_phase", optimizer, scheduler, criterion, train_loader, val_loader, 1)


def test_model(load_weights=False):
    get_predict(model, root, test_dir, load_weights=load_weights)


if __name__ == '__main__':
    train_model()
    test_model()
