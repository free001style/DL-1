import csv
import os

import torch
import torchvision
from PIL import Image
from torchvision.transforms import v2

mean, std = [0.5692, 0.5448, 0.4934], [0.1823, 0.1810, 0.1854]
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda')


@torch.no_grad()
def get_predict(model, root, test_dir, load_weights=False):
    if load_weights:
        model.load_state_dict(torch.load('final_model_epoch100.pt', map_location=torch.device(device)))
    model.eval()
    test_labels = []

    test_transform = v2.Compose([
        torchvision.transforms.ToTensor(),
        v2.Normalize(mean=mean, std=std),
    ])

    for filename in os.listdir(f'{root}/{test_dir}'):
        image = Image.open(f'{root}/{test_dir}/{filename}').convert('RGB')
        input = test_transform(image).unsqueeze(0).to(device)
        logits = model(input)
        test_labels.append({'Id': filename, 'Category': logits.argmax(dim=1).item()})

    with open('labels_test.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, ('Id', 'Category'))
        dict_writer.writeheader()
        dict_writer.writerows(test_labels)
