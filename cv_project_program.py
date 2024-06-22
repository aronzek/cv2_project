import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v2 as imageio
import os
import re
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torchvision.transforms.functional as F_Transforms
from torchvision.transforms import ConvertImageDtype

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv2 = nn.Conv2d(2048, 1, kernel_size=1)  # Reduce to single channel for binary output
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)  # Resize

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)  # Reduce to single channel
        x = self.upsample(x)  # Upsample to target size

        return {'out': x}

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

class FixationDataset(Dataset):
    def __init__(self, root_dir, image_file, fixation_file, image_transform=None, fixation_transform=None):
        self.root_dir = root_dir
        self.image_files = read_text_file(image_file)
        self.fixation_files = read_text_file(fixation_file)
        self.image_transform = image_transform
        self.fixation_transform = fixation_transform
        assert len(self.image_files) == len(self.fixation_files), "lengths of image files and fixation files do not match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)

        fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
        fix = imageio.imread(fix_name)

        if self.image_transform:
            image = self.image_transform(image)
        if self.fixation_transform:
            fix = self.fixation_transform(fix)

        sample = {"img_name": self.image_files[idx], "image": image, "fixation": fix, "raw_image": image}

        mean, std = image.mean([1, 2]), image.std([1, 2])
        transform_norm = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        sample["image"] = transform_norm(image)

        return sample

class EyeFixationTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        new_x = x
        return new_x

class Eye_Fixation_CNN(nn.Module):
    def __init__(self, resnet_model, center_bias):
        super().__init__()
        self.resnet_model = resnet_model
        self.gauss_kernel = torch.nn.Parameter(data=gaussian_kernel(25, 11.2), requires_grad=False)

        # Ensuring `center_bias` is correctly formatted
        center_bias = torch.tensor(center_bias, dtype=torch.float32).clone().detach().requires_grad_(False)
        center_bias = torch.log(center_bias)

        if center_bias.dim() == 2:  # If initially 2D, reshape to [1, C, 1, 1]
            center_bias = center_bias.unsqueeze(2).unsqueeze(3)
        center_bias = center_bias.unsqueeze(0)  # Adding batch dimension initially

        self.center_bias = nn.Parameter(center_bias, requires_grad=False)

    def forward(self, xb):
        xb = F.conv2d(xb, self.gauss_kernel, padding='same')
        output = self.resnet_model.forward(xb)
        feature_maps = output['out']

        # Ensure center_bias has compatible dimensions for broadcasting
        expanded_center_bias = F.interpolate(self.center_bias, size=feature_maps.shape[2:], mode='bilinear', align_corners=False)
        expanded_center_bias = expanded_center_bias.expand(feature_maps.size(0), -1, -1, -1)

        xb = feature_maps + expanded_center_bias

        return xb

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    else:
        device, dtype = None, None
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def gaussian_kernel(window_size, sigma):
    g = gaussian(window_size, sigma)
    kernel = torch.matmul(g.unsqueeze(-1), g.unsqueeze(-1).t())
    kernel = kernel.expand(3, 3, 25, 25)
    return kernel

def construct_fcn_with_resnet_backbone():
    return resnet50(num_classes=1000)

def read_text_file(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines

def read_center_bias():
    data = np.load(load_paths()['center_bias'])
    return torch.tensor(data)

def load_paths():
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'data')

    train_images_dir = os.path.join(data_dir, 'images', 'train')
    validation_images_dir = os.path.join(data_dir, 'images', 'validation')
    test_images_dir = os.path.join(data_dir, 'images', 'test')

    train_images_txt = os.path.join(data_dir, 'train_images.txt')
    validation_images_txt = os.path.join(data_dir, 'val_images.txt')
    test_images_txt = os.path.join(data_dir, 'test_images.txt')

    train_fixations_dir = os.path.join(data_dir, 'fixations', 'train')
    validation_fixations_dir = os.path.join(data_dir, 'fixations', 'validation')

    train_fixations_txt = os.path.join(data_dir,  'train_fixations.txt')
    validation_fixations_txt = os.path.join(data_dir,  'val_fixations.txt')

    logfile_valid = os.path.join(data_dir, 'logfile_valid')
    logfile_train = os.path.join(data_dir, 'logfile_train')

    center_bias = os.path.join(data_dir, 'center_bias_density.npy')
    
    checkpoints_path = os.path.join(data_dir, 'checkpoints')
    predictions_path = os.path.join(data_dir, 'predictions')

    paths_dict = {
        'root_dir': root_dir,
        'data_dir': data_dir,
        'train_images_dir': train_images_dir,
        'validation_images_dir': validation_images_dir,
        'test_images_dir': test_images_dir,
        'train_images_txt': train_images_txt,
        'validation_images_txt': validation_images_txt,
        'test_images_txt': test_images_txt,
        'train_fixations_dir': train_fixations_dir,
        'validation_fixations_dir': validation_fixations_dir,
        'train_fixations_txt': train_fixations_txt,
        'validation_fixations_txt': validation_fixations_txt,
        'logfile_valid': logfile_valid,
        'logfile_train': logfile_train,
        'center_bias': center_bias,
        'checkpoints_path': checkpoints_path,
        'predictions_path': predictions_path
    }

    return paths_dict

def load_data(data_type):
    image_transform = transforms.Compose([transforms.ToTensor(), EyeFixationTransform()])
    fixation_transform = transforms.Compose([transforms.ToTensor(), EyeFixationTransform()])
    paths_dict = load_paths()
    root_dir = paths_dict['data_dir']
    train_images_path = paths_dict['train_images_txt']
    validation_images_path = paths_dict['validation_images_txt']
    test_images_path = paths_dict['test_images_dir']
    train_fixations_path = paths_dict['train_fixations_txt']
    validation_fixations_path = paths_dict['validation_fixations_txt']
    logfile_valid = paths_dict['logfile_valid']
    logfile_training = paths_dict['logfile_train']

    if (data_type == "train"):
        fixation_ds = FixationDataset(root_dir, train_images_path, train_fixations_path, image_transform, fixation_transform)
    elif (data_type == "valid"):
        fixation_ds = FixationDataset(root_dir, validation_images_path, validation_fixations_path, image_transform, fixation_transform)

    samples = []
    for sample_index in range(fixation_ds.__len__()):
        samples.append(fixation_ds.__getitem__(sample_index))
        
    fixation_loader = DataLoader(fixation_ds, batch_size=16)
    
    return fixation_loader

def save_network_outputs(predictions, epoch, input):
    paths = load_paths()
    predictions_path = paths_dict['predictions_path']

    for i, pred in enumerate(predictions):
    # for i, pred in predictions:
        input_file = re.search(r'\d+', input[i]).group()
        file_path = os.path.join(predictions_path, f"prediction-{epoch}-{input_file}.png")
        pred = torch.squeeze(pred, 0)
        out = ConvertImageDtype(torch.uint8)(torch.sigmoid(pred))
        out_np = out.numpy()
        imageio.imwrite(file_path, out_np)

def log_results(logfile, epoch, train_loss, valid_loss=None):
    with open(logfile + '.log', 'a') as f:
        f.write(f"Epoch: {epoch}, Training loss: {train_loss}\n")
        if valid_loss:
            f.write(f"Epoch: {epoch}, Validation loss: {valid_loss}\n")

# Additional utility functions go here...
# Function definitions for load_paths, load_data, show, visualize_images, save_network_outputs, log_results...

# Main code
center_bias = read_center_bias()
train_data_loader = load_data("train")
valid_data_loader = load_data("valid")

resnet_model = construct_fcn_with_resnet_backbone()
eye_fixation_model = Eye_Fixation_CNN(resnet_model, center_bias)
opt = optim.SGD(eye_fixation_model.parameters(), lr=0.1)
epochs = 100
paths_dict = load_paths()
checkpoints_path = paths_dict['checkpoints_path']
logfile_validation_path = paths_dict['logfile_valid']
logfile_training_path = paths_dict['logfile_train']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eye_fixation_model.to(device)

# training loop
for epoch in range(epochs):
    eye_fixation_model.train()
    train_loss = 0

    for sample in train_data_loader:
        input_image = sample['image'].to(device)
        pred = eye_fixation_model(input_image)
        loss = F.binary_cross_entropy_with_logits(pred, sample["fixation"].to(device))
        train_loss += loss.item()

        # visualize_images(sample['image'], sample['fixation'], pred)
        # save pred images
        save_network_outputs(pred, epoch, sample['img_name'])

        loss.backward()
        opt.step()
        opt.zero_grad()

    print("Epoch:", epoch, "Training loss:", train_loss / len(train_data_loader))

    # log results
    log_results(logfile_training_path, epoch, train_loss)

    # validation
    if (epoch % 5 == 0):
        eye_fixation_model.eval()
        valid_loss = 0
        with torch.no_grad():
            for sample in valid_data_loader:
                valid_pred = eye_fixation_model(sample['image'].to(device))
                loss = F.binary_cross_entropy_with_logits(valid_pred, sample["fixation"].to(device))
                valid_loss += loss.item()

        print('Epoch:', epoch, 'Validation loss:', valid_loss / len(valid_data_loader))
        log_results(logfile_validation_path, epoch, train_loss, valid_loss)

    # save a checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': eye_fixation_model.state_dict(),
        'optimizer_state_dict': opt.state_dict()  # Corrected the typo here
    }, os.path.join(checkpoints_path, f"Eye_Fixation_CNN_epoch{epoch}.pt"))

# Additional TODO for training, logging, handling test files...
# Function for testing loop...

# Functions defined outside go here as needed...