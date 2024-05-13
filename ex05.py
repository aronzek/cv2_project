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
from torchvision.models.segmentation import fcn_resnet50
from torchvision.utils import make_grid
import torchvision.transforms.functional as F_Transforms
from torchvision.transforms import ConvertImageDtype


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
        
        mean, std = image.mean([1,2]), image.std([1,2])
        transform_norm = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        sample["image"] = transform_norm(image)

        return sample

class EyeFixationTransform:
    def __init__(self):
        # initialize any properties if necessary
        pass
    def __call__(self, x):
        # do something to get new_x
        new_x = x
        return new_x
        pass

class Eye_Fixation_CNN(nn.Module):
    def __init__(self, resnet_model, center_bias):
        super().__init__()
        self.resnet_model = resnet_model
        self.gauss_kernel = torch.nn.Parameter(data=gaussian_kernel(25, 11.2), requires_grad=False)
        self.center_bias = torch.nn.Parameter(data=torch.log(center_bias), requires_grad=False)
    def forward(self, xb):
        
        xb = F.conv2d(xb, self.gauss_kernel, padding='same')
        xb = self.resnet_model.forward(xb)
        xb = xb['out']+self.center_bias
        
        return xb

def construct_fcn_with_resnet_backbone():
    resnet_model = fcn_resnet50(pretrained=False, pretrained_backbone=True, num_classes=1)
    for param in resnet_model.backbone.parameters():
        param.requires_grad = False
    return resnet_model

def read_text_file(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file: 
            line = line.strip() #or some other preprocessing
            lines.append(line)
    return lines

def load_data(data_type):
    image_transform = transforms.Compose([transforms.ToTensor(), EyeFixationTransform()])
    fixation_transform = transforms.Compose([transforms.ToTensor(), EyeFixationTransform()])
    paths_dict = load_paths()
    root_dir = paths_dict['root_dir']
    train_images_path = paths_dict['train_images_dir']
    validation_images_path = paths_dict['validation_images_dir']
    test_images_path = paths_dict['test_images_dir']
    train_fixations_path = paths_dict['train_fixations_dir']
    validation_fixations_path = paths_dict['validation_fixations_dir']
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

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
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

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F_Transforms.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
def visualize_images(inputs, fixations, predictions):
    fixations_grid = make_grid(fixations)
    show(fixations_grid)
    pred_normalized = torch.sigmoid(predictions)
    predictions_grid = make_grid(pred_normalized)
    show(predictions_grid)

def save_network_outputs(predictions, epoch, input):
    paths = load_paths()
    predictions_path = paths['predictions']

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


center_bias = read_center_bias()
train_data_loader = load_data("train")
valid_data_loader = load_data('valid')
#test_data_loader = load_data("test")
resnet_model = construct_fcn_with_resnet_backbone()
eye_fixation_model = Eye_Fixation_CNN(resnet_model, center_bias)
opt = optim.SGD(eye_fixation_model.parameters(), lr=0.1)
epochs = 100
paths_dict = load_paths()
checkpoints_path = paths_dict['checkpoints']
logfile_validation_path = paths_dict['logfile_valid']
logfile_training_path = paths_dict['logfile_train']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eye_fixation_model.to(device)

# training loop
for epoch in range(epochs):
    eye_fixation_model.train()
    train_loss = 0

    for sample in train_data_loader:
        input_image = sample['image']
        pred = eye_fixation_model(input_image)
        loss =  F.binary_cross_entropy_with_logits(pred, sample["fixation"])
        train_loss += loss
        
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
    if (epoch%5 == 0):
        eye_fixation_model.eval()
        with torch.no_grad():
            # use a binary cross entropy (BCE) loss for eye fixation prediction
            # loss = F.binary_cross_entropy_with_logits(model(sample["image"]), sample["fixation"])
            valid_loss = sum(F.binary_cross_entropy_with_logits(eye_fixation_model(sample['image']), sample['fixation']) for sample in valid_data_loader)
    
        print('Epoch:', epoch, 'Validation loss:', valid_loss / len(valid_data_loader))
        log_results(logfile_validation_path, epoch, train_loss, valid_loss)

    # save a checkpoint
    file_name = "Eye_Fixation_CNN_epoch"+str(epoch)+".pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': eye_fixation_model.state_dict(),
        'optimizer_state_dict': opt.state_dict()
        }, os.path.join(checkpoints_path, file_name))


# TODO
# Train the network?
# Log results in the log file (store network performances, network parameters)
# Handle test files
# Store network predictions with corresponding file names

"""
# Testing 
# Iterate over test data loader to make predictions
for sample in test_data_loader:
    input_image = sample['image'].to(device)
    image_path = sample['image_path'][0]  # Assuming batch size is 1
    with torch.no_grad():
        pred = eye_fixation_model(input_image)
        # Process prediction as needed
        # For example, visualize or save the prediction
        save_prediction(pred, image_path)
"""
