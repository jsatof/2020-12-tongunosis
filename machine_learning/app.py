import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Function
import glob
import datetime
from PIL import Image

import pdb # debugger

# UNet Component
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)

# UNet Component
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# UNet Component
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 3, stride = 2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)

# UNet Component
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)

# The UNet Architecture
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.input = DoubleConv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownConv(512, 1024 // factor)
        
        self.up1 = UpConv(1024, 512 // factor, bilinear)
        self.up2 = UpConv(512, 256 // factor, bilinear)
        self.up3 = UpConv(256, 128 // factor, bilinear)
        self.up4 = UpConv(128, 64, bilinear)
        self.output = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.output(x)
        return logits

class TongueImages(Dataset):
    def __init__(self, image_path, mask_path, train = True):
        super(TongueImages, self).__init__()
        self.train = train
        self.image_path = image_path
        self.mask_path = mask_path

        self.image_names = [os.path.splitext(filename) for filename in os.listdir(self.image_path)]

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(self.image_path + image_name[0] + image_name[1]).convert("RGB")
        mask = Image.open(self.mask_path + "mask." + image_name[0] + image_name[1]).convert("L")
        image.load(); mask.load()
        
        image_array = self.get_array(image)
        mask_array = self.get_array(mask)

        return {
            "image": torch.from_numpy(image_array).type(torch.FloatTensor),
            "mask": torch.from_numpy(mask_array).type(torch.FloatTensor)
        }

    def __len__(self):
        return len(self.image_names)

    @classmethod
    def get_array(self, image):
        image_array = np.array(image)                               

        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis = 2)

        transformed_array = image_array.transpose((2, 0, 1))        # convert to (channels, height, width)

        if transformed_array.max() > 1:
            transformed_array = transformed_array / 255             # convert channels to grayscale

        return transformed_array    

class DiceCoeff(Function):
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

# Dice coeff for batches
def dice_coeff(input, target):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def evaluate_model(model, loader, device):
    model.eval()
    mask_type = torch.float32 if model.n_classes == 1 else torch.long
    num_values = len(loader)
    total = 0

    for batch in loader:
        images, true_masks = batch["image"], batch["mask"]
        images = images.to(device = device, dtype = torch.float32)
        true_masks = true_masks.to(device = device, dtype = mask_type)

        with torch.no_grad():
            mask_prediction = model(images)
        
        if model.n_classes > 1:
            total += F.cross_entropy(mask_prediction, true_masks).item()
        else:
            prediction = torch.sigmoid(mask_prediction)
            prediction = (prediction > 0.5).float()
            total += dice_coeff(prediction, true_masks).item()

    model.train()
    return total / num_values

def train_model(model, device, epochs, batch_size, learn_rate, save_model = True):
    if save_model:
        save_point = "output/save_point/"

    dataset = TongueImages(image_path = "train_images/", mask_path = "image_masks/", train = True)
    
    split_percent = 0.10
    num_values = int(split_percent * len(dataset))
    num_training = len(dataset) - num_values

    training_set, evaluation_set = random_split(dataset, [num_training, num_values])

    train_loader = DataLoader(training_set, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)
    eval_loader = DataLoader(evaluation_set, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True, drop_last = True)   

    global_step = 0

    optimizer = optim.RMSprop(model.parameters(), lr = learn_rate, weight_decay = 1e-8, momentum = 0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min" if model.n_classes > 1 else "max", patience = 2)
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        print(f"Epoch {epoch + 1} start")

        for batch in train_loader:
            images = batch["image"]
            true_masks = batch["mask"]

            mask_type = torch.float32 if model.n_classes == 1 else torch.long
            true_masks = true_masks.to(device = device, dtype = mask_type)
            images = images.to(device = device, dtype = torch.float32)

            mask_prediction = model(images)

            loss = criterion(mask_prediction, true_masks)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            global_step += 1

            if global_step % (num_training // (10 * batch_size)) == 0:
                for tag, value in model.named_parameters():
                    eval_score = evaluate_model(model, eval_loader, device)
                    scheduler.step(eval_score)
                    print(f"Global Step: {global_step}\nEpoch {epoch + 1} of {epochs}:\nLoss: {epoch_loss}\nEval Score: {eval_score}")

        if save_model:
            try:
                os.mkdir(save_point)
            except OSError:
                pass

            time_now = datetime.datetime.now()
            year = time_now.year
            month = time_now.strftime("%m") # number month
            day = time_now.strftime("%d") # number day of month
            hour = time_now.strftime("%H") # hour digit
            minute = time_now.strftime("%I") # minute digit

            torch.save(model.state_dict(), save_point + f"{year}-{month}-{day}-{hour}{minute}_epoch{epoch + 1}.pth")

def predict_image(model, image, device, result_threshold = 0.5):
    model.eval()
    
    image_tensor = torch.from_numpy(TongueImages.get_array(image))
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device = device, dtype = torch.float32)

    with torch.no_grad():
        output = model(image_tensor)

        if model.n_classes > 1:
            probability = F.softmax(output, dim = 1)
        else:
            probability = torch.sigmoid(output)

        probability = probability.squeeze(0)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image.size[1]),
            transforms.ToTensor()
        ])

        probability = transform(probability.cpu())
        result_mask = probability.squeeze().cpu().numpy()

    return result_mask > result_threshold

def main():
    if not len(sys.argv) == 2:
        print("Incorrect args: Usage <program> <train = 'true' | 'false'>")
        return
    
    # switch to indicate whether to load model for training or to predict an input image
    if sys.argv[1] == "true":
        train = True
    elif sys.argv[1] == "false":
        train = False
    else:
        print("Incorrect args\nUsage: python <program> <train = 'true' | 'false'>")
        exit(0)

    # these are parameters that can be toyed to potentially improve performance (the hyperparameters)
    batch_size = 10
    learn_rate = 0.0001
    epochs = 5
    accept_threshold = 0.80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels = 3, n_classes = 1)         # channels are 3 (color), num of classes = 1 (segment only tongue class)
    model.to(device = device)

    # grab latest .pth file
    pth_file_list = glob.glob("output/save_point/*.pth")
    latest_pth = max(pth_file_list, key = os.path.getctime)

    # load trained weights from file
    if latest_pth is not None:
        model.load_state_dict(torch.load(latest_pth, map_location = device))
        print("Model Loaded: {}".format(latest_pth))
    
    if train:
        print("Begin training\n")
        train_model(model, device, epochs, batch_size, learn_rate)
    else:
        # loading the model post-training
        
        input_files = "eval_input/"
        for filename in os.listdir(input_files):
            image = Image.open(input_files + filename).convert("RGB")
            mask = predict_image(model, image, device, accept_threshold)

            result = Image.fromarray((mask * 255).astype(np.uint8))
            result.show()


if __name__ == "__main__":
    main()
