
import os
import cv2
import torch
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import time

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png') or fname.endswith('.jpg')])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png') or fname.endswith('.jpg')])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # Add channel dimension to mask

        return image, mask

# Example of an augmentation pipeline with resizing and normalization
transform = A.Compose([
    A.Resize(height=256, width=256),  # Resize to a fixed size
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.GaussianBlur(p=0.1),
    A.ColorJitter(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalization
    ToTensorV2()
])

class UNet(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=5, lr=1e-3):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        self.lr = lr

    def forward(self, x):
        enc_out = self.encoder(x)
        mid_out = self.middle(enc_out)
        dec_out = self.decoder(mid_out)
        return dec_out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).long()  # Squeeze the channel dimension for CrossEntropyLoss
        pred = self(x)
        loss = nn.CrossEntropyLoss()(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).long()  # Squeeze the channel dimension for CrossEntropyLoss
        pred = self(x)
        loss = nn.CrossEntropyLoss()(pred, y)
        self.log('val_loss', loss)
        return loss

# Define paths to your image and mask directories
image_dir = '/content/drive/My Drive/images'
mask_dir = '/content/drive/My Drive/masks'

# Create the dataset and dataloaders
dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet(in_channels=3, out_channels=5)

# Initialize the PyTorch Lightning trainer
trainer = pl.Trainer(max_epochs=100, gpus=1 if torch.cuda.is_available() else 0)

# Train the model
trainer.fit(unet, train_loader, val_loader)

# Save the model
torch.save(unet.state_dict(), "unet.pth")

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    return augmented['image']

# Function to predict the mask
def predict_mask(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension and send to device
        output = model(image_tensor)
        output = torch.argmax(output, dim=1).cpu().numpy()  # Get the predicted class for each pixel
    return output[0]  # Remove batch dimension

# Function to visualize the original image and the predicted mask
def visualize(image_path, predicted_mask, save_path=None):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap='jet', alpha=0.5)

    if save_path:
        plt.savefig(save_path)
    plt.show()

# Directory containing test images
test_image_dir = "/content/drive/My Drive/Audi"  # Replace with your test images directory

# Iterate over all test images and make predictions
for image_name in os.listdir(test_image_dir):
    if image_name.endswith('.png') or image_name.endswith('.jpg'):
        image_path = os.path.join(test_image_dir, image_name)
        image_tensor = preprocess_image(image_path)
        predicted_mask = predict_mask(unet, image_tensor)

        # Visualize the results
        visualize(image_path, predicted_mask)
