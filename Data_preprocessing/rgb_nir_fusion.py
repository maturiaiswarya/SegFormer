from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        
        # First stage
        self.conv1_rgb = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_nir = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Second stage
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, rgb, nir):
        # First stage
        rgb_feat = self.relu(self.conv1_rgb(rgb))
        nir_feat = self.relu(self.conv1_nir(nir))
        concat = torch.cat((rgb_feat, nir_feat), dim=1)
        feat = self.relu(self.conv2(concat))
        feat = self.relu(self.conv3(feat))
        
        # Second stage
        out = self.relu(self.conv4(feat))
        out = self.relu(self.conv5(out))
        out = self.conv6(out)
        
        return out


class RGBNIRDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None):
        self.rgb_dir = os.path.join(data_root, split, 'rgb')
        self.nir_dir = os.path.join(data_root, split, 'nir')
        self.transform = transform
        self.image_files = []
        
        for subdir in os.listdir(self.rgb_dir):
            subdir_path = os.path.join(self.rgb_dir, subdir)
            if os.path.isdir(subdir_path):
                self.image_files.extend([os.path.join(subdir, f) for f in os.listdir(subdir_path) if f.endswith('_rgb.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        rgb_path = os.path.join(self.rgb_dir, img_path)
        nir_path = os.path.join(self.nir_dir, img_path.replace('_rgb.png', '_nir.png'))

        rgb_image = Image.open(rgb_path).convert('RGB')
        nir_image = Image.open(nir_path).convert('L')

        if self.transform:
            rgb_image = self.transform(rgb_image)
            nir_image = self.transform(nir_image)

        return rgb_image, nir_image

# Hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 10
data_root = 'datasets/IDDAW_ICPR'

# Data loading
transform = transforms.Compose([
    transforms.Resize((1024, 768)),
    transforms.ToTensor(),
])

train_dataset = RGBNIRDataset(data_root, split='train', transform=transform)
val_dataset = RGBNIRDataset(data_root, split='val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for rgb, nir in tqdm(train_loader):
        rgb, nir = rgb.to(device), nir.to(device)

        optimizer.zero_grad()
        outputs = model(rgb, nir)
        loss = criterion(outputs, rgb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f'Train loss at epoch {epoch+1}: {train_loss:.4f}')
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for rgb, nir in val_loader:
            rgb, nir = rgb.to(device), nir.to(device)
            outputs = model(rgb, nir)
            loss = criterion(outputs, rgb)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_fusionnet_model.pth')

print('Training completed.')
