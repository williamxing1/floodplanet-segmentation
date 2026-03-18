from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import json, os
import time
from unet_model import UNet
from data_code.dataloader import Sentinel2Dataset, FloodPlanetDataset
import torchvision.models.segmentation as segmentation

class Config:
    run: str = "S2" # PS or S2
    in_ch: int = 4 if run == "PS" else 7
    load_weights: bool = False
    load_folder: str = "model46-S2"
    epochs: int = 50
    model: str = "deeplabv3" # "deeplabv3" or "unet"
    learning_rate: float = 1e-3

config = Config()
json_path = "/outputs/index.json"
with open(json_path, "r") as f:
    data = json.load(f)
folder_path = "model" + str(data["index"]) + "-" + config.run
print(f"Folder name: {folder_path}")
data["index"] += 1
with open(json_path, "w") as f:
    json.dump(data, f)
os.makedirs(f"/outputs/{folder_path}", exist_ok=True)

if config.run == "S2":
    dataset = Sentinel2Dataset("/data/tif_model_training")
else:
    dataset = FloodPlanetDataset("/data/FloodPlanet", config.run)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
test_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
print(f"Train length: {len(train_loader.dataset)}; Val length: {len(test_loader.dataset)}")

torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if config.model == "unet":
    model = UNet(in_channels=config.in_ch, num_classes=1).to(device).to(memory_format=torch.channels_last)
elif config.model == "deeplabv3":
    model = segmentation.deeplabv3_resnet101(weights="DEFAULT")
    old = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(config.in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.backbone.conv1.weight[:, :3] = old.weight
        model.backbone.conv1.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model = model.to(device).to(memory_format=torch.channels_last)

if config.load_weights:
    state_dict = torch.load(f"/outputs/{config.load_folder}/best_unet_floodplanet.pth")
    model.load_state_dict(state_dict)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=device))

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
if config.load_weights:
    state_dict = torch.load(f"/outputs/{config.load_folder}/best_optimizer.pth")
    optimizer.load_state_dict(state_dict)
train_losses = []
test_losses = []
best_test = float("inf")
scaler = torch.amp.GradScaler("cuda")

for epoch in range(config.epochs):
    model.train()
    t0 = time.time()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16):
            preds = model(xb)
            if config.model == "deeplabv3":
                preds = preds["out"]
            yb = yb.unsqueeze(1).float()
            loss = criterion(preds, yb)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16):
                preds = model(xb)
                if config.model == "deeplabv3":
                    preds = preds["out"]
                yb = yb.unsqueeze(1).float()
                loss = criterion(preds, yb)
            test_loss += loss.item() * xb.size(0)
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    if test_loss < best_test:
        best_test = test_loss
        torch.save(model.state_dict(), f"/outputs/{folder_path}/best_unet_floodplanet.pth")
        torch.save(optimizer.state_dict(), f"/outputs/{folder_path}/best_optimizer.pth")
    t1 = time.time()
    time_elapsed = t1 - t0
    print(f"Epoch {epoch+1}: Train Loss: {train_loss}, Test Loss: {test_loss}, Time Elapsed: {time_elapsed:.2f} seconds")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"/outputs/{folder_path}/train_test_curve.png", dpi=300, bbox_inches="tight")
plt.show()

if config.model == "unet":
    model = UNet(in_channels=config.in_ch, num_classes=1).to(device).to(memory_format=torch.channels_last)
elif config.model == "deeplabv3":
    model = segmentation.deeplabv3_resnet101(weights="DEFAULT")
    old = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(config.in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.backbone.conv1.weight[:, :3] = old.weight
        model.backbone.conv1.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model = model.to(device).to(memory_format=torch.channels_last)

state_dict = torch.load(f"/outputs/{folder_path}/best_unet_floodplanet.pth", map_location=device)
model.load_state_dict(state_dict)

model.eval()
xb, yb = next(iter(test_loader))
xb = xb.to(device, memory_format=torch.channels_last)
yb = yb.to(device)

with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.float16):
        preds = model(xb)
        if config.model == "deeplabv3":
            preds = preds["out"]

plt.figure()
for index in range(3):
    img = xb[index][:3][[2, 1, 0]].permute(1,2,0).cpu().numpy()
    true_mask = yb[index].cpu().numpy()
    preds = preds.to(torch.float32)
    preds_probs = torch.sigmoid(preds)
    pred_mask = (preds_probs > 0.5).float()
    pred_mask = pred_mask[index].permute(1, 2, 0).cpu().numpy()

    plt.subplot(3, 3, index * 3 + 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(3, 3, index * 3 + 2)
    plt.title("Ground Truth Mask")
    plt.imshow(true_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, index * 3 + 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="gray")
    plt.axis("off")

plt.savefig(f"/outputs/{folder_path}/sample_val_preds.png", dpi=300, bbox_inches="tight")
plt.show()