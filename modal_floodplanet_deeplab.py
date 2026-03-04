import modal

app = modal.App("floodplanet-model")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "matplotlib",
        "rasterio"
    )
)

data_volume = modal.Volume.from_name("floodplanet-data")
output_volume = modal.Volume.from_name("floodplanet-outputs")

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 10,
    volumes={
        "/data": data_volume,
        "/outputs": output_volume,
    },
)

def train():
    from torch.utils.data import Dataset, random_split, DataLoader
    import torchvision.models.segmentation as segmentation
    from pathlib import Path
    import rasterio
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import torch.optim as optim
    import json, os
    import time

    run = "S2" # PS or S2
    in_ch = 4 if run == "PS" else 10
    load_weights = False
    load_folder = "model44-S2"

    json_path = "/outputs/index.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    folder_path = "model" + str(data["index"]) + "-" + run
    print(f"Folder name: {folder_path}")
    data["index"] += 1
    with open(json_path, "w") as f:
        json.dump(data, f)
    os.makedirs(f"/outputs/{folder_path}", exist_ok=True)

    class FloodPlanetDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = Path(root_dir)
            self.samples = []

            for folder in self.root_dir.iterdir():
                if not folder.is_dir():
                    continue
                for dataset_type in folder.iterdir():
                    if dataset_type.name not in [run] or not dataset_type.is_dir():
                        continue
                    for tif in dataset_type.glob("*.tif"):
                        self.samples.append((tif, folder / "labels" / tif.name))
                        
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            tif_path, label_path = self.samples[idx]
            with rasterio.open(tif_path) as f:
                image = f.read().astype(np.float32)
                image = image / 10000.0
            with rasterio.open(label_path) as f:
                mask = f.read().astype(np.float32)
            
            image, mask = torch.from_numpy(image), torch.from_numpy(mask)

            image = image.unsqueeze(0)
            image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
            image = image.squeeze(0)

            mask = mask.unsqueeze(0)
            mask = F.interpolate(mask, size=(256, 256), mode="nearest")
            mask = mask.squeeze(0).squeeze(0).long()
            mask = (mask == 1).long()

            return image, mask

    dataset = FloodPlanetDataset("/data/FloodPlanet")
    train_size = int(0.8 * len(dataset)) + 1
    val_size = int(0.2 * len(dataset))
    print(len(dataset), train_size, val_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model = segmentation.deeplabv3_resnet101(weights="DEFAULT")
    old = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.backbone.conv1.weight[:, :3] = old.weight
        model.backbone.conv1.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model = model.to(device).to(memory_format=torch.channels_last)

    if load_weights:
        state_dict = torch.load(f"/outputs/{load_folder}/best_deeplab_floodplanet.pth")
        model.load_state_dict(state_dict)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if load_weights:
        state_dict = torch.load(f"/outputs/{load_folder}/best_optimizer.pth")
        optimizer.load_state_dict(state_dict)
    epochs = 50
    train_losses = []
    test_losses = []
    best_test = float("inf")
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16):
                preds = model(xb)["out"]
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
            for xb, yb in val_loader:
                xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=torch.float16):
                    preds = model(xb)["out"]
                    yb = yb.unsqueeze(1).float()
                    loss = criterion(preds, yb)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(val_loader.dataset)
        test_losses.append(test_loss)
        if test_loss < best_test:
            best_test = test_loss
            torch.save(model.state_dict(), f"/outputs/{folder_path}/best_deeplab_floodplanet.pth")
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

    model = segmentation.deeplabv3_resnet101(weights="DEFAULT")
    old = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.backbone.conv1.weight[:, :3] = old.weight
        model.backbone.conv1.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    state_dict = torch.load(f"/outputs/{folder_path}/best_deeplab_floodplanet.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).to(memory_format=torch.channels_last)

    model.eval()
    xb, yb = next(iter(val_loader))
    xb = xb.to(device, memory_format=torch.channels_last)
    yb = yb.to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16):
            preds = model(xb)["out"]

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