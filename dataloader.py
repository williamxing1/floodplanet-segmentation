from torch.utils.data import Dataset
from pathlib import Path
import rasterio
import torch
import torch.nn.functional as F
import numpy as np
from data_loading_utils import extract_rgb_nir_swir, get_tile_meta, get_lcc_for_tile, get_dem_for_tile, stack_with_ancillary
import time

run = "S2" # PS or S2
in_ch = 4 if run == "PS" else 10

class FloodPlanetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []
        index = 1
        for folder in self.root_dir.iterdir():
            if not folder.is_dir():
                continue
            for dataset_type in folder.iterdir():
                if dataset_type.name not in [run] or not dataset_type.is_dir():
                    continue
                for tif in dataset_type.glob("*.tif"):
                    json_path = folder.parent / "stac_catalog" / folder.name / run / tif.stem / f"{tif.stem}.json"
                    # Sentinel-2 Channels
                    channels = extract_rgb_nir_swir(tif, json_path)
                    # LCC Extraction
                    tile_info = get_tile_meta(tif)
                    ref_meta = tile_info["meta"]
                    wgs84_bounds = tile_info["wgs84_bounds"]
                    lcc_d = get_lcc_for_tile(ref_meta, wgs84_bounds)
                    lcc = lcc_d["arr"]
                    # DEM Extraction
                    dem_d = get_dem_for_tile(ref_meta, wgs84_bounds)
                    dem = dem_d["arr"]
                    input_out_path = Path("data/tif_model_training") / f"{tif.stem}_7bands.tif"
                    out_path = stack_with_ancillary(ref_meta, channels["R"], channels["G"], channels["B"], channels["NIR"], channels["SWIR"], lcc, dem, input_out_path)
                    self.samples.append((out_path, dataset_type.parent / "labels" / tif.name))
                    if index % 10 == 0:
                        print(f"Index: {index}", end="\r")
                    index += 1
            print()
            
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        tif_path, label_path = self.samples[idx]
        with rasterio.open(tif_path) as f:
            image = f.read().astype(np.float32)
            image[:5] / 10000.0
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

t0 = time.time()
dataset = FloodPlanetDataset("/Volumes/ml_ssd/FloodPlanet/FloodPlanet")
t1 = time.time()
time_passed = t1-t0
print(f"Time elapsed: {(t1-t0):.2f}")