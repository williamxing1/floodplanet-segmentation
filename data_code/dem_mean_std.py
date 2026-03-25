from pathlib import Path
import rasterio
import numpy as np

tifs_path = Path("../data/tif_model_training")
all_values = []

for tif in tifs_path.iterdir():
    with rasterio.open(tif) as src:
        dem = np.array(src.read(7), dtype=np.float32)
        dem = dem.flatten()
        all_values.append(dem)

all_values = np.concatenate(all_values)
mean = all_values.mean()
std = all_values.std()

print(f"Mean: {mean}")
print(f"STD: {std}")