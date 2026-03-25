from pathlib import Path
from data_loading_utils import extract_rgb_nir_swir, get_lcc_for_tile, get_tile_meta, get_dem_for_tile
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# ---------- Sentinel 2 Channel Demo ----------
tif_path = Path("../data/test.tif")
json_path = Path("../data/test.json")

data = extract_rgb_nir_swir(tif_path, json_path)
R, G, B, NIR, SWIR = data["R"], data["G"], data["B"], data["NIR"], data["SWIR"]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()
for ax, (name, arr) in zip(axes, [("R", R), ("G", G), ("B", B), ("NIR", NIR), ("SWIR", SWIR)]):
    vmin, vmax = np.nanpercentile(arr, [2, 98]) if np.any(np.isfinite(arr)) else (0, 1)
    im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap="gray")
    ax.set_title(name)
    plt.colorbar(im, ax=ax)
    ax.axis("off")
axes[5].axis("off")
plt.suptitle("Sentinel 2 Bands")
plt.tight_layout()
plt.show()

def norm(x):
    x = np.asarray(x, dtype=np.float32)
    p2, p98 = np.nanpercentile(x, [2, 98])
    if p98 <= p2:
        return np.clip(x, 0, 1).astype(np.float32)
    return np.clip((x - p2) / (p98 - p2), 0, 1).astype(np.float32)

rgb = np.stack([norm(R), norm(G), norm(B)], axis=-1)
plt.imshow(rgb)
plt.title("Full RGB Image")
plt.show()

# ---------- Land Cover Classification (LCC) Demo ----------
with rasterio.open(tif_path) as src:
    print(f"Original TIF height: {src.height}")
    print(f"Original TIF width: {src.width}")

tile_info = get_tile_meta(tif_path)
ref_meta = tile_info["meta"]
wgs84_bounds = tile_info["wgs84_bounds"]
lcc_d = get_lcc_for_tile(ref_meta, wgs84_bounds)
lcc = lcc_d["arr"]
print(f"Land cover shape: {lcc.shape}")
plt.imshow(lcc, cmap="tab20", vmin=0, vmax=100)
plt.title("Land Cover Data")
plt.colorbar()
plt.axis("off")
plt.show()

# ---------- Digital Elevation Model (DEM) Demo -----------
dem_d = get_dem_for_tile(ref_meta, wgs84_bounds)
dem = dem_d["arr"]
print(f"DEM shape: {dem.shape}")
plt.imshow(dem, cmap="terrain")
plt.title("DEM Data")
plt.colorbar(label="meters")
plt.axis("off")
plt.show()