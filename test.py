from pathlib import Path
import rasterio
import matplotlib.pyplot as plt

image = Path("FloodPlanet/FloodPlanet/Bangladesh/S2/BGD_40_119.tif")

with rasterio.open(image) as f:
    img = f.read()

img = img / 10000
img = img[[1,2,0]].transpose(1,2,0)
print(img.shape)
plt.imshow(img)
plt.show()