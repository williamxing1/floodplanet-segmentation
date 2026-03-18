import modal

app = modal.App("floodplanet-segmentation")
image = (
    modal.Image.debian_slim()
    .apt_install("gdal-bin", "libgdal-dev")
    .pip_install("torch", "torchvision", "requests", "matplotlib", "rasterio", "numpy", "gdal==3.6.2")
    .add_local_file("dataloader.py", "/dataloader.py")
    .add_local_file("train.py", "/train.py")
    .add_local_file("unet_model.py", "/unet_model.py")
    .add_local_file("data_loading_utils.py", "/data_loading_utils.py")
)

data_volume = modal.Volume.from_name("floodplanet-data")
output_volume = modal.Volume.from_name("floodplanet-outputs")

@app.function(
    image=image,
    gpu="A100",
    timeout=60*5,
    volumes={
        "/data": data_volume,
        "/outputs": output_volume
    }
)
def train():
    import subprocess
    subprocess.run(["python3", "-u", "/train.py"], check=True)

@app.local_entrypoint()
def main():
    train.remote()