from pathlib import Path
import rasterio
from rasterio.warp import Resampling, transform_bounds
from osgeo import ogr
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
ogr.UseExceptions()

# ---------- Ancillary functions; Used for loading LCC / DEM ----------
lcc_cache = Path("../data/lcc_cache")
dem_cache = Path("../data/dem_cache")

def get_tile_meta(path):
    """Returns meta, bounds, wgs84 bounds, crs from a TIF file"""
    with rasterio.open(path) as src:
        meta = src.meta.copy()
        bounds = src.bounds
        crs = src.crs
    if crs != rasterio.crs.CRS.from_epsg(4326):
        wgs84_bounds = transform_bounds(crs, rasterio.crs.CRS.from_epsg(4326), *bounds)
    else:
        wgs84_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    return {"meta": meta, "bounds": bounds, "wgs84_bounds": wgs84_bounds, "crs": crs}

def _validate_tif(path):
    """Validates that a TIF is correct"""
    with rasterio.open(path) as src:
        _ = src.count, src.width, src.height
    return True

def download_file(url, dir, retries=3, timeout=60):
    """Downloads a single file into a directory and returns the file path"""
    if isinstance(url, str) and url.startswith("/vsicurl/"):
        url = url[len("/vsicurl/"):]
    folder = Path(dir)
    folder.mkdir(parents=True, exist_ok=True)
    filename = Path(url.split("?")[0]).name or "tile.tif"
    finalpath = folder / filename

    if finalpath.exists():
        try:
            _validate_tif(finalpath)
            return finalpath
        except:
            try:
                finalpath.unlink()
            except:
                pass
    for attempt in range(1, retries+1):
        try:
            r = requests.get(url, stream=True, timeout=timeout)
            r.raise_for_status()
            downloaded = 0
            total = int(r.headers.get("content-length", 0))
            with open(finalpath, "wb") as out:
                for chunk in r.iter_content(chunk_size=16 * 1024**2):
                    if chunk:
                        downloaded += len(chunk)
                        print(f"{downloaded/(1000**2):.1f} MB / {total/(1000**2):.1f} MB", end="\r")
                        out.write(chunk)
            print()
            _validate_tif(finalpath)
            return finalpath
        except:
            if finalpath.exists():
                try:
                    finalpath.unlink()
                except:
                    pass
            if attempt == retries:
                raise
    return finalpath

def download_files(file_list, local_dir):
    """Downloads a bunch of files from a list and returns the list of file paths"""
    paths = []
    for url in file_list:
        paths.append(download_file(url, local_dir))
    return paths


def reproject_and_merge_max(file_list, meta):
    """Reprojects a bunch of LCC files and arranges them according to a metafile. Ideally max does nothing
    but it's there to ensure that there's not too many 0s. Max is used since values represent classes."""
    dst_crs = meta["crs"]
    dst_transform = meta["transform"]
    dst_height = meta["height"]
    dst_width = meta["width"]
    out = np.zeros((dst_height, dst_width), dtype=np.float32)
    for file in file_list:
        try:
            with rasterio.open(file) as src:
                temp = np.zeros((dst_height, dst_width), dtype=np.float32)
                rasterio.warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=temp,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=dst_crs,
                    dst_transform=dst_transform,
                    resampling=Resampling.nearest,
                    src_nodata=getattr(src, "nodata", None),
                    dst_nodata=0,
                )
                np.maximum(out, temp, out=out)
        except:
            continue
    return out

def reproject_and_merge_mean(file_list, meta):
    """Reprojects a bunch of DEM files and reprojects them according to a metafile. Means are similarly ideally
    not used but DEM overlaps are pretty common. Mean is used since elevation is continuous."""
    dst_crs = meta["crs"]
    dst_transform = meta["transform"]
    dst_height = meta["height"]
    dst_width = meta["width"]
    out = np.zeros((dst_height, dst_width), dtype=np.float32)
    count = np.zeros((dst_height, dst_width), dtype=np.float32)
    for file in file_list:
        try:
            with rasterio.open(file) as src:
                temp = np.zeros((dst_height, dst_width), dtype=np.float32)
                rasterio.warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=temp,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=dst_crs,
                    dst_transform=dst_transform,
                    resampling=Resampling.bilinear,
                    src_nodata=getattr(src, "nodata", None),
                    dst_nodata=0,
                )
                valid = np.isfinite(temp)
                out[valid] += temp[valid]
                count[valid] += 1
        except:
            continue
    np.divide(out, count, out=out, where=count > 0)
    out[count == 0] = 0
    return out

def get_lcc_list(bounds):
    """Extracts a bunch of LCC files based on some boundaries."""
    s3_url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    source = ogr.Open("../data/worldcover.geojson")
    layer = source.GetLayer()
    extent_geom = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(bounds[0], bounds[1])
    ring.AddPoint(bounds[2], bounds[1])
    ring.AddPoint(bounds[2], bounds[3])
    ring.AddPoint(bounds[0], bounds[3])
    ring.AddPoint(bounds[0], bounds[1])
    extent_geom.AddGeometry(ring)
    elements = []

    for feature in layer:
        geom = feature.GetGeometryRef()
        if extent_geom.Intersects(geom):
            tile = feature.GetField("ll_tile")
            elements.append(f"{s3_url}/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile}_Map.tif")

    source = None
    return elements

def glo30list(bounds):
    source = ogr.Open("../data/cop30-2021.geojson")
    layer = source.GetLayer()
    extent_geom = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(bounds[0], bounds[1])
    ring.AddPoint(bounds[2], bounds[1])
    ring.AddPoint(bounds[2], bounds[3])
    ring.AddPoint(bounds[0], bounds[3])
    ring.AddPoint(bounds[0], bounds[1])
    extent_geom.AddGeometry(ring)
    elements = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if extent_geom.Intersects(geom):
            elements.append(feature.GetField("file_path"))
    return elements

def get_lcc_for_tile(ref_meta, wgs84_bounds):
    """Gets the relevant LCC files based on boundaries from a TIF file, downloads them to local, reprojects them on local,
    and returns the combined file."""
    tile_list = get_lcc_list(wgs84_bounds)
    if not tile_list:
        h, w = ref_meta["height"], ref_meta["width"]
        return {"arr": np.zeros((h, w), dtype=np.float32), "meta": ref_meta.copy(), "bounds": None}
    local_list = download_files(tile_list, lcc_cache)
    arr = reproject_and_merge_max(local_list, ref_meta)
    return {"arr": arr, "meta": ref_meta.copy(), "bounds": None}

def get_dem_for_tile(ref_meta, wgs84_bounds):
    """Gets the relevant DEM files based on boundaries from a TIF file, downloads them to local, reprojects them,
    and returns the combined file."""
    tile_list = glo30list(wgs84_bounds)
    if not tile_list:
        h, w = ref_meta["height"], ref_meta["width"]
        return {"arr": np.zeros((h, w), dtype=np.float32), "meta": ref_meta.copy(), "bounds": None}
    local_list = download_files(tile_list, dem_cache)
    arr = reproject_and_merge_mean(local_list, ref_meta)
    return {"arr": arr, "meta": ref_meta.copy(), "bounds": None}


# ---------- Sentinel 2 Channel Extraction ----------
default_indices = {"B": 1, "G": 2, "R": 3, "NIR": 7, "SWIR": 9}

def get_indices(path):
    with open(path, "r") as f:
        data = json.load(f)
    bands = data.get("properties", {}).get("eo:bands", [])
    if not bands:
        return None
    output = {}
    for i, element in enumerate(bands, start=1):
        description = element.get("description", "").lower()
        if "blue" in description:
            output["B"] = i
        elif "green" in description:
            output["G"] = i
        elif "red" in description and "edge" not in description:
            output["R"] = i
        elif "nir" in description:
            output["NIR"] = i
        elif "swir 1" in description or "swir1" in description:
            output["SWIR"] = i
        elif ("swir 2" in description or "swir2" in description) and "SWIR" not in output:
            output["SWIR"] = i
    if len(output) == 5:
        return output
    return None

def extract_rgb_nir_swir(tif_path, json_path):
    bands_indices = None
    if json_path and json_path.exists():
        bands_indices = get_indices(json_path)
    bands_indices = bands_indices or default_indices
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        bounds = src.bounds
        R = np.array(src.read(bands_indices["R"]), dtype=np.float32)
        G = np.array(src.read(bands_indices["G"]), dtype=np.float32)
        B = np.array(src.read(bands_indices["B"]), dtype=np.float32)
        NIR = np.array(src.read(bands_indices["NIR"]), dtype=np.float32)
        SWIR = np.array(src.read(bands_indices["SWIR"]), dtype=np.float32)
    return {"R": R, "G": G, "B": B, "NIR": NIR, "SWIR": SWIR, "meta": meta, "bounds": bounds}

def stack_with_ancillary(ref_meta, R, G, B, NIR, SWIR, LCC, DEM, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = ref_meta.copy()
    meta.update(count=7, dtype=np.float32, compress="DEFLATE")
    bands = [np.asarray(R, dtype=np.float32), np.asarray(G, dtype=np.float32), np.asarray(B, dtype=np.float32),
             np.asarray(NIR, dtype=np.float32), np.asarray(SWIR, dtype=np.float32), np.asarray(LCC, dtype=np.float32), np.asarray(DEM, dtype=np.float32)]
    with rasterio.open(out_path, "w", **meta) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr, i)
    return out_path

def load_7band_geotiff(in_path):
    with rasterio.open(in_path) as src:
        meta = src.meta.copy()
        arr = src.read()
    names = ["R", "G", "B", "NIR", "SWIR", "LCC", "DEM"]
    return {"bands": {names[i]: band for i, band in enumerate(arr)}, "meta": meta}