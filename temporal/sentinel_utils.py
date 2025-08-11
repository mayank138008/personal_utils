import os
import json
import yaml
import zipfile
import shutil
import requests
import numpy as np
import rasterio
import geopandas as gpd
import joblib
from pathlib import Path
from datetime import date, datetime
from dotenv import load_dotenv
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from temporalio import activity


def file_exists(path: str) -> bool:
    return os.path.exists(path)


def dir_has_files(path: str) -> bool:
    return os.path.exists(path) and bool(os.listdir(path))


def load_geojson(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ AOI geojson not found at: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_config(config_path: str = "config/config.yaml") -> dict:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config.setdefault("credentials", {})
    config["credentials"]["username"] = config["credentials"].get("username") or os.getenv("COPERNICUS_USERNAME")
    config["credentials"]["password"] = config["credentials"].get("password") or os.getenv("COPERNICUS_PASSWORD")

    if "aoi" in config and isinstance(config["aoi"], dict):
        geo_path = Path(config["aoi"]["geojson_path"])
        # if not geo_path.is_absolute():
        #     geo_path = Path(__file__).resolve().parent / geo_path

        if not geo_path.is_absolute():
            geo_path = Path.cwd() / geo_path    
        config["aoi"]["geojson_path"] = str(geo_path)

    if "date_range" not in config:
        start = config.get("project", {}).get("start_date", "2022-01-01")
        config["date_range"] = {
            "start": str(start),
            "end": date.today().isoformat()
        }

    return config


def get_token(username: str, password: str) -> str:
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


def search_metadata(token: str, geojson: dict, start: str, end: str) -> list:
    url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    geometry = geojson["features"][0]["geometry"]
    payload = {
        "intersects": geometry,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "collections": ["sentinel-2-l2a"],
        "limit": 100
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["features"]


def download_safe(product_id: str, token: str, download_dir: str) -> str:
    odata_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq '{product_id}'"
    odata_response = requests.get(odata_url)
    odata_response.raise_for_status()
    odata_id = odata_response.json()["value"][0]["Id"]
    download_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({odata_id})/$value"

    headers = {"Authorization": f"Bearer {token}", "Accept": "*/*"}
    response = requests.get(download_url, headers=headers, stream=True)
    response.raise_for_status()

    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, f"{product_id}.zip")
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)
    return file_path


def unzip_safe(zip_path: str, final_target_path: str, short_temp_path: str = "tmp/safe_tmp") -> str:
    folder_name = os.path.basename(zip_path).replace(".zip", "")
    temp_extract_path = os.path.join(short_temp_path, folder_name)
    os.makedirs(temp_extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)

    os.makedirs(final_target_path, exist_ok=True)
    entries = os.listdir(temp_extract_path)
    only_entry = os.path.join(temp_extract_path, entries[0])
    if only_entry.endswith(".SAFE") and os.path.isdir(only_entry):
        for item in os.listdir(only_entry):
            shutil.move(os.path.join(only_entry, item), os.path.join(final_target_path, item))
    shutil.rmtree(temp_extract_path)
    return final_target_path




def convert_bands(safe_path: str, output_dir: str) -> str:
    print(f"🎨 Starting band conversion")
    print(f"📂 SAFE path: {safe_path}")
    print(f"📁 Output dir: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    granule_path = os.path.join(safe_path, "GRANULE")
    granule_sub = os.listdir(granule_path)[0]
    img_data_dir = os.path.join(granule_path, granule_sub, "IMG_DATA")

    band_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(img_data_dir)
        for file in files if file.endswith(".jp2")
    ]

    print(f"📸 Found {len(band_files)} .jp2 band files")

    for band_path in band_files:
        out_name = os.path.basename(band_path).replace(".jp2", ".tif")
        out_path = os.path.join(output_dir, out_name)
        with rasterio.open(band_path) as src:
            profile = src.profile
            profile.update(driver="GTiff", compress="deflate")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(src.read())

    # === Generate cloud mask from SCL band ===
    scl_path = next((f for f in band_files if "_SCL_" in f), None)
    if scl_path:
        print(f"🌥️ Found SCL band: {scl_path}")
        scl_out_path = os.path.join(output_dir, "SCL.tif")
        with rasterio.open(scl_path) as src:
            scl_data = src.read(1)

            # Define cloud-related SCL classes: 3 = Cloud shadows, 8–11 = various clouds
            cloud_mask = np.isin(scl_data, [3, 8, 9, 10, 11]).astype(np.uint8)

            # Save SCL band
            profile = src.profile.copy()
            profile.update(dtype=rasterio.uint8, count=1, compress="deflate")
            with rasterio.open(scl_out_path, "w", **profile) as dst:
                dst.write(scl_data, 1)
            print(f"🗺️ Saved SCL band: {scl_out_path}")

            # Save cloud mask
            cloud_out_path = os.path.join(output_dir, "cloud_mask.tif")
            with rasterio.open(cloud_out_path, "w", **profile) as dst:
                dst.write(cloud_mask, 1)
            print(f"🌀 Saved cloud mask: {cloud_out_path}")
    else:
        print("⚠️ No SCL band found — cloud mask will be empty in merged output.")

    return output_dir



def merge_bands(scene_dir: str, output_path: str) -> str:
    print(f"🧩 Starting band merge")
    print(f"📂 Scene dir: {scene_dir}")
    print(f"📁 Output path: {output_path}")

    bands_order = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    all_tifs = [f for f in os.listdir(scene_dir) if f.endswith(".tif") and not f.startswith("cloud_")]
    
    # Use first valid band to get metadata
    first_band_path = None
    for band_name in bands_order:
        match = next((f for f in all_tifs if f"_{band_name}_" in f), None)
        if match:
            first_band_path = os.path.join(scene_dir, match)
            break

    if not first_band_path:
        raise RuntimeError("❌ No matching band files found to determine metadata.")

    with rasterio.open(first_band_path) as src0:
        meta = src0.meta.copy()
        meta.update(count=len(bands_order) + 1, driver="GTiff", compress="deflate")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, "w", **meta) as dst:
        for idx, band_name in enumerate(bands_order, start=1):
            band_file = next((f for f in all_tifs if f"_{band_name}_" in f), None)
            if band_file:
                with rasterio.open(os.path.join(scene_dir, band_file)) as src:
                    data = src.read(1).astype(meta["dtype"])
                    dst.write(data, idx)
                    print(f"  ✅ Added {band_name} to band {idx}")
            else:
                print(f"  ⚠️ Missing {band_name}, filling with zeros")
                dst.write(np.zeros((meta["height"], meta["width"]), dtype=meta["dtype"]), idx)

        # Add cloud mask as final band
        cloud_mask_path = os.path.join(scene_dir, "cloud_mask.tif")
        if os.path.exists(cloud_mask_path):
            with rasterio.open(cloud_mask_path) as src:
                data = src.read(1).astype(meta["dtype"])
                dst.write(data, len(bands_order) + 1)
                print(f"  ☁️ Added cloud mask as band {len(bands_order) + 1}")
        else:
            print("  ⚠️ No cloud mask found, writing empty band")
            dst.write(np.zeros((meta["height"], meta["width"]), dtype=meta["dtype"]), len(bands_order) + 1)

    print(f"✅ Merged bands written to: {output_path}")
    return output_path


def clip_to_aoi(input_tif: str, output_tif: str, aoi_geojson_path: str) -> str:
    aoi = gpd.read_file(aoi_geojson_path)
    with rasterio.open(input_tif) as src:
        aoi_proj = aoi.to_crs(src.crs)
        aoi_geom = aoi_proj.geometry.values
        out_image, out_transform = mask(src, aoi_geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        os.makedirs(os.path.dirname(output_tif), exist_ok=True)
        with rasterio.open(output_tif, "w", **out_meta) as dst:
            dst.write(out_image)
    return output_tif


def generate_maps(input_tif: str, output_dir: str, model_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(input_tif) as src:
        bands = src.read()
        height, width = bands.shape[1:]
        profile = src.profile.copy()
        profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)
        meta = src.meta

    B = {
        "B2": bands[1],
        "B3": bands[2],
        "B4": bands[3],
        "B8": bands[7],
        "B12": bands[11],
    }

    with np.errstate(divide='ignore', invalid='ignore'):
        NDVI = (B["B8"] - B["B4"]) / (B["B8"] + B["B4"])
        NDMI = (B["B8"] - B["B12"]) / (B["B8"] + B["B12"])

    for name, arr in [("ndvi", NDVI), ("ndmi", NDMI)]:
        out_path = os.path.join(output_dir, f"{name}_{os.path.basename(input_tif)}")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr.astype(np.float32), 1)

    valid_mask = np.all(bands != 0, axis=0)
    valid_flat = valid_mask.flatten()
    X = bands[:12].transpose(1, 2, 0).reshape(-1, 12)


    models = {}
    for nut in ["N", "P", "K"]:
        model_path = os.path.join(model_dir, f"rf_model_{nut}.joblib")
        models[nut] = joblib.load(model_path)

    for nut, model in models.items():
        y_pred = np.full((height * width), np.nan)

        if np.count_nonzero(valid_flat) == 0:
            print(f"⚠️ No valid pixels to predict for {nut}. Skipping.")
            continue

        try:
            y_pred[valid_flat] = model.predict(X[valid_flat])
        except Exception as e:
            print(f"❌ Prediction failed for {nut}: {e}")
            continue

        y_pred = y_pred.reshape(height, width)
        out_path = os.path.join(output_dir, f"{nut}_pred_{os.path.basename(input_tif)}")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(y_pred.astype(np.float32), 1)
        print(f"🧪 Saved {nut} prediction map to {out_path}")


import argparse
def run_one_product(product_id: str, config_path: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--product-id", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    geojson = load_geojson(config["aoi"]["geojson_path"])
    creds = config["credentials"]
    project_name = config["project"]["name"]
    token = get_token(creds["username"], creds["password"])

    download_dir = os.path.join(project_name, "raw_downloads")
    unzip_dir = os.path.join(project_name, "unzipped")
    ind_bands_dir = os.path.join(project_name, "ind_bands")
    merged_tif = os.path.join(project_name, "merged", f"merged_{args.product_id}.tif")
    clipped_tif = os.path.join(project_name, "clipped", f"clipped_{args.product_id}.tif")
    maps_output_dir = os.path.join(project_name, "maps")
    model_dir = config["models"]["directory"]

    zip_path = os.path.join(download_dir, f"{args.product_id}.zip")
    safe_path = os.path.join(unzip_dir, f"{args.product_id}.SAFE")
    band_output = os.path.join(ind_bands_dir, args.product_id)
    cloud_mask_path = os.path.join(band_output, "cloud_mask.tif")
    npk_paths = [
        os.path.join(maps_output_dir, f"{prefix}_{os.path.basename(clipped_tif)}")
        for prefix in ["ndvi", "ndmi", "N_pred", "P_pred", "K_pred"]
    ]

    if not os.path.exists(zip_path):
        zip_path = download_safe(args.product_id, token, download_dir)

    if not os.path.exists(safe_path):
        safe_path = unzip_safe(zip_path, safe_path)

    if not os.path.exists(cloud_mask_path):
        convert_bands(safe_path, band_output)

    if not os.path.exists(merged_tif):
        merge_bands(band_output, merged_tif)

    if not os.path.exists(clipped_tif):
        clip_to_aoi(merged_tif, clipped_tif, config["aoi"]["geojson_path"])

    if not all(os.path.exists(p) for p in npk_paths):
        generate_maps(clipped_tif, maps_output_dir, model_dir)

    print(f"✅ Done processing {args.product_id}")
