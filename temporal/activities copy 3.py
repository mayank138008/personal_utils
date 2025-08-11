# activities.py
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
import asyncio
import hashlib
import uuid
import subprocess

# ---------- small helpers ----------
def file_exists(path: str) -> bool:
    return os.path.isfile(path)

def dir_has_files(path: str) -> bool:
    return os.path.isdir(path) and bool(os.listdir(path))

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return obj

def _exists(path: str, kind: str = "file") -> bool:
    if kind == "file":
        return os.path.isfile(path)
    if kind == "dir":
        return os.path.isdir(path) and bool(os.listdir(path))
    return os.path.exists(path)

def _skip(path: str, kind: str = "file") -> bool:
    """Return True if we should skip because the output already exists."""
    if _exists(path, kind):
        print(f"✅ Exists, skipping: {path}")
        return True
    return False

# ---------- activities ----------
@activity.defn
async def load_geojson(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ AOI geojson not found at: {path}")
    with open(path, "r") as f:
        return json.load(f)

@activity.defn
async def load_config(config_path: str = "config/config.yaml") -> dict:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = make_json_safe(config)

    config.setdefault("credentials", {})
    config["credentials"]["username"] = config["credentials"].get("username") or os.getenv("COPERNICUS_USERNAME")
    config["credentials"]["password"] = config["credentials"].get("password") or os.getenv("COPERNICUS_PASSWORD")

    if "aoi" in config and isinstance(config["aoi"], dict):
        geo_path = Path(config["aoi"]["geojson_path"])
        if not geo_path.is_absolute():
            geo_path = Path.cwd() / geo_path
        config["aoi"]["geojson_path"] = str(geo_path)

    if "date_range" not in config:
        start = config.get("project", {}).get("start_date", "2020-01-01")
        config["date_range"] = {"start": str(start), "end": date.today().isoformat()}

    return config

@activity.defn
async def get_token(username: str, password: str) -> str:
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password,
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

# @activity.defn
# async def search_metadata(token: str, geojson: dict, start: str, end: str) -> list:
#     url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
#     headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
#     geometry = geojson["features"][0]["geometry"]
#     payload = {
#         "intersects": geometry,
#         "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
#         "collections": ["sentinel-2-l2a"],
#         "limit": 100,
#     }
#     response = requests.post(url, headers=headers, json=payload)
#     response.raise_for_status()
#     return response.json()["features"]

@activity.defn
async def search_metadata(token: str, geojson: dict, start: str, end: str) -> list:
    """
    STAC query -> features -> collapse to best-per-day.
    'Best' = lowest cloud cover, then lowest no-data %, then latest updated.
    Returns a list of STAC Items (features), one per date, sorted by date.
    """
    import requests
    from datetime import datetime, timezone

    url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    geometry = geojson["features"][0]["geometry"]
    payload = {
        "intersects": geometry,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "collections": ["sentinel-2-l2a"],
        "limit": 1000,  # allow enough to cover duplicates per day
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    def _date_utc_iso(properties: dict) -> str:
        # Prefer 'datetime', fallback to 'start_datetime'
        dt_str = properties.get("datetime") or properties.get("start_datetime")
        if not dt_str:
            return None
        # Normalize to YYYY-MM-DD in UTC
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.date().isoformat()

    def _cloud_pct(properties: dict) -> float:
        # Try common STAC/cloud keys; default to large value if missing
        for k in (
            "eo:cloud_cover",               # STAC EO extension (percent 0-100)
            "s2:cloud_cover",               # sometimes used
            "s2:cloudy_pixel_percentage",   # ESA/S2 metadata
            "cloudCover"                    # occasional alias
        ):
            v = properties.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return 101.0  # worse than any valid %

    def _nodata_pct(properties: dict) -> float:
        for k in ("s2:nodata_pixel_percentage", "nodataPixelPercentage", "s2:nodata_percentage"):
            v = properties.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return 101.0

    def _updated_ts(properties: dict) -> float:
        # Use 'updated' or 'created' or fall back to 'datetime'
        for k in ("updated", "created", "datetime", "start_datetime"):
            v = properties.get(k)
            if v:
                try:
                    return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
                except Exception:
                    continue
        return 0.0

    # Group by UTC date string
    by_day = {}
    for f in features:
        props = f.get("properties", {})
        day = _date_utc_iso(props)
        if not day:
            continue
        score = (_cloud_pct(props), _nodata_pct(props), -_updated_ts(props))  # lower is better; newer is better -> negative
        # Keep the best-scored item per day
        if day not in by_day or score < by_day[day]["__score"]:
            f_copy = dict(f)
            f_copy["__score"] = score
            by_day[day] = f_copy

    # Return items sorted by date
    best_daily = [v for _, v in sorted(by_day.items(), key=lambda x: x[0])]
    # Strip helper field
    for it in best_daily:
        it.pop("__score", None)
    return best_daily



# ---------- local (non-activity) step functions with SKIP guards ----------
def download_safe(product_id: str, token: str, download_dir: str) -> str:
    import time
    print("\n\nINSIDE          download_safe   +++++++++++++++++++++++++")
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(download_dir, f"{product_id}.zip")
    if _skip(file_path, "file"):
        return file_path

    odata_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq '{product_id}'"
    odata_response = requests.get(odata_url)
    odata_response.raise_for_status()
    odata_id = odata_response.json()["value"][0]["Id"]
    download_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({odata_id})/$value"

    headers = {"Authorization": f"Bearer {token}", "Accept": "*/*"}

    MAX_RETRIES = 5
    for i in range(MAX_RETRIES):
        response = requests.get(download_url, headers=headers, stream=True)
        if response.status_code == 429:
            print(f"⏳ Rate limited. Waiting before retrying ({i+1}/{MAX_RETRIES})...")
            time.sleep(2 ** i)
        else:
            response.raise_for_status()
            break
    else:
        raise RuntimeError("Too many retries: still getting 429 Too Many Requests")

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)
    return file_path

def unzip_safe(zip_path: str, final_target_path: str, short_temp_path: str = "C:/T") -> str:
    print("\n\nINSIDE          unzip_safe new   +++++++++++++++++++++++++")

    # If final .SAFE already exists with content, skip entirely
    if _skip(final_target_path, "dir"):
        return final_target_path

    def shorten_path_component(name, max_len=100):
        if len(name) <= max_len:
            return name
        base, ext = os.path.splitext(name)
        hash_part = hashlib.md5(name.encode()).hexdigest()[:6]
        return f"{base[:max_len-10]}_{hash_part}{ext}"

    folder_name = os.path.basename(zip_path).replace(".zip", "")
    temp_extract_path = os.path.join(short_temp_path, folder_name)
    Path(temp_extract_path).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith("/"):
                continue
            parts = member.split('/')
            shortened_parts = [shorten_path_component(p) for p in parts]
            target_path = os.path.join(temp_extract_path, *shortened_parts)
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)

    Path(final_target_path).mkdir(parents=True, exist_ok=True)

    entries = os.listdir(temp_extract_path)
    only_entry = os.path.join(temp_extract_path, entries[0])

    if only_entry.endswith(".SAFE") and os.path.isdir(only_entry):
        for item in os.listdir(only_entry):
            src = os.path.join(only_entry, item)
            dst = os.path.join(final_target_path, item)
            if _exists(dst):  # do not overwrite subfolders/files
                print(f"⚠️ Exists, skipping move of {dst}")
                continue
            shutil.move(src, dst)

    shutil.rmtree(temp_extract_path)
    return final_target_path

def convert_bands(safe_path: str, output_dir: str) -> str:
    print("\n\nINSIDE          convert_bands    +++++++++++++++++++++++++")
    print(f"🎨 Starting band conversion")
    print(f"📂 SAFE path: {safe_path}")
    print(f"📁 Output dir: {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # If cloud mask already exists, assume conversion done
    if _skip(os.path.join(output_dir, "cloud_mask.tif"), "file"):
        return output_dir

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
        if _skip(out_path, "file"):
            continue
        with rasterio.open(band_path) as src:
            profile = src.profile
            profile.update(driver="GTiff", compress="deflate")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(src.read())

    # SCL + cloud mask
    scl_path = next((f for f in band_files if "_SCL_" in f), None)
    if scl_path:
        print(f"🌥️ Found SCL band: {scl_path}")
        scl_out_path = os.path.join(output_dir, "SCL.tif")
        if not file_exists(scl_out_path):
            with rasterio.open(scl_path) as src:
                scl_data = src.read(1)
                profile = src.profile.copy()
                profile.update(dtype=rasterio.uint8, count=1, compress="deflate")
                with rasterio.open(scl_out_path, "w", **profile) as dst:
                    dst.write(scl_data, 1)
                print(f"🗺️ Saved SCL band: {scl_out_path}")

        cloud_out_path = os.path.join(output_dir, "cloud_mask.tif")
        if not file_exists(cloud_out_path):
            with rasterio.open(scl_path) as src:
                scl_data = src.read(1)
                cloud_mask = np.isin(scl_data, [3, 8, 9, 10, 11]).astype(np.uint8)
                profile = src.profile.copy()
                profile.update(dtype=rasterio.uint8, count=1, compress="deflate")
                with rasterio.open(cloud_out_path, "w", **profile) as dst:
                    dst.write(cloud_mask, 1)
                print(f"🌀 Saved cloud mask: {cloud_out_path}")
    else:
        print("⚠️ No SCL band found — cloud mask will be empty in merged output.")

    return output_dir

def merge_bands(scene_dir: str, output_path: str) -> str:
    print("\n\nINSIDE          merge_bands    +++++++++++++++++++++++++")
    print(f"🧩 Starting band merge")
    print(f"📂 Scene dir: {scene_dir}")
    print(f"📁 Output path: {output_path}")

    if _skip(output_path, "file"):
        return output_path

    bands_order = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    all_tifs = [f for f in os.listdir(scene_dir) if f.endswith(".tif") and not f.startswith("cloud_")]

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

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

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
    print("\n\nINSIDE          clip_to_AOI    +++++++++++++++++++++++++")
    if _skip(output_tif, "file"):
        return output_tif

    aoi = gpd.read_file(aoi_geojson_path)
    with rasterio.open(input_tif) as src:
        aoi_proj = aoi.to_crs(src.crs)
        aoi_geom = aoi_proj.geometry.values
        out_image, out_transform = mask(src, aoi_geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})

        Path(os.path.dirname(output_tif)).mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_tif, "w", **out_meta) as dst:
            dst.write(out_image)
    return output_tif

def generate_maps(input_tif: str, output_dir: str, model_dir: str) -> str:
    print("\n\nINSIDE          generate_bands    +++++++++++++++++++++++++")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    base = os.path.basename(input_tif)
    expected = [os.path.join(output_dir, f"{p}_{base}") for p in ["ndvi", "ndmi", "N_pred", "P_pred", "K_pred"]]
    if all(os.path.isfile(p) for p in expected):
        print("✅ All maps already exist, skipping.")
        return output_dir

    with rasterio.open(input_tif) as src:
        bands = src.read()
        height, width = bands.shape[1:]
        profile = src.profile.copy()
        profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)

    B = {"B2": bands[1], "B3": bands[2], "B4": bands[3], "B8": bands[7], "B12": bands[11]}
    with np.errstate(divide='ignore', invalid='ignore'):
        NDVI = (B["B8"] - B["B4"]) / (B["B8"] + B["B4"])
        NDMI = (B["B8"] - B["B12"]) / (B["B8"] + B["B12"])

    for name, arr in [("ndvi", NDVI), ("ndmi", NDMI)]:
        out_path = os.path.join(output_dir, f"{name}_{base}")
        if not file_exists(out_path):
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr.astype(np.float32), 1)

    valid_mask = np.all(bands != 0, axis=0)
    valid_flat = valid_mask.flatten()
    X = bands[:12].transpose(1, 2, 0).reshape(-1, 12)

    models = {}
    for nut in ["N", "P", "K"]:
        model_path = os.path.join(model_dir, f"rf_model_{nut}.joblib")
        models[nut] = joblib.load(model_path)

    height, width = bands.shape[1], bands.shape[2]
    for nut, model in models.items():
        out_path = os.path.join(output_dir, f"{nut}_pred_{base}")
        if file_exists(out_path):
            print(f"✅ {nut} already exists, skipping.")
            continue

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
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(y_pred.astype(np.float32), 1)
        print(f"🧪 Saved {nut} prediction map to {out_path}")

# ---------- optional end-to-end (unused by Temporal) ----------
@activity.defn
async def run_full_pipeline(config_path: str) -> str:
    print("\n\nINSIDE          run_full_pipeline    +++++++++++++++++++++++++\n\n")
    config_path = Path(config_path).resolve()
    print(f"📄 Using config: {config_path}")

    config = load_config(str(config_path))
    geojson = load_geojson(config["aoi"]["geojson_path"])
    creds = (await config)["credentials"] if asyncio.iscoroutine(config) else config["credentials"]
    date_range = (await config)["date_range"] if asyncio.iscoroutine(config) else config["date_range"]

    if asyncio.iscoroutine(geojson):
        geojson = await geojson

    project_name = config["project"]["name"]

    download_dir = os.path.join(project_name, "raw_downloads")
    unzip_dir = os.path.join(project_name, "unzipped")
    ind_bands_dir = os.path.join(project_name, "ind_bands")
    maps_output_dir = os.path.join(project_name, "maps")
    model_dir = config["models"]["directory"]

    token = get_token(creds["username"], creds["password"])
    if asyncio.iscoroutine(token):
        token = await token
    products = search_metadata(token, geojson, date_range["start"], date_range["end"])
    if asyncio.iscoroutine(products):
        products = await products

    print(f"🎯 {len(products)} product(s) found.")
    if not products:
        raise RuntimeError("❌ No products found.")

    for idx, prod in enumerate(products[:2]):
        product_id = prod["id"]
        print(f"\n📦 Processing product {idx+1}/{len(products)}: {product_id}")

        zip_path = os.path.join(download_dir, f"{product_id}.zip")
        safe_path = os.path.join(unzip_dir, f"{product_id}.SAFE")
        band_output = os.path.join(ind_bands_dir, product_id)
        merged_tif = os.path.join(project_name, "merged", f"merged_{product_id}.tif")
        clipped_tif = os.path.join(project_name, "clipped", f"clipped_{product_id}.tif")
        base_name = os.path.basename(clipped_tif)

        npk_paths = [os.path.join(maps_output_dir, f"{p}_{base_name}") for p in ["ndvi", "ndmi", "N_pred", "P_pred", "K_pred"]]

        if not _exists(zip_path, "file"):
            zip_path = download_safe(product_id, token, download_dir)
        if not _exists(safe_path, "dir"):
            safe_path = unzip_safe(zip_path, safe_path)
        if not _exists(os.path.join(band_output, "cloud_mask.tif"), "file"):
            convert_bands(safe_path, band_output)
        if not _exists(merged_tif, "file"):
            merge_bands(band_output, merged_tif)
        if not _exists(clipped_tif, "file"):
            clip_to_aoi(merged_tif, clipped_tif, config["aoi"]["geojson_path"])
        if not all(os.path.isfile(p) for p in npk_paths):
            generate_maps(clipped_tif, maps_output_dir, model_dir)

        print(f"✅ Finished processing {product_id}")

    return "✅ Pipeline complete."

# ---------- container step runner used by Temporal ----------
async def run_step_in_container(step: str, product_id: str, config_path: str) -> str:
    PROJECT_ROOT = Path(__file__).resolve().parent
    container_name = f"s2-{step}-{uuid.uuid4().hex[:6]}"

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{PROJECT_ROOT}:/app",
        "--name", container_name,
        "s2-product-pipeline:latest",
        "python", "run_one_product.py",
        "--step", step,
        "--product-id", product_id,
        "--config", f"/app/{config_path}",
    ]

    print(f"🚀 Running container for step: {step}, product: {product_id}")
    print(f"🐳 Command: {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"❌ Step `{step}` failed:\n{stderr.decode()}")
    return stdout.decode()

# Activity wrappers for Temporal
@activity.defn
async def download_container_activity(product_id: str, config_path: str) -> str:
    return await run_step_in_container("download", product_id, config_path)

@activity.defn
async def unzip_container_activity(product_id: str, config_path: str) -> str:
    return await run_step_in_container("unzip", product_id, config_path)

@activity.defn
async def convert_container_activity(product_id: str, config_path: str) -> str:
    return await run_step_in_container("convert", product_id, config_path)

@activity.defn
async def merge_container_activity(product_id: str, config_path: str) -> str:
    return await run_step_in_container("merge", product_id, config_path)

@activity.defn
async def clip_container_activity(product_id: str, config_path: str) -> str:
    return await run_step_in_container("clip", product_id, config_path)

@activity.defn
async def generate_maps_container_activity(product_id: str, config_path: str) -> str:
    return await run_step_in_container("maps", product_id, config_path)
