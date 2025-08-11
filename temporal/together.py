
import os
import json
import yaml
import requests
import geopandas as gpd
import time

from datetime import datetime, date
from dotenv import load_dotenv


# step 1 functions =============================================================================


# === Token functions ===
def get_copernicus_token(username, password):
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"], response.json()["refresh_token"]


def renew_access_key():
    return get_copernicus_token(copernicus_username, copernicus_password)


# === Search and download ===
def search_sentinel2_data(token, refresh_token, geojson, start, end):
    url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "intersects": geojson["features"][0]["geometry"],
        "datetime": f"{start.isoformat()}Z/{end.isoformat()}Z",
        "collections": ["sentinel-2-l2a"],
        "limit": 100
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 401:
        print("⚠️ Token expired. Refreshing...")
        token, refresh_token = renew_access_key()
        headers["Authorization"] = f"Bearer {token}"
        response = requests.post(url, headers=headers, json=payload)

    response.raise_for_status()
    return response.json()["features"], token



# step 2 functions =============================================================================


def flatten_and_move_contents(src_dir, dest_dir):
    entries = os.listdir(src_dir)
    if len(entries) == 1:
        only_entry = os.path.join(src_dir, entries[0])
        if only_entry.endswith(".SAFE") and os.path.isdir(only_entry):
            print(f"📦 Flattening inner SAFE: {only_entry}")
            for item in os.listdir(only_entry):
                shutil.move(os.path.join(only_entry, item), os.path.join(dest_dir, item))
            return
    for item in entries:
        shutil.move(os.path.join(src_dir, item), os.path.join(dest_dir, item))


def safe_unzip(zip_path, final_target_path):
    zip_name = os.path.basename(zip_path)
    folder_name = zip_name.replace(".zip", "")
    temp_extract_path = os.path.join(short_temp_path, folder_name)
    os.makedirs(temp_extract_path, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                target_file = os.path.join(temp_extract_path, member.filename)
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                with zip_ref.open(member) as src, open(target_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        print(f"✅ Unzipped {zip_name} into temp path.")
    except Exception as e:
        print(f"❌ Unzip failed: {e}")
        return False

    try:
        os.makedirs(final_target_path, exist_ok=True)
        flatten_and_move_contents(temp_extract_path, final_target_path)
        shutil.rmtree(temp_extract_path)
        print(f"📂 Moved to: {final_target_path}")
        return True
    except Exception as e:
        print(f"❌ Move failed: {e}")
        return False



# === Main ===
if __name__ == "__main__":
   
    # === Load environment variables ===
    load_dotenv()

    # === Load YAML config ===
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # === Extract project info and set folder structure ===
    project = config["project"]
    project_name = project["name"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, project_name)

    download_root = os.path.join(project_root, project["download_folder"])
    clipped_root = os.path.join(project_root, project["clipped_folder"])
    output_root = os.path.join(project_root, project["output_folder"])

    os.makedirs(download_root, exist_ok=True)
    os.makedirs(clipped_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    print(f"📁 Download folder: {os.path.abspath(download_root)}")

    # === Get credentials ===
    copernicus_username = os.getenv("COPERNICUS_USERNAME") or config.get("copernicus", {}).get("username")
    copernicus_password = os.getenv("COPERNICUS_PASSWORD") or config.get("copernicus", {}).get("password")

    # === Config dates ===
    aoi_path = os.path.join(project_root, project["aoi_geojson"])
    start_date = project.get("start_date")
    end_date = project.get("end_date") or datetime.utcnow().strftime("%Y-%m-%d")

    if isinstance(start_date, date):
        start_dt = datetime.combine(start_date, datetime.min.time())
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    if isinstance(end_date, date):
        end_dt = datetime.combine(end_date, datetime.min.time())
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")




    if not copernicus_username or not copernicus_password:
        raise ValueError("Copernicus credentials not found.")

    token, refreshed_token = get_copernicus_token(copernicus_username, copernicus_password)
    print("✅ Access token obtained.")

    aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)
    geojson = json.loads(aoi.to_json())

    print(f"🔍 Searching between {start_dt.date()} and {end_dt.date()}...")

    try:
        features, token = search_sentinel2_data(token, refreshed_token, geojson, start_dt, end_dt)

        if not features:
            print("🔍 No imagery found.")
        else:
            print(f"📸 {len(features)} images found.")
            for feature in features:
                prod_id = feature["id"]
                dt = feature["properties"]["datetime"]

                odata_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq '{prod_id}'"
                odata_response = requests.get(odata_url)
                odata_response.raise_for_status()
                odata_id = odata_response.json()['value'][0]['Id']

                download_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({odata_id})/$value"
                headers = {"Authorization": f"Bearer {token}"}
                response = requests.get(download_url, headers=headers, stream=True)

                if response.status_code == 401:
                    print(f"🔁 Token might have expired. Refreshing for {prod_id}...")
                    token, refreshed_token = renew_access_key()
                    headers["Authorization"] = f"Bearer {token}"
                    response = requests.get(download_url, headers=headers, stream=True)

                if response.status_code == 200:
                    filename = os.path.join(download_root, f"{prod_id}.zip")
                    print(f"🔽 Downloading {filename}...")
                    with open(filename, "wb") as f:
                        for chunk in response.iter_content(8192):
                            f.write(chunk)
                    print("✅ Done.")
                else:
                    print(f"❌ Download failed after retry: {response.status_code} → {prod_id}")

                time.sleep(2)  # brief pause to avoid rate-limiting

    except Exception as e:
        print(f"⚠️ Error: {e}")



    # ===================================== step 2 file [UNZIP AND INDIVIDUAL BANDS]   ====================================================== 
    project = config["project"]
    process_safe = project.get("process_safe", True)
    project_name = project["name"]

    download_root = os.path.abspath(os.path.join(project_name, project["download_folder"]))
    unzipped_root = os.path.abspath(os.path.join(project_name, project["unzipped_folder"]))
    output_base = os.path.abspath(os.path.join(project_name, project["ind_bands"]))

    os.makedirs(output_base, exist_ok=True)
    os.makedirs(unzipped_root, exist_ok=True)

    short_temp_path = "D:/safe_tmp"
    os.makedirs(short_temp_path, exist_ok=True)

    print(f"📁 Scanning: {download_root}")
    safe_zip_files = [f for f in os.listdir(download_root) if f.endswith(".SAFE.zip")]



        
    # === Step 1: Unzip safely
    for zip_name in safe_zip_files:
        zip_path = os.path.join(download_root, zip_name)
        target_folder = os.path.join(unzipped_root, zip_name.replace(".zip", ""))

        if os.path.exists(target_folder) and os.path.isdir(target_folder):
            print(f"✅ Already unzipped: {zip_name}")
            continue

        print(f"📦 Extracting: {zip_name}")
        success = safe_unzip(zip_path, target_folder)
        if not success:
            print(f"🚫 Skipping {zip_name} due to extraction error.")



    # === Step 2: Process unzipped .SAFE folders
    if not process_safe:
        print("🚫 .SAFE processing skipped (disabled in config).")
        exit(0)

    safe_folders = [
        os.path.join(unzipped_root, f)
        for f in os.listdir(unzipped_root)
        if f.endswith(".SAFE") and os.path.isdir(os.path.join(unzipped_root, f))
    ]

    print(f"\n🔍 Found {len(safe_folders)} .SAFE folders.")

    for safe_path in safe_folders:
        scene_name = os.path.basename(safe_path).replace(".SAFE", "")
        scene_output_dir = os.path.join(output_base, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        # ✅ Skip if already processed
        existing_tifs = [f for f in os.listdir(scene_output_dir) if f.endswith(".tif")]
        if existing_tifs:
            print(f"⏭️ Skipping {scene_name}: Already converted to GeoTIFF.")
            continue

        print(f"\n📦 Processing: {scene_name}")
        granule_path = os.path.join(safe_path, "GRANULE")

        if not os.path.isdir(granule_path):
            print(f"❌ No GRANULE directory found in {granule_path}")
            print(f"📂 Contents: {os.listdir(safe_path)}")
            continue

        granule_subdirs = os.listdir(granule_path)
        if not granule_subdirs:
            print("❌ No subdirectories inside GRANULE.")
            continue

        granule_sub = os.path.join(granule_path, granule_subdirs[0])
        img_data_dir = os.path.join(granule_sub, "IMG_DATA")

        if not os.path.isdir(img_data_dir):
            print("❌ IMG_DATA not found.")
            continue

        band_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(img_data_dir)
            for file in files if file.endswith(".jp2")
        ]

        print(f"📸 Found {len(band_files)} bands.")
        
        # === STEP: Load SCL and generate cloud mask
        scl_band = next((f for f in band_files if "_SCL_" in f), None)
        cloud_mask = None
        if scl_band:
            with rasterio.open(scl_band) as scl_src:
                scl_data = scl_src.read(1)
                cloud_mask = np.isin(scl_data, [8, 9, 10])  # Cloud, shadows, cirrus
                mask_profile = scl_src.profile.copy()
                mask_profile.update(dtype=rasterio.uint8, count=1, driver="GTiff")

                # Save cloud mask
                cloud_mask_path = os.path.join(scene_output_dir, "cloud_mask.tif")
                with rasterio.open(cloud_mask_path, "w", **mask_profile) as dst:
                    dst.write(cloud_mask.astype(np.uint8), 1)
                print(f"☁️ Cloud mask saved to: {cloud_mask_path}")
        else:
            print("⚠️ No SCL band found; cloud masking skipped.")

        # === Save bands (applying cloud mask if available)
        for band_path in band_files:
            print("  -", os.path.basename(band_path))
            try:
                with rasterio.open(band_path) as src:
                    out_name = os.path.basename(band_path).replace(".jp2", ".tif")
                    out_path = os.path.join(scene_output_dir, out_name)

                    profile = src.profile
                    profile.update(driver="GTiff")

                    data = src.read()

                    if cloud_mask is not None and "_SCL_" not in band_path and data.shape[1:] == cloud_mask.shape:
                        data[:, cloud_mask] = 0  # or np.nan if dtype is float and you prefer
                        print(f"☁️ Applied cloud mask to: {out_name}")

                    with rasterio.open(out_path, "w", **profile) as dst:
                        dst.write(data)

                    print(f"✅ Saved: {out_name}")
            except Exception as e:
                print(f"❌ Failed to save {band_path}: {e}")


    # ===================================== step 3 MERGE_2   ====================================================== 

    project = config["project"]
    project_name = project["name"]
    download_root = os.path.abspath(os.path.join(project_name, project["download_folder"]))
    output_root = os.path.abspath(os.path.join(project_name, "merged_bands"))
    os.makedirs(output_root, exist_ok=True)

    # List of bands in correct order
    bands_order = [
        "B01", "B02", "B03", "B04",
        "B05", "B06", "B07",
        "B08", "B8A", "B09", "B11", "B12"
    ]

    # Get per-scene folders from outputs/
    clipped_root = os.path.abspath(os.path.join(project_name, project["ind_bands"]))
    scene_folders = [
        os.path.join(clipped_root, f)
        for f in os.listdir(clipped_root)
        if os.path.isdir(os.path.join(clipped_root, f))
    ]

    for scene_folder in scene_folders:
        scene_name = os.path.basename(scene_folder)
        output_path = os.path.join(output_root, f"{scene_name}.tif")

        # ✅ Skip if output already exists
        if os.path.exists(output_path):
            print(f"⏭️ Skipping {scene_name} (merged file already exists)\n")
            continue

        print(f"🔍 Checking scene: {scene_name}")

        all_tifs = [f for f in os.listdir(scene_folder) if f.endswith(".tif")]
        if not all_tifs:
            print(f"❌ Skipping {scene_name} (no .tif files found)\n")
            continue

        # Try to find a sample band to copy metadata
        sample_path = next((os.path.join(scene_folder, f) for f in all_tifs if any(b in f for b in bands_order)), None)
        if not sample_path:
            print(f"❌ Skipping {scene_name} (no matching bands found)\n")
            continue

        with rasterio.open(sample_path) as src0:
            meta = src0.meta.copy()
            meta.update({
                "count": len(bands_order) + 1,  # +1 for cloud mask
                "driver": "GTiff",
                "compress": "deflate",
                "dtype": src0.dtypes[0],
                "crs": src0.crs if src0.crs else "EPSG:32635"
            })

        print(f"💾 Creating merged output at: {output_path}")

        with rasterio.open(output_path, "w", **meta) as dst:
            # Write spectral bands
            for idx, band_name in enumerate(bands_order, start=1):
                band_file = next((f for f in all_tifs if f"_{band_name}_" in f), None)
                if not band_file:
                    print(f"⚠️ Missing {band_name} for {scene_name} (filling with zeros)")
                    dst.write(np.zeros((meta["height"], meta["width"]), dtype=meta["dtype"]), idx)
                    continue

                band_path = os.path.join(scene_folder, band_file)
                print(f"   └─ Reading {band_file} as Band {idx}")

                with rasterio.open(band_path) as src:
                    dst.write(src.read(1), idx)

            # Write cloud mask as final band
            cloud_mask_path = os.path.join(scene_folder, "cloud_mask.tif")
            if os.path.exists(cloud_mask_path):
                with rasterio.open(cloud_mask_path) as mask_src:
                    mask_data = mask_src.read(1)
                    dst.write(mask_data, len(bands_order) + 1)
                    print(f"☁️ Appended cloud mask as Band {len(bands_order) + 1}")
            else:
                print("⚠️ No cloud_mask.tif found, filling last band with zeros")
                dst.write(np.zeros((meta["height"], meta["width"]), dtype=meta["dtype"]), len(bands_order) + 1)

        print(f"🎉 Merged file saved to: {output_path}\n")



    # ===================================== step 4 AOI_MASK AFTER MERGE_2   ====================================================== 
        
    project = config["project"]
    project_name = project["name"]

    # === Paths
    merged_root = os.path.abspath(os.path.join(project_name, "merged_bands"))
    clipped_merged_root = os.path.abspath(os.path.join(project_name, "clipped_after_merge"))
    os.makedirs(clipped_merged_root, exist_ok=True)

    # === Load AOI
    aoi_path = os.path.abspath(os.path.join(project_name, project["aoi_geojson"]))
    aoi = gpd.read_file(aoi_path)
    print("🗺️ AOI loaded:", aoi_path)

    # === Process each .tif file
    merged_files = [f for f in os.listdir(merged_root) if f.endswith(".tif")]

    for file in merged_files:
        input_path = os.path.join(merged_root, file)
        output_path = os.path.join(clipped_merged_root, file)

        print(f"\n✂️  Clipping: {file}")

        try:
            with rasterio.open(input_path) as src:
                # Reproject AOI to match raster CRS
                aoi_proj = aoi.to_crs(src.crs)
                aoi_geom = aoi_proj.geometry.values

                # Check if AOI intersects raster bounds
                raster_bounds = box(*src.bounds)
                if not aoi_proj.intersects(raster_bounds).any():
                    print("🚫 Skipped: AOI does not overlap raster.")
                    continue

                # Clip raster using the AOI
                out_image, out_transform = mask(src, aoi_geom, crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                with rasterio.open(output_path, "w", **out_meta) as dst:
                    dst.write(out_image)

            print(f"✅ Saved clipped: {output_path}")

        except Exception as e:
            print(f"❌ Failed to clip {file}: {e}")
    
    # ===================================== step 4 AOI_MASK AFTER MERGE_2   ====================================================== 


        
    project = config["project"]
    project_name = project["name"]

    # === Paths
    merged_root = os.path.abspath(os.path.join(project_name, "merged_bands"))
    clipped_merged_root = os.path.abspath(os.path.join(project_name, "clipped_after_merge"))
    os.makedirs(clipped_merged_root, exist_ok=True)

    # === Load AOI
    aoi_path = os.path.abspath(os.path.join(project_name, project["aoi_geojson"]))
    aoi = gpd.read_file(aoi_path)
    print("🗺️ AOI loaded:", aoi_path)

    # === Process each .tif file
    merged_files = [f for f in os.listdir(merged_root) if f.endswith(".tif")]

    for file in merged_files:
        input_path = os.path.join(merged_root, file)
        output_path = os.path.join(clipped_merged_root, file)

        print(f"\n✂️  Clipping: {file}")

        try:
            with rasterio.open(input_path) as src:
                # Reproject AOI to match raster CRS
                aoi_proj = aoi.to_crs(src.crs)
                aoi_geom = aoi_proj.geometry.values

                # Check if AOI intersects raster bounds
                raster_bounds = box(*src.bounds)
                if not aoi_proj.intersects(raster_bounds).any():
                    print("🚫 Skipped: AOI does not overlap raster.")
                    continue

                # Clip raster using the AOI
                out_image, out_transform = mask(src, aoi_geom, crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                with rasterio.open(output_path, "w", **out_meta) as dst:
                    dst.write(out_image)

            print(f"✅ Saved clipped: {output_path}")

        except Exception as e:
            print(f"❌ Failed to clip {file}: {e}")



    # ===================================== step 10_11 npk  AND OTHER MAPS   ====================================================== 


    import os
    import joblib
    import rasterio
    import numpy as np
    import folium
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    import pandas as pd
    from PIL import Image
    import matplotlib.pyplot as plt

    # === Paths ===
    input_folder = r"E:\D_2\DA_3\sentinel_download\cruz\clipped_after_merge"
    output_folder = os.path.join(input_folder, "npk_moisture_health_maps_3")
    npk_output = os.path.join(output_folder, "npk")
    moisture_output = os.path.join(output_folder, "moisture")
    health_output = os.path.join(output_folder, "health")
    model_folder = r"D:\one_drive_personal\OneDrive\apple sync\ALL_CODES\DA_2\WORKS\models"

    os.makedirs(npk_output, exist_ok=True)
    os.makedirs(moisture_output, exist_ok=True)
    os.makedirs(health_output, exist_ok=True)

    # === Load Models ===
    model_paths = {
        'N': os.path.join(model_folder, 'rf_model_N.joblib'),
        'P': os.path.join(model_folder, 'rf_model_P.joblib'),
        'K': os.path.join(model_folder, 'rf_model_K.joblib')
    }
    models = {nutrient: joblib.load(path) for nutrient, path in model_paths.items()}

    # === Utilities ===
    def compute_index(numerator, denominator):
        return np.where((denominator != 0), (numerator - denominator) / (numerator + denominator), 0)

    def render_folium_overlay_from_array(array, geotiff_path, title, output_html, vmin=None, vmax=None):
        try:
            with rasterio.open(geotiff_path) as src:
                bounds = src.bounds
                src_crs = src.crs
                if src_crs is None or not src_crs.is_valid:
                    print(f"❌ Invalid CRS in: {geotiff_path}")
                    return
                bounds_latlon = transform_bounds(src_crs, 'EPSG:4326', *bounds)

            [[west, south], [east, north]] = [[bounds_latlon[0], bounds_latlon[1]], [bounds_latlon[2], bounds_latlon[3]]]
            center_lat = (south + north) / 2
            center_lon = (west + east) / 2

            # Convert array to PNG using matplotlib colormap
            norm = plt.Normalize(vmin=np.nanmin(array) if vmin is None else vmin,
                                vmax=np.nanmax(array) if vmax is None else vmax)
            cmap = plt.get_cmap("viridis")
            rgba_img = (cmap(norm(array)) * 255).astype(np.uint8)
            png_path = output_html.replace(".html", ".png")
            Image.fromarray(rgba_img).save(png_path)

            # Render folium map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        attr="Tiles © Esri")

            folium.raster_layers.ImageOverlay(
                name=title,
                image=png_path,
                bounds=[[south, west], [north, east]],
                opacity=0.6,
                interactive=True
            ).add_to(m)

            folium.LayerControl().add_to(m)
            m.save(output_html)
            print(f"✅ Map saved to: {output_html}")
        except Exception as e:
            print(f"⚠️ Error generating map for {geotiff_path}: {e}")

    # === Main Processing Loop ===
    overlay_maps = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            filepath = os.path.join(input_folder, filename)
            print(f"🔄 Processing: {filename}")

            with rasterio.open(filepath) as src:
                bands = src.read()
                if bands.dtype in [np.float32, np.float64] and bands.max() < 1.5:
                    bands = bands * 10000
                profile = src.profile
                crs = profile.get('crs')
                if crs is None or not crs.is_valid:
                    print(f"⚠️ Invalid CRS in {filename}, defaulting to EPSG:32636")
                    crs = CRS.from_epsg(32636)

            # === Mask where any band == 0
            valid_mask = np.all(bands != 0, axis=0)

            # === Compute NDMI and NDVI ===
            B = {f'B{i+1}': bands[i] for i in range(12)}
            NDMI = compute_index(B['B8'], B['B12'])
            NDVI = compute_index(B['B8'], B['B4'])

            NDMI[~valid_mask] = np.nan
            NDVI[~valid_mask] = np.nan

            profile.update(count=1, dtype=rasterio.float32, crs=crs, nodata=np.nan)

            ndmi_path = os.path.join(moisture_output, f"ndmi_{filename}")
            ndvi_path = os.path.join(health_output, f"ndvi_{filename}")
            with rasterio.open(ndmi_path, "w", **profile) as dst:
                dst.write(NDMI.astype(np.float32), 1)
            with rasterio.open(ndvi_path, "w", **profile) as dst:
                dst.write(NDVI.astype(np.float32), 1)

            render_folium_overlay_from_array(NDMI, ndmi_path, "NDMI", ndmi_path.replace(".tif", ".html"))
            render_folium_overlay_from_array(NDVI, ndvi_path, "NDVI", ndvi_path.replace(".tif", ".html"))

            # === Predict NPK continuous values ===
            height, width = bands.shape[1:]
            X_all = np.stack([bands[i] for i in range(12)], axis=-1).reshape(-1, 12)
            valid_flat = valid_mask.flatten()

            for nutrient, model in models.items():
                y_pred = np.full((height * width), np.nan, dtype=np.float32)
                y_pred[valid_flat] = model.predict(X_all[valid_flat])
                y_pred = y_pred.reshape(height, width)

                profile.update(dtype=rasterio.float32, nodata=np.nan)
                cont_tif = os.path.join(npk_output, f"{nutrient.lower()}_pred_{filename}")
                with rasterio.open(cont_tif, "w", **profile) as dst:
                    dst.write(y_pred.astype(np.float32), 1)

                html_path = cont_tif.replace(".tif", ".html")
                render_folium_overlay_from_array(y_pred, cont_tif, f"{nutrient} Prediction", html_path)
                overlay_maps.append((nutrient, html_path))