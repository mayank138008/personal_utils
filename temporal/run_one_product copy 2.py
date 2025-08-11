# run_one_product.py
import argparse
from sentinel_utils import *
from pathlib import Path
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=[
        "download", "unzip", "convert", "merge", "clip", "maps"
    ])
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    geojson = load_geojson(config["aoi"]["geojson_path"])
    creds = config["credentials"]
    token = get_token(creds["username"], creds["password"])
    product_id = args.product_id

    project = config["project"]["name"]
    download_dir = os.path.join(project, "raw_downloads")
    unzip_dir = os.path.join(project, "unzipped")
    ind_bands_dir = os.path.join(project, "ind_bands")
    
    # merged_tif = os.path.join("merged", project, f"merged_{product_id}.tif")
    # clipped_tif = os.path.join("clipped", project, f"clipped_{product_id}.tif")

    merged_tif  = os.path.join(project, "merged",  f"merged_{product_id}.tif")
    clipped_tif = os.path.join(project, "clipped", f"clipped_{product_id}.tif")


    maps_output_dir = os.path.join(project, "maps")
    model_dir = config["models"]["directory"]

    zip_path = os.path.join(download_dir, f"{product_id}.zip")
    safe_path = os.path.join(unzip_dir, f"{product_id}.SAFE")
    band_output = os.path.join(ind_bands_dir, product_id)

    if args.step == "download":
        download_safe(product_id, token, download_dir)

    elif args.step == "unzip":
        unzip_safe(zip_path, safe_path)

    elif args.step == "convert":
        convert_bands(safe_path, band_output)

    elif args.step == "merge":
        merge_bands(band_output, merged_tif)

    elif args.step == "clip":
        clip_to_aoi(merged_tif, clipped_tif, config["aoi"]["geojson_path"])

    elif args.step == "maps":
        generate_maps(clipped_tif, maps_output_dir, model_dir)
