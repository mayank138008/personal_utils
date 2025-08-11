# run_one_product.py
import argparse
import os
import sys
import yaml
from pathlib import Path
from sentinel_utils import (
    load_config, load_geojson, get_token,
    download_safe, unzip_safe, convert_bands,
    merge_bands, clip_to_aoi, generate_maps
)

# --- Fill missing args from ENV and optional runtime.yaml ---
def resolve_runtime_defaults(cli_step, cli_pid, cli_cfg):
    step = cli_step or os.getenv("STEP")
    product_id = cli_pid or os.getenv("PRODUCT_ID")
    config_path = cli_cfg or os.getenv("CONFIG_PATH")

    # Try optional runtime.yaml (any one of these if present)
    for candidate in (os.getenv("RUNTIME_YAML"), "/app/config/runtime.yaml", "./config/runtime.yaml"):
        if not (step and product_id and config_path) and candidate and os.path.isfile(candidate):
            try:
                with open(candidate, "r") as f:
                    y = yaml.safe_load(f) or {}
                step = step or y.get("step")
                product_id = product_id or y.get("product_id")
                config_path = config_path or y.get("config_path")
                break
            except Exception as e:
                print(f"⚠️ Failed to read {candidate}: {e}", file=sys.stderr)

    # Final default for config if still missing
    if not config_path:
        config_path = "/app/config/config.yaml"

    return step, product_id, config_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["download", "unzip", "convert", "merge", "clip", "maps"])
    p.add_argument("--product-id")
    p.add_argument("--config")
    args = p.parse_args()

    # Fill from ENV/runtime.yaml when CLI is absent
    step, product_id, config_path = resolve_runtime_defaults(args.step, args.product_id, args.config)
    if not (step and product_id and config_path):
        p.error("--step, --product-id, and --config are required (via CLI or env/runtime.yaml).")

    # Load config + AOI (token only needed for download step)
    config = load_config(config_path)
    geojson = load_geojson(config["aoi"]["geojson_path"])

    project = config["project"]["name"]

    # ---------- PATHS (fixed hierarchy: project/... not merged/project) ----------
    base = Path(project)
    download_dir   = base / "raw_downloads"
    unzip_dir      = base / "unzipped"
    ind_bands_dir  = base / "ind_bands"
    merged_tif     = base / "merged"  / f"merged_{product_id}.tif"
    clipped_tif    = base / "clipped" / f"clipped_{product_id}.tif"
    maps_output_dir= base / "maps"

    zip_path   = download_dir / f"{product_id}.zip"
    safe_path  = unzip_dir / f"{product_id}.SAFE"
    band_output= ind_bands_dir / product_id

    # Ensure target dirs exist
    for d in [download_dir, unzip_dir, ind_bands_dir, merged_tif.parent, clipped_tif.parent, maps_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"▶️ step={step}  product_id={product_id}  config={config_path}")

    # ---------- DISPATCH ----------
    if step == "download":
        creds = config["credentials"]
        token = get_token(creds["username"], creds["password"])
        download_safe(product_id, token, str(download_dir))

    elif step == "unzip":
        unzip_safe(str(zip_path), str(safe_path))

    elif step == "convert":
        convert_bands(str(safe_path), str(band_output))

    elif step == "merge":
        merge_bands(str(band_output), str(merged_tif))

    elif step == "clip":
        clip_to_aoi(str(merged_tif), str(clipped_tif), config["aoi"]["geojson_path"])

    elif step == "maps":
        model_dir = config["models"]["directory"]
        generate_maps(str(clipped_tif), str(maps_output_dir), model_dir)
