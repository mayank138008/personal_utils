import argparse
from sentinel_utils import run_one_product

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--product-id", required=True)
    args = parser.parse_args()

    print(f"⚙️ Running one product: {args.product_id}")
    run_one_product(args.product_id, args.config)