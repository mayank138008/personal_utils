# workflows.py
from temporalio import workflow
from datetime import timedelta
import asyncio

@workflow.defn
class SentinelWorkflow:
    @workflow.run
    async def run(self, config_path: str) -> str:
        logger = workflow.logger
        logger.info(f"📥 Workflow started with config: {config_path}")

        # === Load config and supporting data ===
        config = await workflow.execute_activity(
            "load_config",
            args=[config_path],
            start_to_close_timeout=timedelta(minutes=2),
        )

        geojson = await workflow.execute_activity(
            "load_geojson",
            args=[config["aoi"]["geojson_path"]],
            start_to_close_timeout=timedelta(minutes=2),
        )

        token = await workflow.execute_activity(
            "get_token",
            args=[config["credentials"]["username"], config["credentials"]["password"]],
            start_to_close_timeout=timedelta(minutes=2),
        )

        products = await workflow.execute_activity(
            "search_metadata",
            args=[token, geojson, config["date_range"]["start"], config["date_range"]["end"]],
            start_to_close_timeout=timedelta(minutes=5),
        )

        if not products:
            raise RuntimeError("❌ No products found.")

        logger.info(f"🎯 {len(products)} product(s) found. Starting processing...")

        async def process_one_product(product_id: str) -> str:
            logger.info(f"🚀 Starting container activities for: {product_id}")

            await workflow.execute_activity(
                "download_container_activity",
                args=[product_id, config_path],
                start_to_close_timeout=timedelta(minutes=30),
            )

            await workflow.execute_activity(
                "unzip_container_activity",
                args=[product_id, config_path],
                start_to_close_timeout=timedelta(minutes=30),
            )

            await workflow.execute_activity(
                "convert_container_activity",
                args=[product_id, config_path],
                start_to_close_timeout=timedelta(minutes=30),
            )

            await workflow.execute_activity(
                "merge_container_activity",
                args=[product_id, config_path],
                start_to_close_timeout=timedelta(minutes=30),
            )

            await workflow.execute_activity(
                "clip_container_activity",
                args=[product_id, config_path],
                start_to_close_timeout=timedelta(minutes=30),
            )

            await workflow.execute_activity(
                "generate_maps_container_activity",
                args=[product_id, config_path],
                start_to_close_timeout=timedelta(minutes=30),
            )

            logger.info(f"✅ Finished product: {product_id}")
            return f"✅ Done: {product_id}"

        # Fan out all product pipelines in parallel
        tasks = [
            process_one_product(prod["id"])
            for prod in products[:3]  # adjust limit as needed
        ]

        results = await asyncio.gather(*tasks)
        return "\n".join(results)
