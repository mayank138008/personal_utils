# ==============================================================================================================
# workflows.py
from temporalio import workflow
from datetime import timedelta
from typing import List
# from activities import load_config, get_token, search_metadata, load_geojson
import temporalio.activity



from temporalio import workflow
from datetime import timedelta
import asyncio  # ✅ Required for gather

@workflow.defn
class SentinelWorkflow:
    @workflow.run
    async def run(self, config_path: str) -> str:
        logger = workflow.logger
        logger.info(f"📥 Workflow started with config: {config_path}")

        config = await workflow.execute_activity(
            "load_config", args=[config_path], start_to_close_timeout=timedelta(minutes=2)
        )

        geojson = await workflow.execute_activity(
            "load_geojson", args=[config["aoi"]["geojson_path"]], start_to_close_timeout=timedelta(minutes=2)
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

        # print("products===================================================\n",products)
        if not products:
            raise RuntimeError("❌ No products found.")

        logger.info(f"🎯 {len(products)} product(s) found. Starting containers...")

        # ✅ TRUE PARALLELISM
        tasks = [
            workflow.execute_activity(
                "run_product_in_container",
                args=[prod["id"], config_path],
                start_to_close_timeout=timedelta(hours=1),
            )
            for prod in products[:6]
        ]

        results = await asyncio.gather(*tasks)  # ✅ THIS is the magic line

        return "\n".join(results)
