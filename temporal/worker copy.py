# worker.py
import asyncio
from temporalio.worker import Worker
from workflows import SentinelWorkflow
from activities import run_full_pipeline,run_product_in_container
from activities import (
    run_full_pipeline,
    run_product_in_container,
    load_config,
    get_token,
    search_metadata,
    load_geojson,
)
async def main():
    from temporalio.client import Client

    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="sentinel-task-queue",
        workflows=[SentinelWorkflow],
        activities=[run_full_pipeline,run_product_in_container,    load_config,
                    get_token,
                    search_metadata,
                    load_geojson,
                    ],
                    )

    print("🛠️ Worker running...")
    await worker.run()

# if __name__ == "__main__":
#     asyncio.run(main())
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Gracefully shutting down worker.") 