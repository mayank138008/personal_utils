# worker.py
import asyncio
from temporalio.worker import Worker
from temporalio.client import Client

from workflows import SentinelWorkflow
from activities import (
    load_config,
    load_geojson,
    get_token,
    search_metadata,
    download_container_activity,
    unzip_container_activity,
    convert_container_activity,
    merge_container_activity,
    clip_container_activity,
    generate_maps_container_activity,
)

async def main():
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="sentinel-task-queue",
        workflows=[SentinelWorkflow],
        activities=[
            load_config,
            get_token,
            search_metadata,
            load_geojson,
            download_container_activity,
            unzip_container_activity,
            convert_container_activity,
            merge_container_activity,
            clip_container_activity,
            generate_maps_container_activity,
        ],
    )

    print("🛠️ Worker running...")
    await worker.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Gracefully shutting down worker.")
