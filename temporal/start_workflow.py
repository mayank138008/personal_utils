# start_workflow.py
import asyncio
from temporalio.client import Client
import uuid

async def main():
    client = await Client.connect("localhost:7233")

    result = await client.start_workflow(
        "SentinelWorkflow",
        "config/config.yaml",  # path to your config file
        id=f"sentinel-{uuid.uuid4()}",
        task_queue="sentinel-task-queue",
    )

    print(f"🚀 Workflow started: {result.id}")
    res = await result.result()
    print(f"✅ Workflow result: {res}")

if __name__ == "__main__":
    asyncio.run(main())
