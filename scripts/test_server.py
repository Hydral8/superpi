
import asyncio
import numpy as np
from openpi_client import msgpack_numpy
import websockets.asyncio.client as _client

async def test_client():
    uri = "ws://localhost:8000"
    async with _client.connect(uri, close_timeout=120) as websocket:
        # 1. Receive Metadata
        metadata = msgpack_numpy.unpackb(await websocket.recv())
        print("Received Metadata.")
        
        # 2. Send Mock Observation
        obs = {
            "base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "state": np.zeros(14),
            "prompt": "pour a cup of coffee"
        }
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(obs))
        print("Sent Mock Observation.")
        
        # 3. Receive Action
        result = msgpack_numpy.unpackb(await websocket.recv())
        actions = result["actions"]
        print(f"Received Actions. Shape: {actions.shape}")
        
        if actions.shape[1] == 6:
            print("SUCCESS: Received 6-DOF actions!")
        else:
            print(f"FAILURE: Expected 6-DOF, got {actions.shape[1]}-DOF")

if __name__ == "__main__":
    asyncio.run(test_client())
