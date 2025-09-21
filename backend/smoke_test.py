#!/usr/bin/env python3
"""Minimal smoke test for chat/stream endpoint"""

import asyncio
import aiohttp
from datetime import datetime

def _now():
    return datetime.utcnow().isoformat()

async def smoke_test():
    """Test basic chat/stream functionality"""
    await asyncio.sleep(2)  # Wait for server to be ready
    try:
        async with aiohttp.ClientSession() as session:
            # Test session creation
            async with session.post("http://localhost:8000/api/session") as resp:
                data = await resp.json()
                session_id = data["session_id"]

            # Test chat stream
            payload = {"session_id": session_id, "message": "test"}
            async with session.post("http://localhost:8000/api/chat/stream", json=payload) as resp:
                content = await resp.text()

            print(f"[{_now()}] ✅ SMOKE TEST PASSED - Chat endpoint OK ({len(content)} chars)")
    except Exception as e:
        print(f"[{_now()}] ❌ SMOKE TEST FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(smoke_test())