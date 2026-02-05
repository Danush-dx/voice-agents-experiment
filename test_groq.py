import os
import asyncio
import time
from groq import AsyncGroq
from src.config import config

async def make_request(client, i):
    start_time = time.time()
    try:
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Request {i}: Say hello."}
            ],
            model=config.GROQ_LLM_MODEL,
            max_tokens=20,
        )
        duration = time.time() - start_time
        print(f"✅ Req {i}: Success ({duration:.4f}s)")
        return True
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Req {i}: Failed after {duration:.4f}s: {e}")
        return False

async def test_groq_concurrency():
    print(f"Testing Groq API with {config.GROQ_LLM_MODEL} - 20 Concurrent Requests")
    client = AsyncGroq(api_key=config.GROQ_API_KEY)
    
    tasks = [make_request(client, i) for i in range(1, 21)]
    
    start_total = time.time()
    results = await asyncio.gather(*tasks)
    total_duration = time.time() - start_total
    
    success_count = sum(results)
    print(f"\nSummary:")
    print(f"Total Requests: 20")
    print(f"Successful: {success_count}")
    print(f"Failed: {20 - success_count}")
    print(f"Total Time: {total_duration:.4f}s")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_groq_concurrency())
