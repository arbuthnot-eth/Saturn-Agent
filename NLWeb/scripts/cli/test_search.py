#!/usr/bin/env python3
import sys
sys.path.append('../../code')
from retrieval.vectorize_client import CloudflareVectorizeClient
import asyncio

async def test_search():
    client = CloudflareVectorizeClient('vectorize_primary')
    print("Testing search with 'latest news' query...")
    
    # Test with a simple query and see what happens
    results = await client.search('latest news', query_vector=None, num_results=5, site_filter='bbcnews')
    print(f'Search returned {len(results)} results')
    
    for i, result in enumerate(results):
        score = result.get("score", "N/A")
        title = result.get("metadata", {}).get("title", "N/A")
        text = result.get("metadata", {}).get("text", "N/A")[:100] + "..."
        print(f'Result {i+1}: score={score}, title={title}, text={text}')

if __name__ == "__main__":
    asyncio.run(test_search()) 