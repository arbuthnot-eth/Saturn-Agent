#!/usr/bin/env python3
from retrieval.vectorize_client import CloudflareVectorizeClient
import asyncio
import json

async def test_direct_search_string_bool():
    client = CloudflareVectorizeClient('cloudflare_vectorize')
    print("Testing direct CloudflareVectorizeClient search with string booleans...")
    
    payload = {
        "topK": 3,
        "returnValues": True, # Python bool
        "returnMetadata": "true", # String "true"
        "filter": {"site": "bbcnews"},
        "vector": [0.0] * 768
    }
    
    try:
        # Use the internal _request method directly to bypass client.search modifications for this test
        # This also means we are directly passing the payload to httpx via the json= parameter
        response = await client._request("POST", "query", json=payload)
        response_data = response.json()
        print("SUCCESS! Response received:")
        print(f"Status: {response.status_code}")
        # print(json.dumps(response_data, indent=2))
        if response_data and response_data.get("result") and response_data["result"].get("matches"):
            matches = response_data["result"]["matches"]
            print(f"Found {len(matches)} matches.")
        else:
            print("No matches found or unexpected response structure.")
            print(f"Full response: {json.dumps(response_data, indent=2)}")

    except Exception as e:
        print(f"Error: {e}")

async def test_direct_search():
    client = CloudflareVectorizeClient('cloudflare_vectorize')
    print("Testing direct CloudflareVectorizeClient search...")
    
    # Create a simple vector
    dummy_vector = [0.0] * 768
    
    # Test with a simple query
    results = await client.search('latest news', query_vector=dummy_vector, num_results=3, site_filter='bbcnews')
    print(f'Direct search returned {len(results)} results')
    
    for i, result in enumerate(results):
        score = result.get("score", "N/A")
        title = result.get("title", "N/A")
        text = result.get("text", "N/A")[:100] + "..."
        print(f'Result {i+1}: score={score}, title={title}, text={text}')

async def test_manual_json():
    client = CloudflareVectorizeClient('cloudflare_vectorize')
    print("Testing with manual JSON serialization...")
    
    # Create the payload
    payload = {
        "topK": 3,
        "returnValues": True,
        "returnMetadata": True,
        "filter": {"site": "bbcnews"},
        "vector": [0.0] * 768
    }
    
    # Manually serialize
    json_str = json.dumps(payload)
    print(f"JSON length: {len(json_str)}")
    print(f"JSON start: {json_str[:100]}...")
    
    # Send as content instead of json parameter
    try:
        response = await client._request("POST", "query", content=json_str.encode('utf-8'))
        response_data = response.json()
        print("SUCCESS! Response received:")
        print(f"Status: {response.status_code}")
        print(f"Response keys: {list(response_data.keys())}")
        
        # Parse results like the normal search method
        result_data = response_data.get("result", {})
        matches = result_data.get("matches", [])
        print(f"Found {len(matches)} matches")
        
    except Exception as e:
        print(f"Error: {e}")

async def test_debug_method_direct():
    client = CloudflareVectorizeClient('cloudflare_vectorize')
    print("Testing debug_list_vectors_by_site directly...")
    try:
        response_data = await client.debug_list_vectors_by_site('bbcnews', limit=3)
        print("SUCCESS! Response from debug_list_vectors_by_site:")
        # print(json.dumps(response_data, indent=2))
        # Simplified output:
        if response_data and response_data.get("result") and response_data["result"].get("matches"):
            matches = response_data["result"]["matches"]
            print(f"Found {len(matches)} matches.")
            for i, match in enumerate(matches):
                print(f"Match {i+1} ID: {match.get('id')}, Score: {match.get('score')}")
                if match.get('metadata'):
                    print(f"  Metadata site: {match['metadata'].get('site')}, title: {match['metadata'].get('title')}")
        else:
            print("No matches found or unexpected response structure.")
            print(f"Full response: {json.dumps(response_data, indent=2)}")
            
    except Exception as e:
        print(f"Error calling debug_list_vectors_by_site: {e}")

if __name__ == "__main__":
    # asyncio.run(test_direct_search_string_bool())
    asyncio.run(test_direct_search())
    # asyncio.run(test_manual_json())
    # asyncio.run(test_debug_method_direct()) 