#!/usr/bin/env python3

import json
import requests
import urllib.parse
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Query the NLWeb server.")
    parser.add_argument("--query", type=str, help="The query to send to the server.")
    parser.add_argument("--site", type=str, default="all", help="The site to query (default: all).")
    parser.add_argument("--server", type=str, default="localhost:8000", help="The server address (default: localhost:8000).")
    
    args = parser.parse_args()

    # Get query, site, and server, using args or prompting if not provided
    query = args.query
    if not query:
        query = input("Enter your query: ")
        if not query:
            print("Query cannot be empty. Exiting.")
            return
    
    site = args.site
    if args.site == "all" and not any(arg.startswith('--site') for arg in sys.argv): # Only prompt if default is used AND not explicitly set
        prompted_site = input(f"Enter site (press Enter for '{args.site}'): ")
        if prompted_site:
            site = prompted_site
        else:
            print(f"Using default site: {site}")
    else:
        print(f"Using site: {site}")


    server = args.server
    if args.server == "localhost:8000" and not any(arg.startswith('--server') for arg in sys.argv): # Only prompt if default is used AND not explicitly set
        prompted_server = input(f"Enter server (press Enter for '{args.server}'): ")
        if prompted_server:
            server = prompted_server
        else:
            print(f"Using default server: {server}")
    else:
        print(f"Using server: {server}")
    
    # Encode the query
    print("Encoding query...")
    encoded_query = urllib.parse.quote(query)
    
    # Construct the URL
    url = f"http://{server}/ask"
    params = {
        "query": encoded_query,
        "site": site,
        "model": "auto",
        "prev": "[]",
        "item_to_remember": "",
        "context_url": "",
        "streaming": "false"
    }
    
    # payload = { # New JSON payload for POST
    #     "query": query, # Send raw query
    #     "site": site,
    #     "model": "auto",
    #     "prev": [], # Send as actual list
    #     "item_to_remember": "",
    #     "context_url": "",
    #     "streaming": False # Send as boolean
    # }
    
    # Make the request
    print(f"Contacting server at http://{server}...")
    print(f"Sending query: \"{query}\" for site: \"{site}\"...")
    
    try:
        response = requests.get(url, params=params)
        # response = requests.post(url, json=payload) # New POST request with JSON payload
        print("Response received. Processing data...")
        
        # Check if response is successful
        if response.status_code != 200:
            print(f"Error: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        # Parse JSON
        print("Parsing JSON response...")
        try:
            data = response.json()
        except json.JSONDecodeError:
            print("Failed to parse JSON response")
            print(f"Raw response: {response.text}")
            return
        
        # Extract and print results
        print("Extracting results...")
        if 'results' in data:
            results = data['results']
            print(f"Found {len(results)} results.")
            print("\nResults:")
            print("========")
            
            for i, item in enumerate(results, 1):
                name = item.get('name', 'No name available')
                description = item.get('description', 'No description available')
                print(f"{i}. {name}")
                print(f"   {description}\n")
        else:
            print("No 'results' field found in the response")
            print(f"Response keys: {list(data.keys())}")
        
        print("Done!")
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")