# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Cloudflare Vectorize Client implementation.
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional, Union
import json

from config.config import CONFIG, RetrievalProviderConfig
from retrieval.interfaces import VectorDBClientInterface # Updated import
from utils.logging_config_helper import get_configured_logger

logger = get_configured_logger(__name__)

# Cloudflare API base URL (adjust if necessary, though usually part of specific API calls)
CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"

class CloudflareVectorizeClient(VectorDBClientInterface):
    """
    Client for interacting with Cloudflare Vectorize.
    """

    def __init__(self, endpoint_name: str):
        logger.critical("CRITICAL_VCLIENT_LOG: CloudflareVectorizeClient __init__ called")
        self.endpoint_name = endpoint_name
        self.config: RetrievalProviderConfig = CONFIG.retrieval_endpoints.get(endpoint_name)
        if not self.config:
            raise ValueError(f"Configuration for endpoint '{endpoint_name}' not found.")
        if not self.config.db_type == "vectorize":
            raise ValueError(f"Endpoint '{endpoint_name}' is not configured for Cloudflare Vectorize (db_type 'vectorize').")

        self.api_key = self.config.api_key
        self.account_id = self.config.account_id
        self.index_name = self.config.index_name # Assuming 'index' in YAML maps to 'index_name'

        if not all([self.api_key, self.account_id, self.index_name]):
            missing = []
            if not self.api_key: missing.append("api_key (CLOUDFLARE_API_TOKEN)")
            if not self.account_id: missing.append("account_id")
            if not self.index_name: missing.append("index_name (VECTORIZE_INDEX_NAME)")
            raise ValueError(f"Missing Cloudflare Vectorize configuration for endpoint '{endpoint_name}': {', '.join(missing)}")

        logger.info(f"CloudflareVectorizeClient initialized for endpoint: {endpoint_name}, index: {self.index_name}")

    def _sanitize_payload_for_logging(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitizes a copy of the payload for logging, abbreviating long vector lists."""
        payload_copy = payload.copy() # Avoid modifying the original payload
        if "vector" in payload_copy and isinstance(payload_copy["vector"], list):
            vector_list = payload_copy["vector"]
            if len(vector_list) > 10: # Only abbreviate if long
                payload_copy["vector"] = str(vector_list[:5])[:-1] + ", ..., " + str(vector_list[-5:])[1:] + f" (Total {len(vector_list)} items)"
            else:
                payload_copy["vector"] = str(vector_list) # Keep as string if short
        return payload_copy

    async def _request(self, method: str, path: str, is_ndjson: bool = False, **kwargs) -> httpx.Response:
        logger.critical(f"CRITICAL_VCLIENT_LOG: _request called with method='{method}', path='{path}'")
        url = f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/vectorize/v2/indexes/{self.index_name}/{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            # Content-Type will be set based on is_ndjson or default
        }
        if is_ndjson:
            headers["Content-Type"] = "application/x-ndjson"
        else:
            headers["Content-Type"] = "application/json" # Default for other requests like query

        async with httpx.AsyncClient() as client:
            payload_info = "No Payload" # Default if neither json nor content is present
            if 'json' in kwargs: # For standard JSON requests
                json_payload = kwargs['json']
                if isinstance(json_payload, dict):
                    payload_info = f"JSON Dict - Keys: {list(json_payload.keys())}"
                elif isinstance(json_payload, list):
                    payload_info = f"JSON List of {len(json_payload)} items"
                else:
                    payload_info = "Non-dict/list JSON payload"
            elif 'content' in kwargs: # For NDJSON or other raw content requests
                if is_ndjson:
                    payload_info = f"NDJSON data of {len(kwargs['content'])} bytes"
                else:
                    payload_info = f"Raw content of {len(kwargs['content'])} bytes"
            
            logger.debug(f"Cloudflare Vectorize request: {method} {url} Headers: {headers} Payload: {payload_info}")
            try:
                response = await client.request(method, url, headers=headers, **kwargs)
                logger.debug(f"Cloudflare Vectorize API Raw Response Status: {response.status_code}")
                logger.debug(f"Cloudflare Vectorize API Raw Response Body: {response.text[:1000]}...") # Log first 1000 chars
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                logger.error(f"Cloudflare Vectorize API error: {e.response.status_code} - {e.response.text}")
                # Try to parse JSON error if possible
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {error_details}")
                except Exception:
                    pass # Ignore if error response is not JSON
                raise
            except httpx.RequestError as e:
                logger.error(f"Cloudflare Vectorize request error: {e}")
                raise

    async def upload_documents(self, documents: List[Dict[str, Any]], site_name: Optional[str] = None) -> int:
        if not documents:
            return 0

        logger.info(f"CloudflareVectorizeClient: Starting upload of {len(documents)} documents for site: {site_name}")
        successful_uploads = 0
        # Cloudflare Vectorize API has a limit of 1MB per request and 1000 vectors per request.
        # We'll batch by 100 for now, assuming vectors are not excessively large.
        # Max 2MB per request, max 1,000 vectors. Recommended 100-500.
        # https://developers.cloudflare.com/vectorize/platform/limits/
        batch_size = 100 #self.config.get("batch_size", 100)

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectors_to_upsert = []
            for doc in batch:
                # Log the structure of the doc received by this method
                logger.critical(f"CRITICAL_VCLIENT_UPLOAD_DOC_RECEIVED: {str(doc)[:500]}...") # Log first 500 chars

                doc_id = str(doc.get("id"))
                doc_embedding = doc.get("embedding") # CORRECTED: Was doc.get("vector")

                if not doc_embedding or not doc_id:
                    logger.warning(f"Document missing 'embedding' or 'id', skipping ID: {doc_id if doc_id else 'ID_MISSING'}. Embedding found: {doc_embedding is not None}")
                    logger.debug(f"Problematic doc structure: {str(doc)[:500]}...")
                    continue

                # Constructing metadata from available fields
                page_content = doc.get("page_content", "") # Try page_content first
                if not page_content:
                    page_content = doc.get("schema_json", "") # Fallback to schema_json if page_content is empty/missing
                
                doc_meta_field = doc.get("metadata", {})   # <--- doc_meta_field is defined here

                # ---- START NEW LOGGING (1/2) ----
                logger.critical(f"CRITICAL_VCLIENT_UPLOAD_DOC_DETAILS: doc_id={doc_id}, page_content_present={bool(page_content)}, page_content_len={len(page_content if page_content else '')}, doc_meta_field_keys={list(doc_meta_field.keys()) if doc_meta_field else 'None'}")
                # ---- END NEW LOGGING (1/2) ----

                # Robustly get metadata, trying sub-dictionary then top-level, then empty string
                url_val = str(doc_meta_field.get("url", doc.get("url", "")))
                title_val = str(doc_meta_field.get("title", doc.get("name", "")))
                publish_date_val = str(doc_meta_field.get("publish_date", doc.get("publish_date", "")))

                metadata_to_store = { # <--- metadata_to_store is populated here
                    "text": page_content,  # Full text content
                    "url": url_val,
                    "title": title_val,
                    "publish_date": publish_date_val,
                    "site": site_name,  # Add site_name here for filtering
                    "original_doc_id": doc_id # Keep original ID if needed
                }
                # ---- START NEW LOGGING (2/2) ----
                logger.critical(f"CRITICAL_VCLIENT_METADATA_INITIAL: For doc ID {doc_id}, site_name='{site_name}', metadata_initial={json.dumps(metadata_to_store)}")
                # ---- END NEW LOGGING (2/2) ----
                
                # Filter out None/empty string values and ensure all are strings for Cloudflare
                # Vectorize metadata values must be strings, numbers, or booleans.
                # We'll stringify all for simplicity and to avoid issues with None.
                current_metadata_keys = list(metadata_to_store.keys()) # Avoid issues with changing dict during iteration
                for k in current_metadata_keys:
                    v = metadata_to_store[k]
                    if v is None: # Remove if None
                        del metadata_to_store[k]
                    else:
                        v_str = str(v)
                        if v_str == "": # Remove if empty string after conversion
                             del metadata_to_store[k]
                        else:
                            metadata_to_store[k] = v_str # Store as string

                logger.debug(f"Preparing vector with ID {doc_id} for site: '{site_name}'. Metadata: {metadata_to_store}")
                # ---- EXISTING CRITICAL LOG (shows metadata *after* filtering) ----
                logger.critical(f"CRITICAL_VCLIENT_METADATA_PRE_UPLOAD: For doc ID {doc_id}, site_name='{site_name}', metadata_to_store={json.dumps(metadata_to_store)}")
                # ---- END EXISTING CRITICAL LOG ----

                vectors_to_upsert.append(
                    {
                        "id": doc_id, # This ID must be unique for each vector in the index
                        "values": doc_embedding,
                        "metadata": metadata_to_store,
                    }
                )

            if not vectors_to_upsert:
                logger.info("No valid vectors to upsert after processing.")
                continue
            
            # Prepare NDJSON payload
            ndjson_payload_lines = [json.dumps(vec) for vec in vectors_to_upsert]
            ndjson_payload_string = "\n".join(ndjson_payload_lines)
            ndjson_payload_bytes = ndjson_payload_string.encode('utf-8')

            # Log details for upload
            upload_url = f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/vectorize/v2/indexes/{self.index_name}/upsert"
            logger.info(f"Attempting to upload {len(vectors_to_upsert)} vectors (as NDJSON) to {upload_url}")
            logger.debug(f"Upload NDJSON payload size: {len(ndjson_payload_bytes)} bytes")

            # Log the exact payload being sent to Cloudflare for upsert (NDJSON format)
            try:
                # Log a snippet of the NDJSON string
                log_limit = 2000
                if len(ndjson_payload_string) > log_limit:
                    logger.critical(f"CRITICAL_VCLIENT_UPLOAD_PAYLOAD_NDJSON (first {log_limit} chars): {ndjson_payload_string[:log_limit]}...")
                else:
                    logger.critical(f"CRITICAL_VCLIENT_UPLOAD_PAYLOAD_NDJSON: {ndjson_payload_string}")
            except Exception as log_e:
                logger.error(f"Error preparing NDJSON payload for logging: {log_e}")

            try:
                response = await self._request("POST", "upsert", content=ndjson_payload_bytes, is_ndjson=True)
                response_data = response.json()
                logger.info(f"Cloudflare Upload API Full Response: Status={response.status_code}, Body={response_data}")
                logger.info(f"Cloudflare Upload API Response Keys: {list(response_data.keys())}") # Log keys
                
                # Attempt to get count from the known structure, or new structure if different
                if "count" in response_data:
                    uploaded_count = response_data.get("count", 0)
                    retrieved_ids = response_data.get("ids")
                elif response_data.get("success") is True and "result" in response_data and "mutationId" in response_data["result"]:
                    # If success is true and we have a mutationId, assume all vectors in the batch were accepted if no errors
                    # However, the API *should* return a count. This is a fallback interpretation.
                    logger.warning("Cloudflare upsert response has 'success:true' and 'mutationId' but no 'count'. Assuming all submitted vectors were accepted for this batch.")
                    uploaded_count = len(vectors_to_upsert) # Assume all in the current batch were processed
                    retrieved_ids = [v["id"] for v in vectors_to_upsert] # Assume all IDs from the batch
                    logger.warning(f"Assumed uploaded_count: {uploaded_count} based on batch size.")
                else:
                    uploaded_count = 0
                    retrieved_ids = None
                    logger.warning("Could not determine uploaded count from Cloudflare response.")

                logger.info(f"Successfully uploaded {uploaded_count} vectors to Cloudflare Vectorize. IDs: {retrieved_ids}")
                successful_uploads += uploaded_count
            except Exception as e:
                logger.error(f"Error uploading to Cloudflare Vectorize: {e}")
                # Propagate the error to be caught by the caller in retriever.py
                raise 

        return successful_uploads

    async def delete_documents_by_site(self, site: str, **kwargs) -> int:
        logger.warning("delete_documents_by_site is not fully implemented for Cloudflare Vectorize yet. It requires querying by metadata and then deleting by ID.")
        # To implement this, you'd typically:
        # 1. Query for documents with metadata.site == site.
        #    POST /accounts/{account_id}/vectorize/indexes/{index_name}/query
        #    Payload: { "filter": { "site": site }, "return_values": false, "top_k": 1000 } (adjust top_k)
        # 2. Collect all IDs from the query results.
        # 3. Delete these IDs.
        #    POST /accounts/{account_id}/vectorize/indexes/{index_name}/delete-by-ids
        #    Payload: { "ids": ["id1", "id2", ...] }
        # This can be complex due to pagination if many documents match.
        # For now, returning 0.
        return 0

    async def search(self, query_text: str, query_vector: Optional[List[float]], num_results: int = 10, site_filter: Optional[Union[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        logger.critical(f"CRITICAL_VCLIENT_LOG: CloudflareVectorizeClient SEARCH called with query_text='{query_text}', num_results={num_results}, site_filter='{site_filter}', query_vector_present={query_vector is not None}")

        effective_top_k = num_results
        current_return_values = True # We want the vectors themselves
        current_return_metadata = "all" # CORRECT FIX: Cloudflare expects "none", "indexed", or "all"

        if current_return_values and current_return_metadata == "all": # Condition for capping topK
            if num_results > 20:
                logger.warning(f"Cloudflare Vectorize: topK reduced from {num_results} to 20 because returnValues is true and returnMetadata is '{current_return_metadata}'.")
                effective_top_k = 20

        # Reordered payload construction
        payload = {}

        if query_vector:
            payload["vector"] = query_vector
        else:
            # This case should ideally not happen for a production search call
            # as the Cloudflare /query API requires the 'vector' field.
            logger.critical("CRITICAL_VCLIENT_LOG: query_vector is None in search method. This is unexpected for a standard search operation.")
            # If this happens, the API will likely reject the request.
            # For testing specific scenarios without a vector, different handling or a dummy vector might be needed.
            # However, the main 'search' flow assumes a vector is provided from embedding.
            # To proceed with the reordering test, we are assuming query_vector will be present for valid searches.
            # If not, the API error will be about the missing 'vector', not 'returnMetadata'.
            pass # Let it proceed; API will reject if vector is missing and required.

        if site_filter:
            payload["filter"] = {"site": site_filter}
            logger.critical(f"CRITICAL_VCLIENT_LOG: Applying site_filter: {site_filter}")
        else:
            logger.critical("CRITICAL_VCLIENT_LOG: No site_filter applied.")

        payload["topK"] = effective_top_k
        payload["returnValues"] = current_return_values
        payload["returnMetadata"] = current_return_metadata
        
        logger.critical(f"CRITICAL_VCLIENT_LOG: Cloudflare Vectorize: Exact search payload being sent: {self._sanitize_payload_for_logging(payload)}")
        
        # Add detailed JSON debugging
        import json
        json_payload = json.dumps(payload)
        logger.critical(f"CRITICAL_VCLIENT_LOG: Raw JSON payload: {json_payload[:200]}...")
        logger.critical(f"CRITICAL_VCLIENT_LOG: JSON payload length: {len(json_payload)}")
        
        response = await self._request("POST", "query", json=payload)
        response_data = response.json()
        logger.critical(f"CRITICAL_VCLIENT_LOG: Cloudflare Search API Full Response: Status={response.status_code}, Body={response_data}")
        
        results = []
        # Fix: Cloudflare response has matches under 'result' key
        result_data = response_data.get("result", {})
        matches = result_data.get("matches", [])
        
        for match in matches:
            # Reconstruct document similar to how it might be stored or expected by downstream code
            doc = {
                "id": match.get("id"),
                "score": match.get("score"),
                "vector": match.get("values"), # if returnValues was true
                **(match.get("metadata", {})) # Spread metadata fields
            }
            results.append(doc)
        
        logger.critical(f"CRITICAL_VCLIENT_LOG: Cloudflare Vectorize search returned {len(results)} results.")
        return results

    async def search_by_url(self, url: str, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve documents by their exact URL using metadata filtering.
        Vectorize GetByIds: GET /accounts/{account_id}/vectorize/indexes/{index_name}/get-by-ids?id=...&id=...
        However, we don't have the vector ID, only the URL. So we must query.
        """
        logger.info(f"Searching by URL (metadata filter): {url}")
        
        # This is a metadata search. We don't have a query vector.
        # Vectorize's query endpoint requires a vector.
        # A common pattern is to fetch by ID, but we don't have the vector ID.
        # We must use the filter capability of the query endpoint.
        # This means we need a *dummy* vector to satisfy the API, or the API must support filter-only queries.
        # Assuming for now the API might support a query with a filter but no vector, or a special endpoint.
        # Let's try to use the general query with a filter and a zero vector if needed.
        
        # According to Cloudflare docs for "Query an index":
        # "vector" (array<number>) - The vector to query for.
        # It seems "vector" is required.
        # We need to know the dimensionality of the vectors in the index to provide a dummy one.
        # This is a problem.
        
        # Alternative: if the 'id' field used during upsert was the URL or a hash of it,
        # and if Vectorize has a "get by id" endpoint that's efficient.
        # Vectorize "Get vectors by ID": GET .../indexes/{index_name}/get-by-ids?id=ID1&id=ID2
        # The `id` we stored was a UUID. The URL is in metadata.
        
        # So, we must use query with a filter. We need a vector.
        # This requires knowing the vector dimension.
        # For now, this method cannot be reliably implemented without knowing the vector dimension
        # or if the API allows querying only by filter.
        
        # Use the confirmed dimension of 768 for the dummy vector
        dimension = 768 # Standardized to 768
        logger.info(f"search_by_url for Vectorize: Using a zero-vector of dimension {dimension} for querying by metadata filter.")
        dummy_vector = [0.0] * dimension
            
        payload = {
            "vector": dummy_vector, # Dummy vector
            "topK": 5, # Expecting few matches for a specific URL
            "returnValues": True,
            "returnMetadata": "all", # CORRECT FIX
            "filter": {"url": url}
        }

        try:
            response_data = (await self._request("POST", "query", json=payload)).json()
            results = []
            # Fix: Cloudflare response has matches under 'result' key
            result_data = response_data.get("result", {})
            matches = result_data.get("matches", [])
            
            for match in matches:
                # Filter further if multiple results (e.g. if URL is not unique, though it should be)
                if match.get("metadata", {}).get("url") == url:
                    doc = {
                        "id": match.get("id"),
                        "score": match.get("score"), # Score might not be relevant here
                        "vector": match.get("values"),
                        **(match.get("metadata", {}))
                    }
                    results.append(doc)
            
            logger.info(f"Cloudflare Vectorize search_by_url for '{url}' returned {len(results)} results.")
            return results if results else None # Return None if no exact match
        except Exception as e:
            logger.error(f"Error in search_by_url for Cloudflare Vectorize: {e}")
            return None

    async def debug_list_vectors_by_site(self, site_filter: str, limit: int = 5) -> Dict[str, Any]:
        """Debug method to list vectors with a site filter by using the query endpoint."""
        if not self.account_id or not self.api_key or not self.index_name:
            logger.error("CRITICAL_VCLIENT_LOG: Account ID, API Key, or Index Name is not configured for debug_list_vectors_by_site.")
            return {"error": "Client not configured"}

        path = "query"  # Changed from vectors/list to query
        
        # Create a dummy vector of the correct dimension (768)
        dummy_vector = [0.0] * 768
        
        payload = {
            "vector": dummy_vector,
            "filter": {"site": site_filter},
            "topK": limit,
            "returnValues": True,      # Typically true to see the vectors
            "returnMetadata": "all"    # CORRECT FIX
        }
        
        logger.info(f"CRITICAL_VCLIENT_LOG: Sending DEBUG LIST VECTORS (as POST query) request to Cloudflare Vectorize. Path: {path}, Payload: {self._sanitize_payload_for_logging(payload)}")
        
        try:
            # Changed from GET with params to POST with json
            response = await self._request("POST", path, json=payload)
            response_data = response.json()
            logger.info(f"CRITICAL_VCLIENT_LOG: Cloudflare DEBUG LIST VECTORS (as POST query) API Full Response: Status={response.status_code}, Body={response_data}")
            return response_data
        except Exception as e:
            logger.error(f"CRITICAL_VCLIENT_LOG: Exception in debug_list_vectors_by_site: {e}")
            return {"error": str(e)}

    async def search_all_sites(self, query_text: str, query_vector: List[float], num_results: int = 50) -> List[Dict[str, Any]]:
        """Search across all configured sites, aggregating results."""
        logger.info(f"CloudflareVectorizeClient: search_all_sites called for query: '{query_text[:50]}...', num_results: {num_results}")

        all_results = []
        processed_doc_ids = set() # To avoid duplicates if a doc is in multiple "sites" somehow or if "all" is searched first

        # Get the list of allowed sites from the global CONFIG object
        # This requires CONFIG to be imported and accessible here.
        # from config.config import CONFIG # Ensure CONFIG is available
        # ^^^ Assuming CONFIG is already imported at the module level or passed appropriately.
        
        # If CONFIG.sites is not directly accessible here, this needs adjustment.
        # For now, let's assume it is available (it is imported at the top of the file).
        configured_sites = CONFIG.get_allowed_sites()
        logger.info(f"CloudflareVectorizeClient: search_all_sites - Configured sites: {configured_sites}")

        if not configured_sites or configured_sites == ["all"]:
            # Fallback: if no sites are specifically configured, or only 'all' is listed, perform a single search without a site filter.
            # This matches the previous behavior for 'all' but might not be what's always intended if 'all' is the only configured site.
            logger.warning("CloudflareVectorizeClient: search_all_sites - No specific sites configured or only 'all'. Performing a single search with no site_filter.")
            # Note: The `search` method's `num_results` parameter is used here.
            # The original `search_all_sites` signature had `top_k`, changed to `num_results` for consistency.
            site_results = await self.search(query_text=query_text, query_vector=query_vector, num_results=num_results, site_filter=None)
            for res in site_results:
                if res.get("id") not in processed_doc_ids:
                    all_results.append(res)
                    processed_doc_ids.add(res.get("id"))
        else:
            # Distribute num_results among sites, ensuring at least 1 per site if possible.
            # This is a simple distribution. More sophisticated logic might be needed.
            num_sites = len(configured_sites)
            results_per_site = max(1, num_results // num_sites) # Ensure at least 1 result per site if num_results allows
            remaining_results = num_results % num_sites

            logger.info(f"CloudflareVectorizeClient: search_all_sites - Iterating over {num_sites} sites. Aiming for ~{results_per_site} results per site.")

            for site_name in configured_sites:
                if site_name.lower() == 'all': # Skip 'all' if it's in the list of specific sites to avoid re-querying without filter
                    logger.info(f"CloudflareVectorizeClient: search_all_sites - Skipping '{site_name}' in specific site iteration.")
                    continue
                
                current_site_num_results = results_per_site
                if remaining_results > 0:
                    current_site_num_results += 1
                    remaining_results -= 1

                logger.info(f"CloudflareVectorizeClient: search_all_sites - Searching site: '{site_name}' for query: '{query_text[:50]}...', num_results: {current_site_num_results}")
                try:
                    # Use current_site_num_results for this site's search
                    site_results = await self.search(query_text=query_text, query_vector=query_vector, num_results=current_site_num_results, site_filter=site_name)
                    logger.info(f"CloudflareVectorizeClient: search_all_sites - Site '{site_name}' returned {len(site_results)} results.")
                    for res in site_results:
                        if res.get("id") not in processed_doc_ids:
                            all_results.append(res)
                            processed_doc_ids.add(res.get("id"))
                except Exception as e:
                    logger.error(f"CloudflareVectorizeClient: search_all_sites - Error searching site '{site_name}': {e}")
        
        # Optional: Re-rank or sort all_results if needed. For now, just concatenate.
        # If results exceed original num_results due to 'at least 1 per site', they might need trimming.
        # However, the per-site limit aims to manage this.
        logger.info(f"CloudflareVectorizeClient: search_all_sites - Total aggregated results: {len(all_results)} for query: '{query_text[:50]}...'")
        return all_results # Return all aggregated results, could be more than original num_results if many sites have matches 