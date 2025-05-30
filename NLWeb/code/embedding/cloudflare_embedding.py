# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Cloudflare AI Gateway Embedding Provider.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

import asyncio
import httpx # Using httpx as it's already a dependency
from typing import List, Optional

from config.config import CONFIG
from utils.logging_config_helper import get_configured_logger, LogLevel

logger = get_configured_logger(__name__)

# Cloudflare AI Gateway base URL structure
CLOUDFLARE_GATEWAY_BASE = "https://gateway.ai.cloudflare.com/v1"

async def _make_cloudflare_request(
    account_id: Optional[str], # Now optional if base_url is used
    gateway_id: Optional[str], # Now optional if base_url is used
    model_name: Optional[str], # Now optional if base_url is used
    api_token: str,
    text_input: List[str],
    base_url_override: Optional[str] = None # New param for explicit base_url
) -> List[List[float]]:
    """Makes a request to Cloudflare for embeddings, using AI Gateway or a direct base_url."""
    
    if base_url_override:
        url = base_url_override
        # Model name for logging is part of the base_url, or could be passed separately if needed for clarity
        effective_model_name = base_url_override.split("/")[-1] if base_url_override else model_name
    elif account_id and gateway_id and model_name: # Original AI Gateway path
        url = f"{CLOUDFLARE_GATEWAY_BASE}/{account_id}/{gateway_id}/workers-ai/{model_name}"
        effective_model_name = model_name
    else:
        raise ValueError("Insufficient parameters for Cloudflare request: Need either base_url_override or (account_id, gateway_id, and model_name)")

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    # The Cloudflare Workers AI embedding API expects a simple { "text": [...] } structure
    json_payload = { "text": text_input }

    async with httpx.AsyncClient() as client:
        try:
            logger.debug(f"Requesting Cloudflare embedding. URL: {url}, Model: {effective_model_name}, Texts: {len(text_input)}")
            response = await client.post(url, headers=headers, json=json_payload, timeout=30.0)
            response.raise_for_status() # Will raise an HTTPStatusError for 4xx/5xx responses
            
            response_json = response.json()
            
            if response_json.get("success") and response_json.get("result") and response_json["result"].get("data"):
                embeddings = response_json["result"]["data"]
                logger.debug(f"Successfully received embeddings from Cloudflare, count: {len(embeddings)}")
                # Log the first 5 elements of the first embedding for comparison
                if embeddings and embeddings[0]:
                    logger.critical(f"CRITICAL_EMBEDDING_LOG: First 5 elements of first embedding: {embeddings[0][:5]}")
                return embeddings
            else:
                error_detail = response_json.get("errors") or response_json.get("messages") or response.text
                logger.error(f"Cloudflare embedding API call successful but response format unexpected or indicated failure: {error_detail}")
                raise ValueError(f"Cloudflare API response error: {error_detail}")

        except httpx.HTTPStatusError as e:
            logger.error(f"Cloudflare API HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Cloudflare API request error: {e}")
            raise
        except Exception as e:
            logger.exception("An unexpected error occurred during Cloudflare embedding request.")
            raise

async def get_cloudflare_embedding(
    text: str,
    model: Optional[str] = None # Model name like '@cf/baai/bge-large-en-v1.5'
) -> List[float]:
    """
    Get embedding for a single text using Cloudflare AI Gateway.
    """
    provider_config = CONFIG.get_embedding_provider("cloudflare")
    if not provider_config:
        raise ValueError("Cloudflare embedding provider configuration not found.")

    api_token = provider_config.api_key
    account_id = provider_config.endpoint # Storing account_id in endpoint field
    gateway_id = getattr(provider_config, 'gateway_id', None) # Get gateway_id if present
    model_to_use = model or provider_config.model # Model name from config or call
    base_url_from_config = getattr(provider_config, 'base_url', None)

    # Prioritize base_url from config if available
    if base_url_from_config:
        if not api_token:
            raise ValueError("Missing Cloudflare configuration: CLOUDFLARE_API_TOKEN is required even with base_url")
        # When using base_url, account_id, gateway_id, and model_name for URL construction are not directly needed by _make_cloudflare_request
        # as they are part of the base_url itself. Model name is still needed for logging.
        effective_model_name_for_log = model_to_use or base_url_from_config.split("/")[-1]
    elif not all([api_token, account_id, gateway_id, model_to_use]): # Fallback to AI Gateway validation
        missing = []
        if not api_token: missing.append("CLOUDFLARE_API_TOKEN")
        if not account_id: missing.append("CLOUDFLARE_ACCOUNT_ID (from endpoint in config)")
        if not gateway_id: missing.append("CLOUDFLARE_GATEWAY_ID (from gateway_id_env in config)")
        if not model_to_use: missing.append("Model name")
        raise ValueError(f"Missing Cloudflare configuration: {', '.join(missing)}")

    embeddings_list = await _make_cloudflare_request(
        account_id=account_id if not base_url_from_config else None, # Pass None if base_url is used
        gateway_id=gateway_id if not base_url_from_config else None, # Pass None if base_url is used
        model_name=model_to_use if not base_url_from_config else None, # Pass None if base_url is used for URL construction
        api_token=api_token,
        text_input=[text],
        base_url_override=base_url_from_config
    )
    if embeddings_list and len(embeddings_list) > 0:
        return embeddings_list[0]
    return []

async def get_cloudflare_batch_embeddings(
    texts: List[str],
    model: Optional[str] = None # Model name like '@cf/baai/bge-large-en-v1.5'
) -> List[List[float]]:
    """
    Get embeddings for a batch of texts using Cloudflare AI Gateway.
    """
    provider_config = CONFIG.get_embedding_provider("cloudflare")
    if not provider_config:
        raise ValueError("Cloudflare embedding provider configuration not found.")

    api_token = provider_config.api_key
    account_id = provider_config.endpoint
    gateway_id = getattr(provider_config, 'gateway_id', None)
    model_to_use = model or provider_config.model
    base_url_from_config = getattr(provider_config, 'base_url', None)

    if base_url_from_config:
        if not api_token:
            raise ValueError("Missing Cloudflare configuration: CLOUDFLARE_API_TOKEN is required even with base_url")
    elif not all([api_token, account_id, gateway_id, model_to_use]):
        missing = []
        if not api_token: missing.append("CLOUDFLARE_API_TOKEN")
        if not account_id: missing.append("CLOUDFLARE_ACCOUNT_ID (from endpoint in config)")
        if not gateway_id: missing.append("CLOUDFLARE_GATEWAY_ID (from gateway_id_env in config)")
        if not model_to_use: missing.append("Model name")
        raise ValueError(f"Missing Cloudflare configuration: {', '.join(missing)}")

    return await _make_cloudflare_request(
        account_id=account_id if not base_url_from_config else None,
        gateway_id=gateway_id if not base_url_from_config else None,
        model_name=model_to_use if not base_url_from_config else None,
        api_token=api_token,
        text_input=texts,
        base_url_override=base_url_from_config
    ) 