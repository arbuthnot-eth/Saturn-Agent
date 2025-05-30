# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

\"\"\"
OpenRouter Embedding Provider.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
\"\"\"

import asyncio
from typing import List, Optional

from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIStatusError

from config.config import CONFIG
from utils.logging_config_helper import get_configured_logger, LogLevel

logger = get_configured_logger(__name__)

# Global variable for the OpenRouter client
_openrouter_client: Optional[AsyncOpenAI] = None

def get_openrouter_client() -> AsyncOpenAI:
    \"\"\"Get the OpenRouter client, initializing it if necessary.\"\"\"
    global _openrouter_client
    if _openrouter_client is None:
        provider_config = CONFIG.get_embedding_provider("openrouter")
        if not provider_config:
            raise ValueError("OpenRouter embedding provider configuration not found.")

        api_key = provider_config.api_key
        base_url = provider_config.endpoint # In config_embedding.yaml, we named it api_endpoint_env

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment or config.")
        if not base_url:
            raise ValueError("OPENROUTER_API_BASE_URL is not set in environment or config.")

        logger.info(f"Initializing OpenRouter client with base URL: {base_url}")
        _openrouter_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    return _openrouter_client

async def get_openrouter_embedding(
    text: str,
    model: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 2  # seconds
) -> List[float]:
    \"\"\"
    Get embedding for a single text using OpenRouter.
    
    Args:
        text: The text to embed.
        model: The embedding model to use (e.g., "thenlper/gte-large").
        max_retries: Maximum number of retries for API calls.
        retry_delay: Delay between retries in seconds.
        
    Returns:
        List of floats representing the embedding vector.
    \"\"\"
    if not model:
        provider_config = CONFIG.get_embedding_provider("openrouter")
        model = provider_config.model if provider_config else "thenlper/gte-large" # Default if not in config

    client = get_openrouter_client()
    text = text.replace("\\n", " ") # OpenAI recommendation

    for attempt in range(max_retries):
        try:
            logger.debug(f"Requesting OpenRouter embedding for model: {model}, attempt: {attempt + 1}")
            response = await client.embeddings.create(
                input=[text],
                model=model
            )
            embedding = response.data[0].embedding
            logger.debug(f"Successfully received embedding from OpenRouter, dimension: {len(embedding)}")
            return embedding
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            logger.warning(f"OpenRouter API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached for OpenRouter embedding.")
                raise
            await asyncio.sleep(retry_delay * (attempt + 1)) # Exponential backoff
        except Exception as e:
            logger.exception("An unexpected error occurred during OpenRouter embedding generation.")
            raise
    return [] # Should not be reached if max_retries > 0

async def get_openrouter_batch_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 2  # seconds
) -> List[List[float]]:
    \"\"\"
    Get embeddings for a batch of texts using OpenRouter.
    
    Args:
        texts: List of texts to embed.
        model: The embedding model to use.
        max_retries: Maximum number of retries for API calls.
        retry_delay: Delay between retries in seconds.

    Returns:
        List of embedding vectors.
    \"\"\"
    if not model:
        provider_config = CONFIG.get_embedding_provider("openrouter")
        model = provider_config.model if provider_config else "thenlper/gte-large"

    client = get_openrouter_client()
    # OpenAI recommendation: replace newlines
    processed_texts = [text.replace("\\n", " ") for text in texts]

    for attempt in range(max_retries):
        try:
            logger.debug(f"Requesting OpenRouter batch embeddings for model: {model}, batch size: {len(texts)}, attempt: {attempt + 1}")
            response = await client.embeddings.create(
                input=processed_texts,
                model=model
            )
            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Successfully received batch embeddings from OpenRouter, count: {len(embeddings)}")
            return embeddings
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            logger.warning(f"OpenRouter API error (batch attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached for OpenRouter batch embedding.")
                raise
            await asyncio.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            logger.exception("An unexpected error occurred during OpenRouter batch embedding generation.")
            raise
    return [[] for _ in texts] # Should not be reached 