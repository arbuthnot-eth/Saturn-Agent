# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Wrapper around the various embedding providers.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

from typing import Optional, List
import asyncio
import threading

from config.config import CONFIG
from utils.logging_config_helper import get_configured_logger, LogLevel

logger = get_configured_logger(__name__) # Use __name__ for module-level logger

# Add locks for thread-safe provider access
_provider_locks = {
    "openai": threading.Lock(),
    "gemini": threading.Lock(),
    "azure_openai": threading.Lock(),
    "snowflake": threading.Lock(),
    "openrouter": threading.Lock(),
    "cloudflare": threading.Lock()
}

# Define a class that will be returned by get_embedding_model
class EmbeddingModel:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or CONFIG.preferred_embedding_provider
        provider_config = CONFIG.get_embedding_provider(self.provider)
        self.model = model or (provider_config.model if provider_config else None)
        
        if not self.provider or self.provider not in CONFIG.embedding_providers:
            raise ValueError(f"Unknown or unspecified embedding provider for EmbeddingModel: '{self.provider}'")
        if not self.model:
            raise ValueError(f"No embedding model could be determined for provider '{self.provider}' in EmbeddingModel")

    async def get_embedding(self, text: str, timeout: int = 30) -> List[float]:
        """Instance method to call the module-level get_embedding function."""
        return await get_embedding(text, provider=self.provider, model=self.model, timeout=timeout)

    async def batch_get_embeddings(self, texts: List[str], timeout: int = 60) -> List[List[float]]:
        """Instance method to call the module-level batch_get_embeddings function."""
        return await batch_get_embeddings(texts, provider=self.provider, model=self.model, timeout=timeout)


# Function that retriever.py is looking for
def get_embedding_model(provider: Optional[str] = None, model: Optional[str] = None) -> EmbeddingModel:
    """
    Factory function to get an instance of the EmbeddingModel.
    This instance will be configured to use a specific provider and model,
    defaulting to the preferred ones in the application configuration.
    """
    return EmbeddingModel(provider=provider, model=model)


async def get_embedding(
    text: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 30
) -> List[float]:
    """
    Get embedding for the provided text using the specified provider and model.
    
    Args:
        text: The text to embed
        provider: Optional provider name, defaults to preferred_embedding_provider
        model: Optional model name, defaults to the provider's configured model
        timeout: Maximum time to wait for embedding response in seconds
        
    Returns:
        List of floats representing the embedding vector
    """
    # Use preferred_embedding_provider from CONFIG if provider is not specified
    provider = provider or CONFIG.preferred_embedding_provider
    logger.debug(f"Getting embedding with provider: {provider}")
    logger.debug(f"Text length: {len(text)} chars")
    
    if not provider or provider not in CONFIG.embedding_providers:
        error_msg = f"Unknown or unspecified embedding provider: '{provider}'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    provider_config = CONFIG.get_embedding_provider(provider)
    if not provider_config:
        error_msg = f"Missing configuration for embedding provider '{provider}'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    model_id = model or provider_config.model
    if not model_id:
        error_msg = f"No embedding model specified for provider '{provider}'"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug(f"Using embedding model: {model_id}")

    try:
        if provider == "cloudflare":
            logger.debug("Getting Cloudflare embeddings")
            # Assuming cloudflare_embedding.py exists in the same directory
            from .cloudflare_embedding import get_cloudflare_embedding 
            result = await asyncio.wait_for(
                get_cloudflare_embedding(text, model=model_id),
                timeout=timeout
            )
            logger.debug(f"Cloudflare embeddings received, dimension: {len(result) if result else 'N/A'}")
            return result
        
        # Stubs for other providers - assuming their helper modules might be missing or need review
        # For a complete restoration, these would need their respective "<provider>_embedding.py" files.
        elif provider in ["openai", "gemini", "azure_openai", "snowflake", "openrouter"]:
            logger.warning(f"Provider '{provider}' called, but its implementation might be missing or incomplete.")
            # Example:
            # if provider == "openai":
            #     from .openai_embedding import get_openai_embeddings # Assuming this file exists
            #     return await asyncio.wait_for(get_openai_embeddings(text, model=model_id), timeout=timeout)
            raise NotImplementedError(f"Embedding implementation for provider '{provider}' is not fully restored or available.")

        else:
            error_msg = f"No embedding implementation for provider '{provider}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    except asyncio.TimeoutError:
        logger.error(f"Embedding request timed out after {timeout}s with provider {provider}")
        raise
    except ImportError as e:
        logger.error(f"Failed to import helper module for provider '{provider}': {e}. Make sure '{provider}_embedding.py' exists.")
        raise NotImplementedError(f"Helper module for provider '{provider}' could not be imported.")
    except Exception as e:
        logger.exception(f"Error during embedding generation with provider {provider}")
        # Consider using log_with_context if available and appropriate
        # logger.log_with_context(
        #     LogLevel.ERROR,
        #     "Embedding generation failed",
        #     {
        #         "provider": provider,
        #         "model": model_id,
        #         "text_length": len(text),
        #         "error_type": type(e).__name__,
        #         "error_message": str(e)
        #     }
        # )
        raise

async def batch_get_embeddings(
    texts: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 60 # Longer timeout for batches
) -> List[List[float]]:
    """
    Get embeddings for a batch of texts.
    
    Args:
        texts: List of texts to embed
        provider: Optional provider name, defaults to preferred_embedding_provider
        model: Optional model name, defaults to the provider's configured model
        timeout: Maximum time to wait for batch embedding response in seconds
        
    Returns:
        List of embedding vectors, each a list of floats
    """
    provider = provider or CONFIG.preferred_embedding_provider
    logger.debug(f"Getting batch embeddings with provider: {provider}")
    logger.debug(f"Batch size: {len(texts)} texts")
    
    provider_config = CONFIG.get_embedding_provider(provider)
    if not provider_config:
        error_msg = f"Missing configuration for embedding provider '{provider}'"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    model_id = model or provider_config.model
    if not model_id:
        error_msg = f"No embedding model specified for provider '{provider}'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Using batch embedding model: {model_id}")
    
    try:
        if provider == "cloudflare":
            logger.debug("Getting Cloudflare batch embeddings")
            from .cloudflare_embedding import get_cloudflare_batch_embeddings # Assuming this file exists
            result = await asyncio.wait_for(
                get_cloudflare_batch_embeddings(texts, model=model_id),
                timeout=timeout
            )
            logger.debug(f"Cloudflare batch embeddings received, count: {len(result) if result else 'N/A'}")
            return result

        # Stubs/Fallback for other providers
        elif provider in ["openai", "gemini", "azure_openai", "snowflake", "openrouter"]:
            logger.warning(f"Batch implementation for provider '{provider}' might be missing or fallback to sequential.")
            # Example for a provider that has a batch implementation:
            # if provider == "openai":
            #     from .openai_embedding import get_openai_batch_embeddings # Assuming this file exists
            #     return await asyncio.wait_for(get_openai_batch_embeddings(texts, model=model_id), timeout=timeout)
            
            # Fallback to sequential processing if no batch method for other providers
            logger.debug(f"No specific batch implementation for {provider}, processing sequentially.")
            results = []
            # Individual timeout for each item in sequential processing should be shorter
            individual_timeout = max(10, timeout // len(texts)) if texts else 10 
            for text in texts:
                embedding = await get_embedding(text, provider, model, timeout=individual_timeout)
                results.append(embedding)
            return results

        else:
            error_msg = f"No batch embedding implementation for provider '{provider}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
    except asyncio.TimeoutError:
        logger.error(f"Batch embedding request timed out after {timeout}s with provider {provider}")
        raise
    except ImportError as e:
        logger.error(f"Failed to import helper module for batch provider '{provider}': {e}. Make sure '{provider}_embedding.py' exists.")
        raise NotImplementedError(f"Helper module for batch provider '{provider}' could not be imported.")
    except Exception as e:
        logger.exception(f"Error during batch embedding generation with provider {provider}")
        # Consider using log_with_context
        # logger.log_with_context(
        #     LogLevel.ERROR,
        #     "Batch embedding generation failed",
        #     {
        #         "provider": provider,
        #         "model": model_id,
        #         "batch_size": len(texts),
        #         "error_type": type(e).__name__,
        #         "error_message": str(e)
        #     }
        # )
        raise 