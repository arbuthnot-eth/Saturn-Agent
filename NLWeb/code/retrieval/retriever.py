# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Unified vector database interface with support for Azure AI Search, Milvus, and Qdrant.
This module provides abstract base classes and concrete implementations for database operations.
"""

import time
import asyncio
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple, Type

from config.config import CONFIG
from utils.utils import get_param
from utils.logging_config_helper import get_configured_logger
from utils.logger import LogLevel
from embedding.embedding import get_embedding_model
from retrieval.interfaces import VectorDBClientInterface

# Import client classes
from retrieval.azure_search_client import AzureSearchClient
from retrieval.milvus_client import MilvusVectorClient
from retrieval.qdrant import QdrantVectorClient
from retrieval.snowflake_client import SnowflakeCortexSearchClient
from .vectorize_client import CloudflareVectorizeClient # Simplified relative import

logger = get_configured_logger("retriever")

# Client cache for reusing instances
_client_cache = {}
_client_cache_lock = asyncio.Lock()


class VectorDBClient:
    """
    Unified client for vector database operations. This class routes operations to the appropriate
    client implementation based on the database type specified in configuration.
    """
    
    def __init__(self, endpoint_name: Optional[str] = None, query_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the database client.
        
        Args:
            endpoint_name: Optional name of the endpoint to use
            query_params: Optional query parameters for overriding endpoint
        """
        self.query_params = query_params or {}
        
        # Get default endpoint from config
        self.endpoint_name = endpoint_name or CONFIG.preferred_retrieval_endpoint
        
        # In development mode, allow query param override
        if CONFIG.is_development_mode() and self.query_params:
            self.endpoint_name = get_param(self.query_params, "db", str, self.endpoint_name)
            logger.debug(f"Development mode: endpoint overridden to {self.endpoint_name}")
        
        # Validate endpoint exists in config
        if self.endpoint_name not in CONFIG.retrieval_endpoints:
            error_msg = f"Invalid endpoint: {self.endpoint_name}. Must be one of: {list(CONFIG.retrieval_endpoints.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get endpoint config and extract db_type
        self.endpoint_config = CONFIG.retrieval_endpoints[self.endpoint_name]
        self.db_type = self.endpoint_config.db_type
        self._retrieval_lock = asyncio.Lock()
        
        logger.info(f"VectorDBClient initialized - endpoint: {self.endpoint_name}, db_type: {self.db_type}")
    
    async def get_client(self, endpoint_name: Optional[str] = None) -> VectorDBClientInterface:
        """
        Get or initialize the appropriate vector database client based on database type.
        Uses a cache to avoid creating duplicate client instances.
        
        Args:
            endpoint_name: Optional specific endpoint name to use
        
        Returns:
            Appropriate vector database client
        """
        # Use cache key combining db_type and endpoint
        cache_key = f"{self.db_type}_{endpoint_name or self.endpoint_name}"
        
        # Check if client already exists in cache
        async with _client_cache_lock:
            if self.db_type == "vectorize":
                logger.info("Bypassing client cache for db_type 'vectorize' to ensure fresh instance.")
            elif cache_key in _client_cache:
                return _client_cache[cache_key]
            
            # Create the appropriate client
            logger.debug(f"Creating new client for {self.db_type} with endpoint {endpoint_name or self.endpoint_name}")
            
            if self.db_type == "azure_ai_search":
                client = AzureSearchClient(endpoint_name or self.endpoint_name)
            elif self.db_type == "milvus":
                client = MilvusVectorClient(endpoint_name or self.endpoint_name)
            elif self.db_type == "qdrant":
                client = QdrantVectorClient(endpoint_name or self.endpoint_name)
            elif self.db_type == "snowflake_cortex_search":
                client = SnowflakeCortexSearchClient(endpoint_name or self.endpoint_name)
            elif self.db_type == "vectorize":
                logger.critical("CRITICAL_LOG: Attempting to instantiate CloudflareVectorizeClient from retriever.py get_client()")
                
                # Get the specific endpoint configuration for the current endpoint name
                endpoint_config_data = CONFIG.retrieval_endpoints.get(self.endpoint_name)

                if not endpoint_config_data or endpoint_config_data.db_type != "vectorize":
                    error_msg = f"Cloudflare Vectorize endpoint '{self.endpoint_name}' not configured correctly or not found in retrieval_endpoints."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                account_id = endpoint_config_data.account_id
                api_key = endpoint_config_data.api_key
                index_name = endpoint_config_data.index_name
                
                logger.critical(f"CRITICAL_RETRIEVER_LOG: Initializing CloudflareVectorizeClient with retrieved params:")
                logger.critical(f"CRITICAL_RETRIEVER_LOG:   Endpoint Name: {self.endpoint_name}")
                logger.critical(f"CRITICAL_RETRIEVER_LOG:   Account ID: {account_id}")
                logger.critical(f"CRITICAL_RETRIEVER_LOG:   API Key: {'******' if api_key else 'None'}")
                logger.critical(f"CRITICAL_RETRIEVER_LOG:   Index Name: {index_name}")

                if not account_id or not api_key or not index_name:
                    error_msg = f"Missing one or more required parameters (account_id, api_key, index_name) for Cloudflare Vectorize endpoint '{self.endpoint_name}' in configuration."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                client = CloudflareVectorizeClient(
                    endpoint_name=endpoint_name or self.endpoint_name
                )
            else:
                error_msg = f"Unsupported database type: {self.db_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Store in cache and return
            if self.db_type != "vectorize":
                _client_cache[cache_key] = client
            return client
    
    async def delete_documents_by_site(self, site: str, **kwargs) -> int:
        """
        Delete all documents matching the specified site.
        
        Args:
            site: Site identifier
            **kwargs: Additional parameters
            
        Returns:
            Number of documents deleted
        """
        async with self._retrieval_lock:
            logger.info(f"Deleting documents for site: {site}")
            
            try:
                client: VectorDBClientInterface = await self.get_client()
                count = await client.delete_documents_by_site(site, **kwargs)
                logger.info(f"Successfully deleted {count} documents for site: {site}")
                return count
            except Exception as e:
                logger.exception(f"Error deleting documents for site {site}: {e}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "Document deletion failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "site": site,
                        "db_type": self.db_type,
                        "endpoint": self.endpoint_name
                    }
                )
                raise
    
    async def upload_documents(self, documents: List[Dict[str, Any]], site_name: str, **kwargs) -> int:
        """
        Upload documents to the database.
        
        Args:
            documents: List of document objects
            site_name: The name of the site these documents belong to.
            **kwargs: Additional parameters
            
        Returns:
            Number of documents uploaded
        """
        async with self._retrieval_lock:
            logger.info(f"Uploading {len(documents)} documents for site: {site_name}")
            
            try:
                client: VectorDBClientInterface = await self.get_client()
                count = await client.upload_documents(documents, site_name=site_name, **kwargs)
                logger.info(f"Successfully uploaded {count} documents for site: {site_name}")
                return count
            except Exception as e:
                logger.exception(f"Error uploading documents: {e}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "Document upload failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "document_count": len(documents),
                        "db_type": self.db_type,
                        "endpoint": self.endpoint_name
                    }
                )
                raise
    
    async def search(self, query: str, site: Union[str, List[str]], 
                    num_results: int = 50, endpoint_name: Optional[str] = None, **kwargs) -> List[List[str]]:
        """
        Search for documents matching the query and site.
        Converts text query to vector before calling the specific client.
        
        Args:
            query: Search query string
            site: Site identifier or list of sites
            num_results: Maximum number of results to return
            endpoint_name: Optional specific endpoint name to use
            **kwargs: Additional parameters
            
        Returns:
            List of search results (documents as dicts).
        """
        async with self._retrieval_lock:
            logger.info(f"Searching for query: '{query}' in site(s): {site}")
            actual_endpoint_name = endpoint_name or self.endpoint_name
            
            try:
                embedding_model = get_embedding_model()
                query_vector = await embedding_model.get_embedding(query)
                if query_vector is None:
                    logger.error("Failed to generate embedding for the query.")
                    return []

                client: VectorDBClientInterface = await self.get_client(actual_endpoint_name)
                results = await client.search(query_text=query, query_vector=query_vector, site_filter=site, num_results=num_results, **kwargs)
                logger.info(f"Search returned {len(results)} results for query: '{query}' in site(s): {site}")
                return results
            except Exception as e:
                logger.exception(f"Error in search: {e}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "Search failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "site": site,
                        "db_type": self.db_type,
                        "endpoint": self.endpoint_name
                    }
                )
                raise
    
    async def search_by_url(self, url: str, endpoint_name: Optional[str] = None, **kwargs) -> Optional[List[str]]:
        """
        Retrieve a document by its exact URL.
        
        Args:
            url: URL to search for
            endpoint_name: Optional endpoint name override
            **kwargs: Additional parameters
            
        Returns:
            Document data or None if not found
        """
        async with self._retrieval_lock:
            logger.info(f"Searching by URL: {url}")
            actual_endpoint_name = endpoint_name or self.endpoint_name

            try:
                client: VectorDBClientInterface = await self.get_client(actual_endpoint_name)
                document_data = await client.search_by_url(url, **kwargs)
                
                if document_data:
                    logger.info(f"Successfully retrieved document by URL: {url}")
                else:
                    logger.info(f"No document found for URL: {url}")
                return document_data
            except Exception as e:
                logger.exception(f"Error retrieving item with URL: {url}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "Item retrieval failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "url": url,
                        "db_type": self.db_type,
                        "endpoint": self.endpoint_name
                    }
                )
                raise
    
    async def search_all_sites(self, query: str, num_results: int = 50, 
                             endpoint_name: Optional[str] = None, **kwargs) -> List[List[str]]:
        """
        Search across all sites.
        Converts text query to vector before calling the specific client.

        Args:
            query: Search query string
            num_results: Maximum number of results to return
            endpoint_name: Optional specific endpoint to use
            **kwargs: Additional parameters
            
        Returns:
            List of search results (documents as dicts).
        """
        async with self._retrieval_lock:
            logger.info(f"Searching all sites for query: '{query}'")
            actual_endpoint_name = endpoint_name or self.endpoint_name
            
            try:
                embedding_model = get_embedding_model()
                query_vector = await embedding_model.get_embedding(query)
                if query_vector is None:
                    logger.error("Failed to generate embedding for the query for search_all_sites.")
                    return []

                client: VectorDBClientInterface = await self.get_client(actual_endpoint_name)
                results = await client.search_all_sites(query_text=query, query_vector=query_vector, num_results=num_results, **kwargs)
                logger.info(f"Search all sites returned {len(results)} results for query: '{query}'")
                return results
            except Exception as e:
                logger.exception(f"Error in search_all_sites: {e}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "All-sites search failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "db_type": self.db_type,
                        "endpoint": self.endpoint_name
                    }
                )
                raise

    async def debug_list_vectors_by_site(self, site_filter: str, limit: int = 5, endpoint_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Debug method to list vectors with a site filter, routed through the specific client."""
        async with self._retrieval_lock:
            logger.info(f"Debug listing vectors for site: {site_filter} with limit: {limit}")
            actual_endpoint_name = endpoint_name or self.endpoint_name

            try:
                client: VectorDBClientInterface = await self.get_client(actual_endpoint_name)
                if hasattr(client, 'debug_list_vectors_by_site'):
                    results = await client.debug_list_vectors_by_site(site_filter=site_filter, limit=limit, **kwargs)
                    logger.info(f"Debug list vectors returned for site: {site_filter}")
                    return results
                else:
                    logger.error(f"The configured client for {self.db_type} does not support debug_list_vectors_by_site.")
                    return {"error": f"Debug method not available on client type {self.db_type}"}
            except Exception as e:
                logger.exception(f"Error in debug_list_vectors_by_site: {e}")
                logger.log_with_context(
                    LogLevel.ERROR,
                    "Debug list vectors failed",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "site_filter": site_filter,
                        "db_type": self.db_type,
                        "endpoint": self.endpoint_name
                    }
                )
                # Return an error structure similar to what the Cloudflare client might return on error
                return {"error": str(e), "matches": [], "success": False}


# Factory function to make it easier to get a client with the right type
def get_vector_db_client(endpoint_name: Optional[str] = None, 
                        query_params: Optional[Dict[str, Any]] = None) -> VectorDBClient:
    """
    Factory function to create a vector database client with the appropriate configuration.
    
    Args:
        endpoint_name: Optional name of the endpoint to use
        query_params: Optional query parameters for overriding endpoint
        
    Returns:
        Configured VectorDBClient instance
    """
    return VectorDBClient(endpoint_name=endpoint_name, query_params=query_params)