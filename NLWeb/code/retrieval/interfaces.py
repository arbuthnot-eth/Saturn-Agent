# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
This module defines abstract base classes for retrieval components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class VectorDBClientInterface(ABC):
    """
    Abstract base class defining the interface for vector database clients.
    All vector database implementations should implement these methods.
    """
    
    @abstractmethod
    async def delete_documents_by_site(self, site: str, **kwargs) -> int:
        """
        Delete all documents matching the specified site.
        
        Args:
            site: Site identifier
            **kwargs: Additional parameters
            
        Returns:
            Number of documents deleted
        """
        pass
    
    @abstractmethod
    async def upload_documents(self, documents: List[Dict[str, Any]], **kwargs) -> int:
        """
        Upload documents to the database.
        
        Args:
            documents: List of document objects
            **kwargs: Additional parameters
            
        Returns:
            Number of documents uploaded
        """
        pass
    
    # The search method in the original retriever.py takes 'query: str'
    # but client implementations like vectorize_client expect 'query_vector: List[float]'.
    # This interface should reflect what the actual clients will implement directly.
    # The main Retriever class can handle the text-to-vector conversion before calling the client.
    @abstractmethod
    async def search(self, query_vector: List[float], site: Union[str, List[str]], 
                     num_results: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query vector and site.
        
        Args:
            query_vector: The vector representation of the query.
            site: Site identifier or list of sites to filter by.
            num_results: Maximum number of results to return.
            **kwargs: Additional parameters.
            
        Returns:
            List of search result documents (as dicts).
        """
        pass
    
    @abstractmethod
    async def search_by_url(self, url: str, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve a document by its exact URL.
        
        Args:
            url: URL to search for
            **kwargs: Additional parameters
            
        Returns:
            List of document data (as dicts) or None if not found.
        """
        pass
    
    # search_all_sites also implies a query_vector input for clients
    @abstractmethod
    async def search_all_sites(self, query_vector: List[float], num_results: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search across all sites using a query vector.
        
        Args:
            query_vector: The vector representation of the query.
            num_results: Maximum number of results to return.
            **kwargs: Additional parameters.
            
        Returns:
            List of search result documents (as dicts).
        """
        pass 