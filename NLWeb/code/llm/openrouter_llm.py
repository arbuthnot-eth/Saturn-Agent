# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
OpenRouter LLM provider.
Uses the OpenAI SDK as OpenRouter is OpenAI-compatible.
"""

import os
import json
import re
import asyncio
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI # Using AsyncOpenAI
from config.config import CONFIG, LLMProviderConfig
import threading

from llm.llm_provider import LLMProvider
from utils.logging_config_helper import get_configured_logger, LogLevel

logger = get_configured_logger(__name__)

class ConfigurationError(RuntimeError):
    """Raised when configuration is missing or invalid."""
    pass

class OpenRouterProvider(LLMProvider):
    """Implementation of LLMProvider for OpenRouter API."""
    
    _client_lock = threading.Lock()
    _client = None # Stores AsyncOpenAI client

    @classmethod
    def _get_provider_config(cls) -> Optional[LLMProviderConfig]:
        """Helper to get the OpenRouter specific configuration."""
        return CONFIG.get_llm_provider("openrouter")

    @classmethod
    def get_api_key(cls) -> str:
        """Retrieve the OpenRouter API key from config or raise an error."""
        provider_config = cls._get_provider_config()
        if not provider_config or not provider_config.api_key:
            logger.error("OpenRouter API key not found in configuration.")
            raise ConfigurationError("Missing OpenRouter API key in configuration.")
        return provider_config.api_key

    @classmethod
    def get_base_url(cls) -> str:
        """Retrieve the OpenRouter API base URL from config or raise an error."""
        provider_config = cls._get_provider_config()
        # The 'endpoint' field in LLMProviderConfig is used for base_url for OpenRouter
        if not provider_config or not provider_config.endpoint: 
            logger.error("OpenRouter API base URL (endpoint) not found in configuration.")
            raise ConfigurationError("Missing OpenRouter API base URL (endpoint) in configuration.")
        return provider_config.endpoint

    @classmethod
    def get_client(cls) -> AsyncOpenAI:
        """Configure and return an asynchronous OpenAI client for OpenRouter."""
        with cls._client_lock:  # Thread-safe client initialization
            if cls._client is None:
                logger.info("Initializing OpenRouter client (via OpenAI SDK)")
                api_key = cls.get_api_key()
                base_url = cls.get_base_url()
                
                if not api_key: # Should be caught by get_api_key, but double check
                    raise ConfigurationError("OpenRouter API key is not set.")
                if not base_url: # Should be caught by get_base_url, but double check
                     raise ConfigurationError("OpenRouter base URL is not set.")

                try:
                    cls._client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=base_url
                    )
                    logger.info("OpenRouter client initialized successfully.")
                except Exception as e:
                    logger.exception("Failed to initialize OpenRouter client")
                    raise
        return cls._client

    @classmethod
    def _build_messages(cls, prompt: str, schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Construct the system and user message sequence enforcing a JSON schema.
        """
        # OpenRouter generally expects a JSON response if the last message asks for it
        # or if function calling / tools are used. Here, we rely on the prompt
        # and schema to imply JSON is desired.
        system_message_content = f"You are a helpful assistant. Provide a valid JSON response that strictly adheres to the following JSON schema: {json.dumps(schema)}. Only output the JSON object itself, with no additional text, explanations, or markdown formatting like ```json ... ```."

        return [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": prompt}
        ]

    @classmethod
    def clean_response(cls, content: str) -> Dict[str, Any]:
        """
        Strip markdown fences (if any) and extract the JSON object.
        OpenRouter might or might not wrap in markdown, so be flexible.
        """
        logger.debug(f"Raw OpenRouter response content: {content[:500]}...")
        
        # Attempt to remove markdown JSON fences if present
        cleaned_content = content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[len("```json"):].strip()
        if cleaned_content.startswith("```"): # Broader check for just ```
             cleaned_content = cleaned_content[len("```"):].strip()
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-len("```")].strip()
        
        # Sometimes the model might still include text like "Here is the JSON:"
        # Try to find the first '{' and last '}'
        json_start = cleaned_content.find('{')
        json_end = cleaned_content.rfind('}')

        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = cleaned_content[json_start : json_end+1]
            try:
                parsed_json = json.loads(json_str)
                logger.debug("Successfully parsed JSON from OpenRouter response after cleaning.")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from cleaned content: {json_str[:500]}... Error: {e}")
                # Fall through to try parsing the whole cleaned_content
        
        # If specific extraction failed or wasn't needed, try parsing the whole cleaned string
        try:
            parsed_json = json.loads(cleaned_content)
            logger.debug("Successfully parsed JSON from OpenRouter response (direct parse).")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from OpenRouter response: {cleaned_content[:500]}... Error: {e}")
            raise ValueError(f"No valid JSON object found in OpenRouter response. Content: {cleaned_content[:200]}")


    async def get_completion(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None, # Model is passed by llm.py from config
        temperature: float = 0.1, # Defaulting to lower temp for more deterministic JSON
        max_tokens: int = 2048,
        timeout: float = 30.0, # Default timeout from LLMProvider
        **kwargs 
    ) -> Dict[str, Any]:
        """
        Send an async chat completion request to OpenRouter and return parsed JSON output.
        """
        if not model:
            # This should ideally be pre-filled by the caller (llm.py)
            # based on 'low' or 'high' tier from config
            provider_config = self._get_provider_config()
            if provider_config and provider_config.models:
                model = provider_config.models.low # Default to low tier if not specified
            if not model:
                logger.error("OpenRouter model not specified and not found in config.")
                raise ConfigurationError("OpenRouter model not specified.")
        
        logger.info(f"OpenRouter request: model='{model}', temp={temperature}, max_tokens={max_tokens}")
        
        client = self.get_client()
        messages = self._build_messages(prompt, schema)

        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # OpenRouter generally infers JSON mode if the prompt asks for it
            # and the schema implies it. Some models might support response_format.
            # "response_format": {"type": "json_object"}, # Add if consistently needed and supported
        }

        # Add site and user headers for OpenRouter analytics/moderation if available
        # These are custom headers OpenRouter recommends.
        # We would need to plumb site/user info down to here if we want to use them.
        # headers = {
        #     "HTTP-Referer": CONFIG.openrouter_site_url or "", # Your app's site URL
        #     "X-Title": CONFIG.openrouter_app_title or "",    # Your app's name
        # }
        # For now, we'll make the call without these optional headers.

        try:
            logger.debug(f"OpenRouter request params: {request_params}")
            response = await asyncio.wait_for(
                client.chat.completions.create(**request_params),
                timeout=timeout
            )
            
            if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                logger.error("OpenRouter response is empty or malformed.")
                raise ValueError("OpenRouter response is empty or malformed.")

            content = response.choices[0].message.content
            logger.debug(f"OpenRouter raw response usage: {response.usage}")
            return self.clean_response(content)
            
        except asyncio.TimeoutError:
            logger.error(f"OpenRouter completion request timed out after {timeout}s for model {model}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter completion failed for model {model}: {type(e).__name__}: {str(e)}")
            # Log more details if it's an API error from OpenRouter
            if hasattr(e, 'status_code'):
                logger.error(f"OpenRouter API Error Status: {e.status_code}, Response: {getattr(e, 'response', '')}")
            raise

# Create a singleton instance
provider = OpenRouterProvider() 