# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
This file is the entry point for the NLWeb Sample App.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

import asyncio
import os
import sys

# Ensure the 'code' directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, '..') 
sys.path.insert(0, code_dir)

from config.config import CONFIG
# from webserver.WebServer import fulfill_request, start_server, get_port, close_logs # Old import
from webserver.WebServer import WebServer, get_port # Import WebServer class and get_port
from utils.logging_config_helper import close_logs, get_configured_logger # Ensure close_logs is imported here too

logger = get_configured_logger("app-file")

if __name__ == "__main__":
    # Initialize configuration (idempotent)
    # CONFIG.reload() # Removed this line as AppConfig does not have a reload() method.
                      # Config is loaded during AppConfig.__init__.
    logger.info(f"CONFIG Keys: {list(CONFIG.__dict__.keys())}") # Log available attributes for info
    
    # Determine if running in Azure based on environment variables
    is_azure = 'WEBSITE_SITE_NAME' in os.environ
    is_local = not is_azure
    
    # Get port from environment or config
    port = get_port()
    
    # Determine HTTPS usage
    use_https = False
    ssl_cert_file = None
    ssl_key_file = None
    
    if is_azure:
        logger.info(f"Running in Azure App Service: {os.environ.get('WEBSITE_SITE_NAME')}")
        logger.info(f"Home directory: {os.environ.get('HOME')}")
        # In Azure, SSL is typically handled by the App Service itself
        use_https = CONFIG.is_ssl_enabled() # Respect config if explicitly set for Azure
    else:
        logger.info(f"Running locally (port {port})")
        # For local development, SSL can be enabled via config
        use_https = CONFIG.is_ssl_enabled()
        if use_https:
            ssl_cert_file = CONFIG.get_ssl_cert_path()
            ssl_key_file = CONFIG.get_ssl_key_path()
            if not ssl_cert_file or not ssl_key_file:
                logger.warning("SSL enabled in config, but cert/key paths not found. Falling back to HTTP.")
                use_https = False

    server_instance = WebServer(
        host=CONFIG.server.host, # Use configured host
        port=port,
        use_https=use_https,
        ssl_cert_file=ssl_cert_file,
        ssl_key_file=ssl_key_file
    )

    try:
        logger.info(f"Starting server {'HTTPS' if use_https else 'HTTP'} on port {port}")
        server_instance.run() # Use the run method of the WebServer instance
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Application shutting down...")
        # The WebServer instance's run method should handle its own cleanup including close_logs
        # but calling it here again ensures it if run() exits early or due to an unhandled exception within run().
        close_logs()
        logger.info("Application shutdown complete.")