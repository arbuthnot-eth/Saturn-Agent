# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
This file contains the code for the static file handler.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

import os
from config.config import CONFIG
from aiohttp import web

# Determine the application root directory based on environment
def get_app_root():
    """Get the application root directory, handling both local and Azure environments."""
    # Check if running in Azure App Service
    if 'WEBSITE_SITE_NAME' in os.environ:
        # Azure App Service - use D:\home\site\wwwroot or the HOME environment variable
        azure_home = os.environ.get('HOME', '/home/site/wwwroot')
        return azure_home
    else:
        # Use configured static directory
        return CONFIG.static_directory

# Get the app root directory
APP_ROOT = get_app_root()

async def send_static_file(request: 'web.Request', path: str) -> 'web.Response':
    # Map file extensions to MIME types
    mime_types = {
        '.html': 'text/html',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.css': 'text/css',
        '.js': 'application/javascript'
    }

    # Get file extension and corresponding MIME type
    file_ext = os.path.splitext(path)[1].lower()
    content_type = mime_types.get(file_ext, 'application/octet-stream')
    
    try:
        # Remove leading slash and sanitize path
        safe_path = os.path.normpath(path.lstrip('/'))

        # Try multiple possible root locations
        possible_roots = [
            APP_ROOT,
            os.path.join(APP_ROOT, 'site', 'wwwroot'),
            '/home/site/wwwroot',
            os.environ.get('HOME', ''),
        ]
        
        # Remove empty paths
        possible_roots = [root for root in possible_roots if root]
        
        file_found = False
        full_path = None
       
        for root in possible_roots:
            try_path = os.path.join(root, safe_path)
            if os.path.isfile(try_path):
                full_path = try_path
                file_found = True
                break
        
        if not file_found:
            # Special case: check if removing 'html/' prefix works
            prefixes = ['html/', 'static/']
            for prefix in prefixes:
                if safe_path.startswith(prefix):
                    stripped_path = safe_path[len(prefix):]  # Remove prefix
                    for root in possible_roots:
                        try_path = os.path.join(root, stripped_path)
                        print(try_path)
                        if os.path.isfile(try_path):
                            full_path = try_path
                            file_found = True
                            break
        
        if not file_found:
            # Special case: check if there's no html/static directory
            # and the files are directly in the root
            parts = safe_path.split('/')
            if len(parts) > 1:
                filename = parts[-1]
                for root in possible_roots:
                    try_path = os.path.join(root, filename)
                    if os.path.isfile(try_path):
                        full_path = try_path
                        file_found = True
                        break
        
        if file_found:
            # Try to open and read the file
            # with open(full_path, 'rb') as f:  # Not needed for FileResponse
            #     content = f.read()
                
            response_headers = {} # Headers can be added to FileResponse
            
            # Add cache headers if caching is enabled
            if CONFIG.server.static.enable_cache:
                response_headers['Cache-Control'] = f'public, max-age={CONFIG.server.static.cache_max_age}'
            else:
                response_headers['Cache-Control'] = 'no-cache'
            
            # Use aiohttp.web.FileResponse
            return web.FileResponse(full_path, headers=response_headers, content_type=content_type) # Return FileResponse
        
        # If we reached here, the file was not found
        error_msg = f"File not found (1): {path} {full_path}{possible_roots}{safe_path}"
       
        return web.Response(text=error_msg, status=404, content_type='text/plain') # Return web.Response
        
    except FileNotFoundError:
        # Send 404 if file not found
        error_msg = f"File not found (2): {path}"
       
        return web.Response(text=error_msg, status=404, content_type='text/plain') # Return web.Response
        
    except Exception as e:
        # Send 500 for other errors
        error_msg = f"Internal server error: {str(e)}"
       
        return web.Response(text=error_msg, status=500, content_type='text/plain') # Return web.Response