# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
This file contains a simple web server, that should be replaced as appropriate.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

import sys
import os

# Add the parent directory of 'webserver' to sys.path to allow imports from 'core', 'retrieval', etc.
# This assumes WebServer.py is in NLWeb/code/webserver/
# and other key directories like 'core' are in NLWeb/code/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_code_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_code_root)

import asyncio
import json
import time
import traceback
import urllib.parse
import importlib.util
import logging
from core.whoHandler import WhoHandler
from core.mcp_handler import handle_mcp_request
from utils.utils import get_param
from webserver.StreamingWrapper import HandleRequest, SendChunkWrapper
from core.generate_answer import GenerateAnswer
from webserver.static_file_handler import send_static_file
from config.config import CONFIG
from core.baseHandler import NLWebHandler
from utils.logging_config_helper import get_configured_logger, close_logs
from retrieval.retriever import get_vector_db_client
from aiohttp import web

# Force reload logging_config_helper
try:
    spec = importlib.util.find_spec('utils.logging_config_helper')
    if spec and spec.loader:
        logging_config_helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(logging_config_helper)
        get_configured_logger = logging_config_helper.get_configured_logger
        close_logs = logging_config_helper.close_logs
        # Optional: force it into sys.modules to ensure other modules pick up the reloaded one
        sys.modules['utils.logging_config_helper'] = logging_config_helper
        print("Successfully reloaded utils.logging_config_helper")
    else:
        raise ImportError("Could not find spec for utils.logging_config_helper")
except ImportError as e:
    print(f"Failed to import or reload utils.logging_config_helper: {e}")
    # Fallback to standard import if reload fails, or re-raise
    from utils.logging_config_helper import get_configured_logger, close_logs

# Initialize module logger
logger = get_configured_logger("webserver")

# Helper class to adapt aiohttp's StreamResponse to what NLWebHandler expects
class AioHttpStreamWriter:
    def __init__(self, stream_response_obj: web.StreamResponse):
        self.stream_response = stream_response_obj
        self._headers_sent = False

    async def write_stream(self, message_dict: dict):
        if not self._headers_sent:
            # Ensure headers are prepared if not already.
            # This might be redundant if stream_response.prepare() was already called.
            if not self.stream_response.prepared:
                # Set default streaming headers if not set by caller
                # This part might need adjustment based on how StreamResponse is setup before this writer is used
                self.stream_response.headers['Content-Type'] = 'text/event-stream'
                self.stream_response.headers['Cache-Control'] = 'no-cache'
                self.stream_response.headers['Connection'] = 'keep-alive'
                await self.stream_response.prepare(None) # Pass request if available, or None
            self._headers_sent = True
            
        json_data = json.dumps(message_dict)
        try:
            await self.stream_response.write(f"data: {json_data}\\n\\n".encode('utf-8'))
        except ConnectionResetError:
            logger.warning("Connection reset while trying to write to stream.")
            # Potentially raise an error or handle to stop further writes
            raise
        except Exception as e:
            logger.error(f"Error writing to stream: {e}")
            raise

async def handle_client(reader, writer, fulfill_request):
    """Handle a client connection by parsing the HTTP request and passing it to fulfill_request."""
    request_id = f"client_{int(time.time()*1000)}"
    connection_alive = True
    
    try:
        # Read the request line
        request_line = await reader.readline()
        if not request_line:
            connection_alive = False
            return
            
        request_line = request_line.decode('utf-8', errors='replace').rstrip('\r\n')
        words = request_line.split()
        if len(words) < 2:
            # Bad request
            logger.warning(f"[{request_id}] Bad request: {request_line}")
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            connection_alive = False
            return
            
        method, path = words[0], words[1]
        logger.debug(f"[{request_id}] {method} {path}")
        
        # Parse headers
        headers = {}
        while True:
            try:
                header_line = await reader.readline()
                if not header_line or header_line == b'\r\n':
                    break
                    
                hdr = header_line.decode('utf-8').rstrip('\r\n')
                if ":" not in hdr:
                    continue
                name, value = hdr.split(":", 1)
                headers[name.strip().lower()] = value.strip()
            except (ConnectionResetError, BrokenPipeError) as e:
                connection_alive = False
                return
        
        # Parse query parameters
        if '?' in path:
            path, query_string = path.split('?', 1)
            query_params = {}
            try:
                # Parse query parameters into a dictionary of lists
                for key, values in urllib.parse.parse_qs(query_string).items():
                    query_params[key] = values
            except Exception as e:
                query_params = {}
        else:
            query_params = {}
        
        # Read request body if Content-Length is provided
        body = None
        if 'content-length' in headers:
            try:
                content_length = int(headers['content-length'])
                body = await reader.read(content_length)
            except (ValueError, ConnectionResetError, BrokenPipeError) as e:
                connection_alive = False
                return
        
        # Create a streaming response handler
        async def send_response(status_code, response_headers, end_response=False):
            """Send HTTP status and headers to the client."""
            nonlocal connection_alive
            
            if not connection_alive:
                return
                
            try:
                status_line = f"HTTP/1.1 {status_code}\r\n"
                writer.write(status_line.encode('utf-8'))
                
                # Add CORS headers if enabled
                if CONFIG.server.enable_cors and 'Origin' in headers:
                    response_headers['Access-Control-Allow-Origin'] = '*'
                    response_headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
                    response_headers['Access-Control-Allow-Headers'] = 'Content-Type'
                
                # Send headers
                for header_name, header_value in response_headers.items():
                    header_line = f"{header_name}: {header_value}\r\n"
                    writer.write(header_line.encode('utf-8'))
                
                # End headers
                writer.write(b"\r\n")
                await writer.drain()   
                # Signal that we've sent the headers
                send_response.headers_sent = True
                send_response.ended = end_response
            except (ConnectionResetError, BrokenPipeError) as e:
                connection_alive = False
            except Exception as e:
                connection_alive = False
        
        # Create a streaming content sender
        async def send_chunk(chunk, end_response=False):
            """Send a chunk of data to the client."""
            nonlocal connection_alive
            
            if not connection_alive:
                return
                
            if not hasattr(send_response, 'headers_sent') or not send_response.headers_sent:
                logger.warning(f"[{request_id}] Headers must be sent before content")
                return
                
            if hasattr(send_response, 'ended') and send_response.ended:
                logger.warning(f"[{request_id}] Response has already been ended")
                return
                
            try:
                chunk_size = 0
                if chunk:
                    if isinstance(chunk, str):
                        data = chunk.encode('utf-8')
                        chunk_size = len(data)
                        writer.write(data)
                    else:
                        chunk_size = len(chunk)
                        writer.write(chunk)
                    await writer.drain()
                
                send_response.ended = end_response
            except (ConnectionResetError, BrokenPipeError) as e:
                logger.warning(f"[{request_id}] Connection lost while sending chunk: {str(e)}")
                connection_alive = False
            except Exception as e:
                logger.warning(f"[{request_id}] Error sending chunk: {str(e)}")
                connection_alive = False
        
        # Call the user-provided fulfill_request function with streaming capabilities
        if connection_alive:
            try:
                await fulfill_request(
                    method=method,
                    path=urllib.parse.unquote(path),
                    headers=headers,
                    query_params=query_params,
                    body=body,
                    send_response=send_response,
                    send_chunk=send_chunk
                )
            except Exception as e:
                logger.error(f"[{request_id}] Error in fulfill_request: {str(e)}", exc_info=True)
                
                if connection_alive and not (hasattr(send_response, 'headers_sent') and send_response.headers_sent):
                    try:
                        # Send a 500 error if headers haven't been sent yet
                        error_headers = {
                            'Content-Type': 'text/plain',
                            'Connection': 'close'
                        }
                        await send_response(500, error_headers)
                        await send_chunk(f"Internal server error: {str(e)}".encode('utf-8'), end_response=True)
                    except:
                        pass
        
    except Exception as e:
        logger.error(f"[{request_id}] Critical error handling request: {str(e)}", exc_info=True)
    finally:
        # Close the connection in a controlled manner
        try:
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            logger.debug(f"[{request_id}] Connection closed")
        except Exception as e:
            logger.warning(f"[{request_id}] Error closing connection: {str(e)}")

def handle_site_parameter(query_params):
    """
    Handle site parameter with configuration validation.
    
    Args:
        query_params (dict): Query parameters from request
        
    Returns:
        dict: Modified query parameters with valid site parameter(s)
    """
    # Create a copy of query_params to avoid modifying the original
    result_params = query_params.copy()
    logger.debug(f"Query params: {query_params}")
    # Get allowed sites from config
    allowed_sites = CONFIG.get_allowed_sites()
    sites = []
    if "site" in query_params and len(query_params["site"]) > 0:
        sites = query_params["site"]
        logger.debug(f"Sites: {sites}")
    # Check if site parameter exists in query params
    if  len(sites) > 0:
        if isinstance(sites, list):
            # Validate each site
            valid_sites = []
            for site in sites:
                if CONFIG.is_site_allowed(site):
                    valid_sites.append(site)
                else:
                    logger.warning(f"Site '{site}' is not in allowed sites list")
            
            if valid_sites:
                result_params["site"] = valid_sites
            else:
                # No valid sites provided, use default from config
                result_params["site"] = allowed_sites
        else:
            # Single site
            if CONFIG.is_site_allowed(sites):
                result_params["site"] = [sites]
            else:
                logger.warning(f"Site '{sites}' is not in allowed sites list")
                result_params["site"] = allowed_sites
    else:
        # No site parameter provided, use all allowed sites from config
        result_params["site"] = allowed_sites
    
    return result_params

async def start_server(host=None, port=None, fulfill_request=None, use_https=False, 
                 ssl_cert_file=None, ssl_key_file=None):
    """
    Start the HTTP/HTTPS server with the provided request handler.
    """
    import ssl
    
    if fulfill_request is None:
        raise ValueError("fulfill_request function must be provided")
    
    # Use configured values if not provided
    host = host or CONFIG.server.host
    port = port or CONFIG.server.port
    
    ssl_context = None
    if use_https or CONFIG.is_ssl_enabled():
        # Get SSL files from config if not provided
        ssl_cert_file = ssl_cert_file or CONFIG.get_ssl_cert_path()
        ssl_key_file = ssl_key_file or CONFIG.get_ssl_key_path()
        
        if not ssl_cert_file or not ssl_key_file:
            if CONFIG.is_ssl_enabled():
                raise ValueError("SSL is enabled in config but certificate/key files not found in environment")
            else:
                raise ValueError("SSL certificate and key files must be provided for HTTPS")
        
        # Create SSL context - using configuration from working code
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
        ssl_context.set_ciphers('ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256')
        ssl_context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
        
        try:
            ssl_context.load_cert_chain(ssl_cert_file, ssl_key_file)
        except (ssl.SSLError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load SSL certificate: {e}")
    
    # Start server with or without SSL
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, fulfill_request), 
        host, 
        port,
        ssl=ssl_context
    )
    
    addr = server.sockets[0].getsockname()
    protocol = "HTTPS" if (use_https or ssl_context) else "HTTP"
    url_protocol = "https" if (use_https or ssl_context) else "http"
    print(f'Serving {protocol} on {addr[0]} port {addr[1]} ({url_protocol}://{addr[0]}:{addr[1]}/) ...')
    async with server:
        await server.serve_forever()

async def fulfill_request(method, path, headers, query_params, body, send_response, send_chunk):
    '''
    Process an HTTP request and stream the response back.
    
    Args:
        method (str): HTTP method (GET, POST, etc.)
        path (str): URL path
        headers (dict): HTTP headers
        query_params (dict): URL query parameters
        body (bytes or None): Request body
        send_response (callable): Function to send response headers
        send_chunk (callable): Function to send response body chunks
    '''
    try:
        streaming = True
        generate_mode = "none"
        if ("streaming" in query_params):
            strval = get_param(query_params, "streaming", str, "True")
            streaming = strval not in ["False", "false", "0"]

        if ("generate_mode" in query_params):
            generate_mode = get_param(query_params, "generate_mode", str, "none")
           
        if path == "/" or path == "":
            # Serve the home page as /static/index.html
            # First check if the file exists
            try:
                await send_static_file("/static/index.html", send_response, send_chunk)
            except FileNotFoundError:
                # If new_test.html doesn't exist, send a 404 error
                await send_response(404, {'Content-Type': 'text/plain'})
                await send_chunk("Home page not found".encode('utf-8'), end_response=True)
            return
        elif (path.find("html/") != -1) or path.find("static/") != -1 or (path.find("png") != -1):
            await send_static_file(path, send_response, send_chunk)
            return
        elif (path.find("who") != -1):
            retval =  await WhoHandler(query_params, None).runQuery()
            await send_response(200, {'Content-Type': 'application/json'})
            await send_chunk(json.dumps(retval), end_response=True)
            return
        elif (path.find("mcp") != -1):
            # Handle MCP health check
            if path == "/mcp/health" or path == "/mcp/healthz":
                await send_response(200, {'Content-Type': 'application/json'})
                await send_chunk(json.dumps({"status": "ok"}), end_response=True)
                return

            # Check if streaming should be used from query parameters
            use_streaming = False
            if ("streaming" in query_params):
                strval = get_param(query_params, "streaming", str, "False")
                use_streaming = strval not in ["False", "false", "0"]
                
            # Handle MCP requests with streaming parameter
            logger.info(f"Routing to MCP handler (streaming={use_streaming})")
            await handle_mcp_request(query_params, body, send_response, send_chunk, streaming=use_streaming)
            return
        elif (path.find("ask") != -1):
            # Handle site parameter validation for ask endpoint
            validated_query_params = handle_site_parameter(query_params)

            # If POST request with JSON body, parse and merge into query_params
            if method == "POST" and headers.get('content-type', '').lower() == 'application/json' and body:
                try:
                    json_body = json.loads(body.decode('utf-8'))
                    logger.debug(f"Parsed JSON body for /ask: {json_body}")
                    if 'query' in json_body:
                        validated_query_params['query'] = [json_body['query']] # parse_qs expects list
                    if 'site' in json_body:
                         # Re-validate site if provided in JSON body, as handle_site_parameter might have defaulted
                        site_from_body = json_body.get('site')
                        if isinstance(site_from_body, str): # ensure it's a string before making a list
                            validated_query_params['site'] = [site_from_body]
                        elif isinstance(site_from_body, list):
                             validated_query_params['site'] = site_from_body
                        else:
                            logger.warning(f"Site in JSON body is not a string or list: {site_from_body}")
                        
                        # Re-run handle_site_parameter with the potentially updated site from JSON body
                        # This ensures consistent validation logic
                        temp_params_for_site_check = {'site': validated_query_params['site']}
                        if 'query' in validated_query_params: # carry over query if it exists
                            temp_params_for_site_check['query'] = validated_query_params['query']
                        
                        validated_query_params = handle_site_parameter(temp_params_for_site_check)
                        # Ensure 'query' is preserved if it was in the original validated_query_params
                        # and not overwritten if only 'site' was in json_body
                        if 'query' not in validated_query_params and 'query' in query_params:
                             validated_query_params['query'] = query_params['query']


                    logger.info(f"Updated query_params from JSON body: query='{validated_query_params.get('query')}', site='{validated_query_params.get('site')}'")
                except json.JSONDecodeError:
                    logger.warning("Failed to decode JSON body for /ask POST request")
                except Exception as e:
                    logger.error(f"Error processing JSON body for /ask: {e}")
            
            if (not streaming):
                if (generate_mode == "generate"):
                    retval = await GenerateAnswer(validated_query_params, None).runQuery()
                else:
                    retval = await NLWebHandler(validated_query_params, None).runQuery()
                await send_response(200, {'Content-Type': 'application/json'})
                await send_chunk(json.dumps(retval), end_response=True)
                return
            else:   
                # Set proper headers for server-sent events (SSE)
                response_headers = {
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'  # Disable proxy buffering
                }
                
                # Send SSE headers
                await send_response(200, response_headers)
                
                # Send initial keep-alive comment to establish connection
                await send_chunk(": keep-alive\n\n", end_response=False)
                
                # Create wrapper for chunk sending
                send_chunk_wrapper = SendChunkWrapper(send_chunk)
                
                # Handle the request with validated query parameters
                hr = HandleRequest(method, path, headers, validated_query_params, 
                                   body, send_response, send_chunk_wrapper, generate_mode)
                await hr.do_GET()
        else:
            # Default handler for unknown paths
            logger.warning(f"No handler found for path: {path}")
            await send_response(404, {'Content-Type': 'text/plain'})
            await send_chunk(f"No handler found for path: {path}".encode('utf-8'), end_response=True)
    except Exception as e:
        logger.error(f"Error in fulfill_request: {e}\n{traceback.format_exc()}")
        raise

def close_logs():
    """Close the log file when the application exits."""
    if hasattr(logging, 'log_file'):
        logging.log_file.close()

# Azure Web App specific: Check for the PORT environment variable
def get_port():
    """Get the port to listen on, using config or environment."""
    if 'PORT' in os.environ:
        port = int(os.environ['PORT'])
        print(f"Using PORT from environment variable: {port}")
        return port
    elif 'WEBSITE_SITE_NAME' in os.environ:
        # Running in Azure App Service
        print("Running in Azure App Service, using default port 8000")
        return 8000  # Azure will redirect requests to this port
    else:
        # Use configured port directly from CONFIG object
        print(f"Using configured port {CONFIG.port}")
        return CONFIG.port

if __name__ == "__main__":
    try:
        port = get_port()
        
        # Check if running in Azure App Service or locally
        is_azure = 'WEBSITE_SITE_NAME' in os.environ
        is_local = not is_azure
        
        if is_azure:
            print(f"Running in Azure App Service: {os.environ.get('WEBSITE_SITE_NAME')}")
            print(f"Home directory: {os.environ.get('HOME')}")
            # List all environment variables
            print("Environment variables:")
            for key, value in os.environ.items():
                print(f"  {key}: {value}")
            use_https = CONFIG.is_ssl_enabled() 
        else:
            print(f"Running on localhost (port {port})")
            use_https = False
        
        if use_https:
            print("Starting HTTPS server")
            # If using command line, use default cert files
            if len(sys.argv) > 1 and sys.argv[1] == "https":
                ssl_cert_file = 'fullchain.pem'
                ssl_key_file = 'privkey.pem'
            else:
                # Use config values
                ssl_cert_file = CONFIG.get_ssl_cert_path()
                ssl_key_file = CONFIG.get_ssl_key_path()
                
            asyncio.run(start_server(
                fulfill_request=fulfill_request,
                use_https=True,
                ssl_cert_file=ssl_cert_file,
                ssl_key_file=ssl_key_file,
                port=port
            ))
        else:
            # Use the detected port
            print(f"Starting HTTP server on port {port}")
            asyncio.run(start_server(port=port, fulfill_request=fulfill_request))
    finally:
        # Make sure to close the log file when the application exits
        close_logs()

async def handle_debug_list_vectors(request: web.Request) -> web.Response:
    logger.warning("/debug_list_vectors is temporarily disabled pending handler refactor.")
    return web.json_response({"message": "/debug_list_vectors temporarily disabled"}, status=503)

def setup_routes(app):
    app.router.add_get("/", handle_index)
    app.router.add_post("/ask_with_context", handle_ask_with_context)
    app.router.add_get("/get_history", handle_get_history)
    app.router.add_post("/debug_list_vectors", handle_debug_list_vectors)

class WebServer:
    def __init__(self, handler_class=NLWebHandler, host=None, port=None, enable_cors=True, loop=None, use_https=False, ssl_cert_file=None, ssl_key_file=None):
        self.app = web.Application()
        self.host = host or CONFIG.server.host
        self.port = port or CONFIG.server.port
        self.handler_class = handler_class
        self.enable_cors = enable_cors
        self._loop = loop or asyncio.get_event_loop()
        self.use_https = use_https
        self.ssl_cert_file = ssl_cert_file
        self.ssl_key_file = ssl_key_file
        self.setup_routes() # Call setup_routes from __init__

    async def handle_index(self, request: web.Request) -> web.Response:
        return await send_static_file(request, 'index.html')

    async def handle_static_file_request(self, request: web.Request) -> web.Response:
        filepath = request.match_info.get('filepath', 'index.html')
        return await send_static_file(request, filepath)

    async def handle_mcp_request_wrapper(self, request: web.Request) -> web.Response:
        return await handle_mcp_request(request, None) # Passing None as handler for now

    async def handle_ask(self, request: web.Request) -> web.Response:
        client_id = request.headers.get("X-Client-ID", f"client_{int(time.time() * 1000)}")
        session_id = request.headers.get("X-Session-ID", f"session_{int(time.time() * 1000)}")
        source_ip = request.remote
        logger.info(f"[{client_id}] Received /ask request from {source_ip}")

        # Determine data_for_nlwhandler first to check streaming flag
        data_for_nlwhandler = {}
        if request.method == "POST":
            if request.content_type == "application/json":
                try:
                    data_for_nlwhandler = await request.json()
                except json.JSONDecodeError:
                    logger.warning(f"[{client_id}] POST request to /ask had non-JSON body or decode error.")
                    return web.json_response({"error": "Invalid JSON body"}, status=400)
            else:
                logger.warning(f"[{client_id}] POST request to /ask had unexpected Content-Type: {request.content_type}")
                return web.json_response({"error": "Unsupported Media Type"}, status=415)
        elif request.method == "GET":
            for key, value in request.query.items():
                if key == "query":
                    data_for_nlwhandler[key] = urllib.parse.unquote(value)
                else:
                    data_for_nlwhandler[key] = value
            logger.info(f"[{client_id}] GET request to /ask, params from query string: {data_for_nlwhandler}")

        # Check streaming parameter from the parsed data
        streaming_param = data_for_nlwhandler.get("streaming", "True") # Default to True if not present
        is_streaming_request = streaming_param.lower() not in ["false", "0"]

        if is_streaming_request:
            logger.info(f"[{client_id}] Handling /ask as a STREAMING request.")
            stream_response = web.StreamResponse(
                status=200,
                reason='OK',
                headers={
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, OPTIONS, POST',
                    'Access-Control-Allow-Headers': 'Content-Type, X-Client-ID, X-Session-ID'
                }
            )
            await stream_response.prepare(request)
            http_stream_writer = AioHttpStreamWriter(stream_response)
            nlweb_h_instance = self.handler_class(data_for_nlwhandler, http_stream_writer)
        else:
            logger.info(f"[{client_id}] Handling /ask as a NON-STREAMING request.")
            # For non-streaming, http_handler in NLWebHandler will be None
            nlweb_h_instance = self.handler_class(data_for_nlwhandler, None) 

        try:
            # For non-streaming, runQuery will collect results in nlweb_h_instance.return_value
            # For streaming, runQuery will use http_stream_writer to send data
            final_payload = await nlweb_h_instance.runQuery()
            
            if not is_streaming_request:
                logger.info(f"[{client_id}] Non-streaming request completed. Returning JSON payload.")
                return web.json_response(final_payload)
            else:
                # For streaming, the data has already been sent. 
                # The stream_response object needs to be returned to keep the connection open
                # until NLWebHandler finishes or client disconnects.
                # Ensure NLWebHandler properly closes the stream or indicates completion if necessary.
                # (Currently, AioHttpStreamWriter's write might raise if connection is closed, which is handled by NLWebHandler)
                logger.info(f"[{client_id}] Streaming request processing finished by NLWebHandler.")
                # If NLWebHandler.runQuery() returns something in streaming mode, it's ignored here.
                # The AioHttpStreamWriter has handled the response.
                pass # Fall through to return stream_response

        except ConnectionResetError:
            logger.warning(f"Connection reset by client {client_id} during /ask handling.")
            if is_streaming_request and stream_response.prepared and not stream_response.task.done():
                # Attempt to close the stream response gracefully if possible
                # await stream_response.write_eof() # This might be needed if not auto-handled
                pass
            # For non-streaming, an error before json_response would be caught by the broader except block
            raise # Re-raise to be handled by aiohttp or outer error handlers
        except Exception as e:
            logger.error(f"Error processing /ask request for client {client_id}: {e}", exc_info=True)
            if is_streaming_request:
                try:
                    if stream_response.prepared and not stream_response.task.done():
                        error_message = {"error": "An internal server error occurred.", "message_type": "error"}
                        json_error = json.dumps(error_message)
                        await stream_response.write(f"data: {json_error}\\n\\n".encode('utf-8'))
                        # await stream_response.write_eof() # Consider closing stream after error
                except Exception as ex_send:
                    logger.error(f"Failed to send error message to client {client_id} over stream: {ex_send}")
            else: # Non-streaming error
                return web.json_response({"error": "An internal server error occurred.", "details": str(e)}, status=500)
        finally:
            logger.debug(f"[{client_id}] /ask handler finished.")
        
        if is_streaming_request:
            return stream_response
        else:
            # This path should ideally not be reached if non-streaming succeeded (returned json_response)
            # or failed (returned error json_response or raised).
            # However, as a fallback, if final_payload was somehow not returned.
            logger.error(f"[{client_id}] Non-streaming request reached end of handler unexpectedly. Returning empty 500.")
            return web.json_response({"error":"Internal server error - unexpected flow"}, status=500)

    async def handle_ask_with_context(self, request: web.Request) -> web.Response:
        # Similar structure to handle_ask, but for /ask_with_context
        # ... (implementation depends on specific needs for this endpoint)
        return web.json_response({"message": "/ask_with_context not fully implemented"})

    async def handle_get_history(self, request: web.Request) -> web.Response:
        # ... (implementation for /get_history)
        return web.json_response({"message": "/get_history not fully implemented"})

    async def handle_debug_list_vectors(self, request: web.Request) -> web.Response:
        client_id = request.headers.get("X-Client-ID", f"client_{int(time.time() * 1000)}")
        logger.info(f"[{client_id}] Received /debug_list_vectors request")
        try:
            data = await request.json()
            site = data.get("site")
            limit = data.get("limit", 10) # Default limit to 10

            if not site:
                return web.json_response({"error": "Missing 'site' parameter"}, status=400)

            # Get the VectorDB client (CloudflareVectorizeClient)
            # The retriever's get_client will give us the configured client
            # We pass dummy query_params as NLWebHandler typically does,
            # ensuring preferred_retrieval_endpoint is read from config.
            client_config_params = {'preferred_retrieval_endpoint': CONFIG.preferred_retrieval_endpoint}
            vector_db_client = get_vector_db_client(query_params=client_config_params)
            
            if not hasattr(vector_db_client, 'debug_list_vectors_by_site'):
                logger.error("Vector DB client does not have 'debug_list_vectors_by_site' method.")
                return web.json_response({"error": "Debug method not available on client"}, status=500)

            logger.info(f"Calling debug_list_vectors_by_site for site: {site}, limit: {limit}")
            results = await vector_db_client.debug_list_vectors_by_site(site_filter=site, limit=int(limit))
            return web.json_response(results)
        except json.JSONDecodeError:
            logger.error(f"[{client_id}] Invalid JSON in /debug_list_vectors request")
            return web.json_response({"error": "Invalid JSON payload"}, status=400)
        except Exception as e:
            logger.error(f"Error processing /debug_list_vectors request for client {client_id}: {e}", exc_info=True)
            return web.json_response({"error": "An internal server error occurred."}, status=500)

    def setup_routes(self):
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_post("/ask", self.handle_ask)
        self.app.router.add_get("/ask", self.handle_ask)
        self.app.router.add_post("/ask_with_context", self.handle_ask_with_context)
        self.app.router.add_get("/get_history", self.handle_get_history)
        self.app.router.add_post("/debug_list_vectors", self.handle_debug_list_vectors)
        self.app.router.add_route("*", "/mcp/{tail:.+}", self.handle_mcp_request_wrapper) # Catch all for MCP
        # Serve static files (ensure this is last or use a more specific path)
        self.app.router.add_get("/{filepath:.*}", self.handle_static_file_request)

    async def start_server_async(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"Server started on http://{self.host}:{self.port}")
        # Keep server running until interrupted
        while True:
            await asyncio.sleep(3600) # Keep alive

    def run(self):
        try:
            self._loop.run_until_complete(self.start_server_async())
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        finally:
            # Perform any cleanup
            self._loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(self._loop))) # Ensure all tasks complete
            self._loop.close()
            close_logs() # Ensure log files are closed
            logger.info("Server stopped.")