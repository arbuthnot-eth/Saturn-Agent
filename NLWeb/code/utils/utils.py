recipe_sites = ['seriouseats', 'hebbarskitchen', 'latam_recipes',
                'woksoflife', 'cheftariq',  'spruce', 'nytimes']

all_sites = recipe_sites + ["imdb", "npr podcasts", "neurips", "backcountry", "tripadvisor"]

def siteToItemType(site):
    # For any single site's deployment, this can stay in code. But for the
    # multi-tenant, it should move to the database
    namespace = "http://nlweb.ai/base"
    et = "Item"
    if site == "imdb":
        et = "Movie"
    elif site in recipe_sites:
        et = "Recipe"
    elif site == "npr podcasts":
        et = "Thing"
    elif site == "neurips":
        et = "Paper"
    elif site == "backcountry":
        et = "Outdoor Gear"
    elif site == "tripadvisor":
        et = "Restaurant"
    elif site == "zillow":
        et = "RealEstate"
    else:
        et = "Items"
    return f"{{{namespace}}}{et}"
    

def itemTypeToSite(item_type):
    # this is used to route queries that this site cannot answer,
    # but some other site can answer. Needs to be generalized.
    sites = []
    for site in all_sites:
        if siteToItemType(site) == item_type:
            sites.append(site)
    return sites
    

def visibleUrlLink(url):
    from urllib.parse import urlparse

def visibleUrl(url):
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.netloc.replace('www.', '')

def get_param(query_params, param_name, param_type=str, default_value=None):
    value = query_params.get(param_name)

    if value is None:
        return default_value

    # If the value from query_params is already the desired type (e.g., a direct string for str)
    if isinstance(value, param_type) and param_type != list: # handle list separately due to its specific parsing
        return value

    # Existing logic for handling list-wrapped values or string-encoded lists
    if isinstance(value, list) and len(value) == 1:
        item = value[0]
    elif isinstance(value, str):
        item = value # If it's a string that needs to be parsed as a list or other type
    else:
        # If value is a list with multiple items, or some other type not handled for direct conversion
        if param_type == list and isinstance(value, list):
             return value # Return the list as is if list is expected and value is already a list
        return default_value

    # Process 'item' based on param_type
    if param_type == str:
        return str(item) if item is not None else ""
    elif param_type == int:
        try:
            return int(item) if item is not None else 0
        except ValueError:
            return default_value if default_value is not None else 0
    elif param_type == float:
        try:
            return float(item) if item is not None else 0.0
        except ValueError:
            return default_value if default_value is not None else 0.0
    elif param_type == bool:
        if isinstance(item, bool):
            return item
        return str(item).lower() == "true" if item is not None else False
    elif param_type == list:
        if isinstance(item, list): # If item was already a list (e.g. from multi-item list in query_params)
            return item
        if isinstance(item, str):
             # Handles string like "[item1, item2]" or "item1,item2"
            cleaned_item = item.strip()
            if cleaned_item.startswith('[') and cleaned_item.endswith(']'):
                cleaned_item = cleaned_item[1:-1]
            return [i.strip() for i in cleaned_item.split(',') if i.strip()] if cleaned_item else []
        return default_value if default_value is not None else []
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")

def log(message):
    print(message)