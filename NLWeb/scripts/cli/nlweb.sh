#!/usr/bin/env bash

#!/usr/bin/env bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_DIR=$SCRIPT_DIR/../../

# includes
source "$SCRIPT_DIR/../lib/banner.sh"
source "$SCRIPT_DIR/../lib/shell_logger.sh"
source "$SCRIPT_DIR/../lib/shell_inputs.sh"
source "$SCRIPT_DIR/../lib/llm_providers.sh"
source "$SCRIPT_DIR/../lib/retrieval_endpoints.sh"

declare DEBUG=false
declare me=$(basename "$0")
declare DATALOAD_RSS_URL="" # Added to store arg
declare DATALOAD_SITE_NAME="" # Added to store arg

function init(){
  configure_llm_provider
  configure_retrieval_endpoint
  }

function dataload(){
  local rss_url_arg=$1
  local site_name_arg=$2
  local rss_url
  local site_name

  if [[ -n "$rss_url_arg" ]]; then
    rss_url="$rss_url_arg"
    _info "Using provided RSS URL: $rss_url"
  else
    _prompt_input "Please enter an RSS url" rss_url
  fi

  if [[ -n "$site_name_arg" ]]; then
    site_name="$site_name_arg"
    _info "Using provided site name: $site_name"
  else
    _prompt_input "Please enter a site name" site_name
  fi

  if [[ -z "$rss_url" ]] || [[ -z "$site_name" ]]; then
    _error "RSS URL and site name are required."
    return 1
  fi

  _debug "Loading data from $rss_url site name: $site_name"
  source "$REPO_DIR/code/.venv/bin/activate"
  pushd "$REPO_DIR/code" > /dev/null || exit 1
  export PYTHONPATH="$REPO_DIR/code"
  # Set logging profile to development for verbose logs during data load
  export NLWEB_LOGGING_PROFILE=development
  _info "NLWEB_LOGGING_PROFILE set to: $NLWEB_LOGGING_PROFILE for data load"

  python3 tools/db_load.py "$rss_url" "$site_name"
  local exit_code=$?
  popd > /dev/null || exit 1
  deactivate
  return $exit_code
}

function init_python(){
  python3 -m venv venv
  source venv/bin/activate

  pushd "$REPO_DIR/code" > /dev/null || exit 1
    pip install -r requirements.txt
  popd || exit 1

  _info "Run 'source venv/bin/activate' to activate the virtual environment"
}

function run(){
  init
  check
  dataload # This would need args if called directly, or rely on global DATALOAD_ vars if called via process_command
  app
}

function check(){
    pushd "$REPO_DIR/code" > /dev/null || exit 1
        python3 azure-connectivity.py 
    popd || exit 1
}

function app(){
    echo "Running NLWeb Server..."
    cd "$REPO_DIR/code/"

    # Activate virtual environment
    if [ -d ".venv" ]; then # Check if .venv directory exists
        echo "Activating Python virtual environment..."
        source .venv/bin/activate
    else
        echo "Warning: .venv directory not found. Skipping venv activation."
        echo "Please ensure dependencies are installed globally or in your active environment."
    fi

    # Set PYTHONPATH to ensure local modules are found
    export PYTHONPATH="$REPO_DIR/code:$PYTHONPATH"
    echo "PYTHONPATH set to: $PYTHONPATH"

    # Set logging profile to development for verbose logs
    export NLWEB_LOGGING_PROFILE=development
    echo "NLWEB_LOGGING_PROFILE set to: $NLWEB_LOGGING_PROFILE"

    # Run the application using python3
    python3 app-file.py

    # Deactivate virtual environment upon exit (if activated)
    if [ -d ".venv" ] && [ -n "$VIRTUAL_ENV" ]; then
        echo "Deactivating Python virtual environment..."
        deactivate
    fi
    cd "$CURRENT_DIR"  # Make sure CURRENT_DIR is defined, usually at the start of script.
}

# utility functions
function parse_args() {
  while (("$#")); do
    case "${1}" in
    data-load)
        shift 1 # Shift past 'data-load'
        export command="dataload"
        # Store the next two arguments if they exist for dataload
        if [[ -n "$1" ]] && [[ "$1" != -* ]]; then # Check if $1 is not an option
            DATALOAD_RSS_URL="$1"
            shift 1
            if [[ -n "$1" ]] && [[ "$1" != -* ]]; then # Check if $1 is not an option
                DATALOAD_SITE_NAME="$1"
                shift 1
            else
                # If only one arg for data-load, or next is an option, site_name will be empty
                # and dataload function will prompt if configured to do so.
                # Or we can error here if both are strictly required as args.
                : # Placeholder, decided to let dataload handle missing arg by prompting
            fi
        else
            : # Placeholder, no args for data-load, dataload will prompt
        fi
        ;;       
    check)
        shift 1
        export command="check"
        ;;   
    run)
        shift 1
        export command="run"
        ;;           
    app)
        shift 1
        export command="app"
        ;;         
    init)
        shift 1
        export command="init"
        ;;
    init-python)
        shift 1
        export command="initpython"
        ;;        
    -h | --help)
        shift 1
        export command="help"
        usage # Make sure usage function is defined
        ;;
    -d | --debug)
        shift 1
        DEBUG=true
        ;;
    *) # preserve positional arguments
        PARAMS+="${1} "
        shift
        ;;
    esac
  done

  args=($PARAMS) # PARAMS should be initialized: PARAMS=()
  if [[ -z "$command" ]]; then
    usage # Make sure usage function is defined
  fi  
}

process_command() {
  case "$command" in
  init)
    init
    ;;
  run)
    # For run, if it calls dataload, dataload needs to know how to get its args.
    # This example assumes 'run' might orchestrate, but specific args for sub-commands like dataload
    # are passed when 'data-load' is the primary command.
    run 
    ;;    
  check)
    check
    ;;
  dataload)
    # Pass the stored arguments to the dataload function
    dataload "$DATALOAD_RSS_URL" "$DATALOAD_SITE_NAME"
    ;;
  initpython)
    init_python
    ;;
  view)
    view # Make sure view function is defined
    ;;
  app)
    app
    ;;
  help) # Added help case
    usage
    ;;
  esac
}

# Define usage function (example)
usage() {
  echo "Usage: $me [command] [options]"
  echo "Commands:"
  echo "  data-load [rss_url] [site_name]  Load data from RSS feed. If rss_url and site_name are not provided, you will be prompted."
  echo "  app                               Run the NLWeb server."
  echo "  init                              Initialize configurations."
  echo "  init-python                       Initialize Python environment."
  echo "  check                             Check connectivity."
  echo "  run                               Run full sequence (init, check, dataload, app)."
  echo "  help                              Show this help message."
  echo "Options:"
  echo "  -d, --debug                       Enable debug mode."
  exit 1
}


function main(){
    # Define CURRENT_DIR here if not defined globally or passed
    CURRENT_DIR=$(pwd)
    # Initialize PARAMS for parse_args
    PARAMS=()

    show_banner

    parse_args "$@"
    process_command
}

# invoke main last to ensure all functions and variables are defined
main "$@"

