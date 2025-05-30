#!/usr/bin/env bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_DIR=$SCRIPT_DIR/../../

source "$SCRIPT_DIR/../lib/shell_logger.sh"
source "$SCRIPT_DIR/../lib/shell_inputs.sh"
# We are not sourcing llm_providers.sh anymore to avoid its logic for this specific case

ENV_FILE="$REPO_DIR/code/.env"

# Ensure .env file exists, copy from template if not
if [ ! -f "$ENV_FILE" ]; then
  if [ -f "${REPO_DIR}code/.env.template" ]; then
    cp "${REPO_DIR}code/.env.template" "$ENV_FILE"
    _info "Copied .env.template to .env"
  else
    touch "$ENV_FILE"
    _info "Created empty .env file"
  fi
fi

# Function to update or add a variable to the .env file
update_env_var() {
  local var_name="$1"
  local var_value="$2"
  if grep -q "^$var_name=" "$ENV_FILE"; then
    sed -i "s|^$var_name=.*|$var_name=\"$var_value\"|" "$ENV_FILE"
  else
    echo "$var_name=\"$var_value\"" >> "$ENV_FILE"
  fi
}

# Prompt for Cloudflare variables
_prompt_input "Please enter value for CLOUDFLARE_API_TOKEN" CF_TOKEN_VAL
update_env_var "CLOUDFLARE_API_TOKEN" "$CF_TOKEN_VAL"

_prompt_input "Please enter value for CLOUDFLARE_ACCOUNT_ID" CF_ACCOUNT_ID_VAL
update_env_var "CLOUDFLARE_ACCOUNT_ID" "$CF_ACCOUNT_ID_VAL"

_prompt_input "Please enter value for CLOUDFLARE_GATEWAY_ID" CF_GATEWAY_ID_VAL
update_env_var "CLOUDFLARE_GATEWAY_ID" "$CF_GATEWAY_ID_VAL"

_info "Cloudflare environment variables updated in $ENV_FILE" 