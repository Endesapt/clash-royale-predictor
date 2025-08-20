#!/bin/sh

# Set the path for the config file inside the NGINX root
CONFIG_FILE_PATH=/usr/share/nginx/html/config.js

# Create the config file with environment variable values
# Use default values if the environment variables are not set
cat <<EOF > $CONFIG_FILE_PATH
window.APP_CONFIG = {
  MAX_LENGTH: "${REACT_APP_MAX_LENGTH:-7}",
  PADDING_IDX: "${REACT_APP_PADDING_IDX:-120}",
  INFERENCE_URL: "${REACT_APP_INFERENCE_URL:-http://clashroyale.ddns.net/v2/models/clashroyale/infer}"
};
EOF

echo "Generated config.js:"
cat $CONFIG_FILE_PATH

# Execute the command passed to this script (i.e., start NGINX)
exec "$@"