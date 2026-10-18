#!/usr/bin/env bash
set -euo pipefail

# Reverse proxy that exposes the LinuxONE backend behind a public host.
BACKEND="${BACKEND:-http://127.0.0.1:7070}"
LISTEN_PORT="${LISTEN_PORT:-8080}"

apt-get update -qq && apt-get install -y nginx

cat > /etc/nginx/sites-available/voxafrik <<EOF
server {
  listen ${LISTEN_PORT};
  server_name _;
  location / {
    proxy_pass ${BACKEND};
    proxy_set_header Host \$host;
    proxy_http_version 1.1;
    proxy_read_timeout 120s;
  }
}
EOF

ln -sf /etc/nginx/sites-available/voxafrik /etc/nginx/sites-enabled/voxafrik
nginx -t && systemctl restart nginx
echo "VoxAfrik proxy listening on :${LISTEN_PORT} -> ${BACKEND}"
