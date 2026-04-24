#!/bin/sh
# Dynamic nginx configuration based on certificate availability
# Adapts to HTTPS if certs exist, falls back to HTTP if missing
# Injects gRPC auth token at runtime

set -e

WEIGHTSLAB_CERTS_DIR="/etc/nginx/certs"
CERT_FILE="$WEIGHTSLAB_CERTS_DIR/envoy-server.crt"
KEY_FILE="$WEIGHTSLAB_CERTS_DIR/envoy-server.key"
NGINX_CONF="/etc/nginx/conf.d/default.conf"

# Check if certificates exist
HAS_CERTS=false
if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    HAS_CERTS=true
    echo "✓ TLS certificates found - enabling HTTPS"
else
    echo "✗ TLS certificates not found - using HTTP"
fi

# Create config directory for token injection
mkdir -p /tmp/nginx_conf

# Inject token into JavaScript config file that will be served
TOKEN_FILE="/tmp/nginx_conf/grpc-config.js"
cat > "$TOKEN_FILE" << TOKEN_END
// Runtime gRPC configuration (injected at container start)
window.__grpcConfig = {
  authToken: '${GRPC_AUTH_TOKEN:-}',
  authEnabled: ${WL_ENABLE_GRPC_AUTH_TOKEN:-0},
  protocol: '${WS_SERVER_PROTOCOL:-https}'
};
TOKEN_END

echo "✓ Runtime gRPC config injected"

# Generate nginx configuration based on certificate availability
if [ "$HAS_CERTS" = true ]; then
    # HTTPS configuration
    cat > "$NGINX_CONF" << 'NGINX_HTTPS_END'
upstream envoy {
    server weightslab_envoy:8080;
}

server {
    listen 80;
    listen 443 ssl;
    server_name _;

    ssl_certificate /etc/nginx/certs/envoy-server.crt;
    ssl_certificate_key /etc/nginx/certs/envoy-server.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    root /usr/share/nginx/html;
    index index.html;

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API proxy to backend through Envoy
    location /api/ {
        proxy_pass http://envoy/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # gRPC proxy
    location ~ ^/grpc {
        proxy_pass http://envoy;
        proxy_http_version 1.1;
        proxy_set_header Content-Type application/grpc;
        proxy_set_header grpc-encoding gzip;
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_set_header Host $host;
    }

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Serve runtime gRPC config
    location /grpc-config.js {
        alias /tmp/nginx_conf/grpc-config.js;
        add_header Content-Type application/javascript;
        add_header Cache-Control "no-cache";
    }

    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
NGINX_HTTPS_END
else
    # HTTP configuration
    cat > "$NGINX_CONF" << 'NGINX_HTTP_END'
upstream envoy {
    server weightslab_envoy:8080;
}

server {
    listen 80;
    listen 443;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API proxy to backend through Envoy
    location /api/ {
        proxy_pass http://envoy/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # gRPC proxy
    location ~ ^/grpc {
        proxy_pass http://envoy;
        proxy_http_version 1.1;
        proxy_set_header Content-Type application/grpc;
        proxy_set_header grpc-encoding gzip;
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_set_header Host $host;
    }

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Serve runtime gRPC config
    location /grpc-config.js {
        alias /tmp/nginx_conf/grpc-config.js;
        add_header Content-Type application/javascript;
        add_header Cache-Control "no-cache";
    }

    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
NGINX_HTTP_END
fi

# Validate nginx configuration
echo "Validating nginx configuration..."
nginx -t

# Start nginx in foreground
echo "Starting nginx..."
exec nginx -g "daemon off;"
