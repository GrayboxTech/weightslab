# Weights Studio Frontend - Production Ready
# Adapts to certificate availability: uses HTTPS if certs present, falls back to HTTP

FROM nginx:alpine

ARG VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
ARG VITE_GRPC_AUTH_TOKEN=

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Create a placeholder index.html for now
# When frontend source is available, replace with built assets
RUN mkdir -p /usr/share/nginx/html && \
    echo '<!DOCTYPE html>' > /usr/share/nginx/html/index.html && \
    echo '<html><head><title>Weights Studio</title></head>' >> /usr/share/nginx/html/index.html && \
    echo '<body><h1>Weights Studio - Frontend Loading</h1>' >> /usr/share/nginx/html/index.html && \
    echo '<p>Backend connection ready. Frontend source will be integrated here.</p></body>' >> /usr/share/nginx/html/index.html && \
    echo '</html>' >> /usr/share/nginx/html/index.html

# Copy entrypoint script that dynamically configures nginx based on available certs
COPY docker/nginx-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 80 443

ENTRYPOINT ["/docker-entrypoint.sh"]
