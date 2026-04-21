# Weights Studio Frontend - Production Ready
# This Dockerfile provides a placeholder frontend server.
# When the full frontend source is available, update to build from source.

FROM nginx:alpine

ARG VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
ARG VITE_GRPC_AUTH_TOKEN=

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Copy Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create a placeholder index.html for now
# When frontend source is available, replace with built assets
RUN mkdir -p /usr/share/nginx/html && \
    echo '<!DOCTYPE html>' > /usr/share/nginx/html/index.html && \
    echo '<html><head><title>Weights Studio</title></head>' >> /usr/share/nginx/html/index.html && \
    echo '<body><h1>Weights Studio - Frontend Loading</h1>' >> /usr/share/nginx/html/index.html && \
    echo '<p>Backend connection ready. Frontend source will be integrated here.</p></body>' >> /usr/share/nginx/html/index.html && \
    echo '</html>' >> /usr/share/nginx/html/index.html

# Copy certificates for HTTPS
COPY envoy/certs /app/envoy/certs

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]
