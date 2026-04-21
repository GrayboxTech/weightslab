# Weights Studio Frontend - Production Ready
FROM node:18-alpine as builder

WORKDIR /app
ARG VITE_WL_ENABLE_GRPC_AUTH_TOKEN=0
ARG VITE_GRPC_AUTH_TOKEN=

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY . .

# Build with build args
RUN VITE_WL_ENABLE_GRPC_AUTH_TOKEN=${VITE_WL_ENABLE_GRPC_AUTH_TOKEN} \
    VITE_GRPC_AUTH_TOKEN=${VITE_GRPC_AUTH_TOKEN} \
    npm run build

# Production server
FROM nginx:alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy certificates for HTTPS
COPY envoy/certs /app/envoy/certs

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]
