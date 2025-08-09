# Combined deployment using existing Docker images
# Uses the pre-built backend as base and adds frontend

# Build frontend
FROM node:22-slim as frontend-builder
WORKDIR /app
COPY frontend/ .
RUN npm install && npm run build

# Use the backend image as base and add frontend + process management
ARG BACKEND_IMAGE="test:cuda"
FROM ${BACKEND_IMAGE}

# Install Node.js and supervisor for running both services
RUN apt-get update && apt-get install -y supervisor nodejs npm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy frontend build
COPY --from=frontend-builder /app/.output /frontend/

# Copy process management files
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY start-combined.sh /start-combined.sh
RUN chmod +x /start-combined.sh

# Expose both ports (frontend: 80, backend: 8000)
EXPOSE 80 8000

CMD ["/start-combined.sh"]