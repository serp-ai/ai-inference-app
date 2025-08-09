# Stage 1: Builder
FROM node:22-slim as builder
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y git make g++ python3

WORKDIR /app
COPY . .
RUN npm install

# Build the application
RUN npm run build

# Stage 2: Deploy
FROM node:22-slim as deploy
COPY --from=builder /app/.output /app/output
ENV HOST=0.0.0.0
ENV PORT=80
EXPOSE 80
ENTRYPOINT ["node", "/app/output/server/index.mjs"]
