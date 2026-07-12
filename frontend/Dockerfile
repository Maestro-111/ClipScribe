# syntax=docker/dockerfile:1
#
# Frontend image: build the Vite/React SPA with pnpm, then serve the static
# bundle with nginx. nginx reverse-proxies /api/* to the API container (SSE-safe)
# — the same path mapping as the Vite dev proxy, so dev and prod behave alike.
#
# Build context is the REPOSITORY ROOT:
#   docker build -f frontend/Dockerfile -t clipscribe-frontend .

# ── Stage 1: build ────────────────────────────────────────────────────────
FROM node:22-slim AS build
RUN corepack enable
WORKDIR /app/frontend

# Install deps first so this layer caches unless the manifest changes.
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

# Build the static bundle (tsr generate + tsc typecheck + vite build). Uses the
# committed src/api/types.ts — no running API needed at build time (regenerate
# those types with `pnpm gen:api` against a live API).
COPY frontend/ ./
RUN pnpm build

# ── Stage 2: serve ────────────────────────────────────────────────────────
FROM nginx:alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/dist /usr/share/nginx/html
EXPOSE 80
