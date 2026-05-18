# Build the Vue app, then serve the static bundle from nginx-alpine.

FROM node:22-slim AS builder

WORKDIR /app
COPY package.json pnpm-lock.yaml* package-lock.json* ./

# Prefer reproducible installs when a lockfile is present, fall back to a
# fresh resolve when bootstrapping a first build.
RUN if [ -f pnpm-lock.yaml ]; then \
        corepack enable && pnpm install --frozen-lockfile; \
    elif [ -f package-lock.json ]; then \
        npm ci; \
    else \
        npm install --no-audit --no-fund; \
    fi

COPY . .
RUN if [ -f pnpm-lock.yaml ]; then pnpm build; else npm run build; fi


FROM nginx:1.27-alpine AS runtime
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD wget --quiet --spider http://localhost/ || exit 1
