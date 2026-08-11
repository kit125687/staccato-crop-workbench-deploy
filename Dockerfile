FROM node:20-slim AS frontend-builder

WORKDIR /frontend
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY index.html vite.config.js ./
COPY src ./src
ENV VITE_PUBLIC_MODE=true \
    VITE_API_BASE=""
RUN pnpm build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend /app/backend
COPY --from=frontend-builder /frontend/dist /app/dist
RUN mkdir -p /app/backend/models && python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', '/app/backend/models/pose_landmarker_lite.task')"

EXPOSE 10000
CMD ["sh", "-c", "python -m uvicorn backend.server:app --host 0.0.0.0 --port ${PORT}"]
