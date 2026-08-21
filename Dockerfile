# Chrono Info — transcription service.
# The model is baked in at build time so a cold container never makes a webhook
# wait on a few hundred MB of weights.
FROM node:22-slim

WORKDIR /app
ENV NODE_ENV=production
ENV WHISPER_MODEL=onnx-community/whisper-small.en
ENV WHISPER_DTYPE=q8
# transformers.js caches under HF_HOME; keep it inside the image layer.
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

COPY package.json ./
RUN npm install --omit=dev --no-audit --no-fund

COPY warm.mjs server.mjs ./
RUN node warm.mjs

EXPOSE 3000
CMD ["node", "server.mjs"]
