# Chrono Info — transcription service.
# The model is baked in at build time so a cold container never makes a webhook
# wait on a few hundred MB of weights.
FROM node:22-slim

WORKDIR /app
ENV NODE_ENV=production
# distil-small.en, chosen by measurement on the real Close recording:
#   tiny.en   532MB  -> "I'll look. I'll look."        (hallucinated)
#   base.en   695MB  -> "All of them... a circle. OK."  (hallucinated)
#   distil-sm 923MB  -> "Hello. Okay."                  <- agrees with small
#   small.en 1220MB  -> "Hello. Hello."                 (502s on this container)
# The small models each invent something different on quiet audio; the two
# largest agree. distil-small is the most accurate that actually runs here.
ENV WHISPER_MODEL=onnx-community/distil-small.en
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
