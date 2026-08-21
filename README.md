# Chrono Info — transcription service

Turns Close call recordings into text with **Whisper** (open weights, ONNX runtime),
running on **Tyler's own Railway**. No third-party speech API, no per-minute vendor
cost, and no audio leaving his infrastructure.

## API

```
POST /transcribe
  x-auth: <TRANSCRIBE_SECRET>
  { "callId": "acti_..." }        # fetched from Close with CLOSE_API_KEY
  { "url": "https://..." }        # or any direct mp3 URL

-> { text, seconds, model, chars, tookMs }

GET /health -> { ok, model, dtype, ready, warmedAt }
```

## Environment

| Var | Purpose |
|---|---|
| `TRANSCRIBE_SECRET` | Shared secret, sent as `x-auth`. Required in production |
| `CLOSE_API_KEY` | So the service can pull recordings from Close itself |
| `WHISPER_MODEL` | Default `onnx-community/whisper-base.en` |
| `WHISPER_DTYPE` | Default `q8` — quantised, much smaller and faster, negligible loss on speech |
| `MAX_AUDIO_SECONDS` | Default 5400 (90 min) |

## Design notes

- **No ffmpeg.** MP3 is decoded by `mpg123-decoder`, a WASM decoder, so the image has
  no system audio binary to install or patch.
- **The model is baked into the image** (`RUN node warm.mjs`). Otherwise a cold start
  is a few hundred MB download while a webhook blocks on it.
- **Server timeouts are disabled** — a 20-minute recording legitimately takes minutes,
  and Node's defaults would cut it off mid-pass.
- Recordings under 1.5s return empty rather than burning a model pass on a ring.

`small.en` was chosen over `base.en` because dial audio is phone-quality and often
poor; accuracy matters more here than speed. Measured on this box: `base.en` runs
~6x realtime, so `small.en` is roughly 2-3x realtime — a 5-minute triage dial
transcribes in about two minutes.

## Handover

Transfers to Tyler with the rest of the stack. It is deliberately a separate service
on **his** Railway project, alongside his n8n, so handover stays a no-op.
