// Chrono Info — transcription service
//
// Turns a Close call recording into text, using Whisper (open weights, ONNX
// runtime) entirely on Tyler's own Railway. No third-party speech API, no
// per-minute vendor cost, nothing leaves his infrastructure.
//
//   POST /transcribe   { "callId": "acti_..." }  or  { "url": "https://..." }
//   Header             x-auth: <TRANSCRIBE_SECRET>
//   -> 200 { text, seconds, model, chars, tookMs }
//
//   POST /transcribe   { "callId": "acti_...", "async": true }
//   -> 202 { jobId, status: "queued" }        the job runs in the background,
//                                              one at a time, in arrival order
//   GET  /jobs/<jobId> -> { jobId, status: queued|running|done|error, result?, error?, ... }
//   Jobs are kept in memory for 24h. A restart forgets them: a caller that gets
//   404 on a job it started should simply start it again.
//
//   WHY ASYNC EXISTS (2026-09-19): the platform edge in front of this service
//   closes any response that takes longer than about five minutes with a bare
//   502 "upstream error". A six-minute phone call takes longer than that to
//   transcribe on this CPU, so the synchronous call can never succeed for it -
//   and the caller (an n8n Code task) is itself killed at 300 seconds. Start,
//   then poll, is the only shape that works for long recordings.
//
//   GET  /health       -> { ok, model, dtype, ready, warmedAt, jobs }
//
// Why the audio is decoded in JS rather than with ffmpeg: the container stays
// small and dependency-free. mpg123-decoder is a WASM MP3 decoder, so there is
// no system audio binary to install or keep patched.

import http from 'node:http';
import { MPEGDecoder } from 'mpg123-decoder';
import { pipeline } from '@huggingface/transformers';

const PORT = Number(process.env.PORT || 3000);
const SECRET = process.env.TRANSCRIBE_SECRET || '';
const CLOSE_KEY = process.env.CLOSE_API_KEY || '';
const MODEL = process.env.WHISPER_MODEL || 'onnx-community/distil-small.en';
const DTYPE = process.env.WHISPER_DTYPE || 'q8';
const MAX_SECONDS = Number(process.env.MAX_AUDIO_SECONDS || 5400); // 90 min

const closeAuth = CLOSE_KEY ? 'Basic ' + Buffer.from(CLOSE_KEY + ':').toString('base64') : '';

let asr = null;
let warmedAt = null;
let loading = null;

// One shared model instance. Loading is idempotent and concurrent callers wait
// on the same promise rather than each pulling their own copy into memory.
function getModel() {
  if (asr) return Promise.resolve(asr);
  if (!loading) {
    loading = pipeline('automatic-speech-recognition', MODEL, { dtype: DTYPE })
      .then((p) => { asr = p; warmedAt = new Date().toISOString(); return p; })
      .catch((e) => { loading = null; throw e; });
  }
  return loading;
}

async function fetchAudio(url) {
  const headers = {};
  // Close recording URLs need the API key; anything else is fetched plain.
  if (closeAuth && /(^|\.)close\.com\//.test(url)) headers.Authorization = closeAuth;
  const r = await fetch(url, { headers, redirect: 'follow' });
  if (!r.ok) throw new Error('recording fetch failed: ' + r.status);
  return Buffer.from(await r.arrayBuffer());
}

// mp3 -> mono Float32 at 16kHz, the only shape Whisper accepts.
async function toPcm16k(buf) {
  const dec = new MPEGDecoder();
  await dec.ready;
  let decoded;
  try {
    decoded = dec.decode(new Uint8Array(buf));
  } finally {
    dec.free();
  }
  const { channelData, sampleRate } = decoded;
  if (!channelData || !channelData.length || !channelData[0].length) {
    throw new Error('no audio decoded — not an mp3, or empty recording');
  }
  const mono = channelData.length > 1
    ? Float32Array.from(channelData[0], (v, i) => (v + channelData[1][i]) / 2)
    : channelData[0];

  if (sampleRate === 16000) return { pcm: mono, seconds: mono.length / 16000 };

  // Linear decimation is enough here: the source is 8-22kHz phone audio, and
  // Whisper's own front end is far more forgiving than the sample rate.
  const ratio = sampleRate / 16000;
  const out = new Float32Array(Math.floor(mono.length / ratio));
  for (let i = 0; i < out.length; i++) out[i] = mono[Math.floor(i * ratio)];
  return { pcm: out, seconds: out.length / 16000 };
}

// ----------------------------------------------------------------- jobs
const jobs = new Map();
let queue = Promise.resolve();
const JOB_TTL_MS = 24 * 60 * 60 * 1000;

function pruneJobs() {
  const cutoff = Date.now() - JOB_TTL_MS;
  for (const [id, j] of jobs) if (j.createdAt < cutoff) jobs.delete(id);
}

async function transcribeTarget(target) {
  const started = Date.now();
  const audio = await fetchAudio(target);
  const { pcm, seconds } = await toPcm16k(audio);
  if (seconds > MAX_SECONDS) {
    const e = new Error('recording too long'); e.code = 413; e.seconds = seconds; throw e;
  }
  // A ring with no conversation is not worth a model pass.
  if (seconds < 1.5) {
    return { text: '', seconds, model: MODEL, chars: 0, skipped: 'too short', tookMs: Date.now() - started };
  }
  const model = await getModel();
  const out = await model(pcm, { chunk_length_s: 30, stride_length_s: 5 });
  const text = String((out && out.text) || '').trim();
  return { text, seconds: Math.round(seconds * 10) / 10, model: MODEL, chars: text.length, tookMs: Date.now() - started };
}

function enqueueJob(target, callId) {
  pruneJobs();
  const id = 'job_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
  const job = { jobId: id, status: 'queued', callId: callId || null, createdAt: Date.now(), startedAt: null, finishedAt: null, result: null, error: null };
  jobs.set(id, job);
  queue = queue.then(async () => {
    job.status = 'running';
    job.startedAt = Date.now();
    try {
      job.result = await transcribeTarget(target);
      job.status = 'done';
    } catch (e) {
      job.status = 'error';
      job.error = String((e && e.message) || e).slice(0, 300);
      job.errorCode = e && e.code ? e.code : null;
      console.error('job failed:', id, job.error);
    } finally {
      job.finishedAt = Date.now();
    }
  });
  return job;
}

function publicJob(j) {
  return {
    jobId: j.jobId, status: j.status, callId: j.callId,
    createdAt: new Date(j.createdAt).toISOString(),
    startedAt: j.startedAt ? new Date(j.startedAt).toISOString() : null,
    finishedAt: j.finishedAt ? new Date(j.finishedAt).toISOString() : null,
    result: j.result, error: j.error, errorCode: j.errorCode || null,
  };
}

function send(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (c) => {
      size += c.length;
      if (size > 2 * 1024 * 1024) { reject(new Error('body too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')); }
      catch (e) { reject(new Error('bad json')); }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');

  if (req.method === 'GET' && url.pathname.startsWith('/jobs/')) {
    if (SECRET && req.headers['x-auth'] !== SECRET) return send(res, 401, { error: 'unauthorized' });
    const j = jobs.get(url.pathname.slice('/jobs/'.length));
    if (!j) return send(res, 404, { error: 'unknown job' });
    return send(res, 200, publicJob(j));
  }

  if (req.method === 'GET' && url.pathname === '/health') {
    return send(res, 200, { ok: true, model: MODEL, dtype: DTYPE, ready: !!asr, warmedAt, jobs: jobs.size });
  }

  if (req.method !== 'POST' || url.pathname !== '/transcribe') {
    return send(res, 404, { error: 'not found' });
  }

  if (SECRET && req.headers['x-auth'] !== SECRET) {
    return send(res, 401, { error: 'unauthorized' });
  }

  const started = Date.now();
  try {
    const body = await readBody(req);
    const target = body.url
      || (body.callId ? 'https://api.close.com/call/' + encodeURIComponent(body.callId) + '/recording/' : '');
    if (!target) return send(res, 400, { error: 'callId or url required' });

    if (body.async === true || body.async === 'true') {
      return send(res, 202, publicJob(enqueueJob(target, body.callId || null)));
    }

    const audio = await fetchAudio(target);
    const { pcm, seconds } = await toPcm16k(audio);

    if (seconds > MAX_SECONDS) {
      return send(res, 413, { error: 'recording too long', seconds, max: MAX_SECONDS });
    }
    // A ring with no conversation is not worth a model pass.
    if (seconds < 1.5) {
      return send(res, 200, { text: '', seconds, model: MODEL, chars: 0, skipped: 'too short' });
    }

    const model = await getModel();
    const out = await model(pcm, { chunk_length_s: 30, stride_length_s: 5 });
    const text = String((out && out.text) || '').trim();

    return send(res, 200, {
      text,
      seconds: Math.round(seconds * 10) / 10,
      model: MODEL,
      chars: text.length,
      tookMs: Date.now() - started,
    });
  } catch (e) {
    const msg = String((e && e.message) || e).slice(0, 300);
    console.error('transcribe failed:', msg);
    return send(res, 500, { error: msg, tookMs: Date.now() - started });
  }
});

// Long recordings take minutes; don't let the platform hang up mid-pass.
server.requestTimeout = 0;
server.headersTimeout = 0;
server.timeout = 0;
server.keepAliveTimeout = 65000;

server.listen(PORT, () => {
  console.log('transcribe listening on ' + PORT + ' · model ' + MODEL + ' (' + DTYPE + ')');
  // Load the model at boot so the first real call isn't the one that pays for it.
  getModel().then(() => console.log('model ready')).catch((e) => console.error('warm failed:', e.message));
});
