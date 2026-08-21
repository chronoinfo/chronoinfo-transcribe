// Pull the model into the image at BUILD time. Without this the first real
// request pays a several-hundred-MB download while a webhook waits on it.
import { pipeline } from '@huggingface/transformers';

const MODEL = process.env.WHISPER_MODEL || 'onnx-community/whisper-base.en';
const DTYPE = process.env.WHISPER_DTYPE || 'q8';
console.log('warming', MODEL, DTYPE);
await pipeline('automatic-speech-recognition', MODEL, { dtype: DTYPE });
console.log('model cached');
