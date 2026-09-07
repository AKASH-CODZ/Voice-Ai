/**
 * Gapless PCM playback for streamed TTS.
 *
 * The naive approach — `new Audio()` per chunk, or scheduling every buffer at
 * `currentTime` — produces an audible click between every sentence, because
 * each chunk starts at the next render quantum rather than exactly where the
 * previous one ended. Instead we keep a running `nextStartTime` cursor in the
 * AudioContext's own clock and schedule each buffer to begin precisely when its
 * predecessor finishes. Sample-accurate, no crossfade needed.
 *
 * `LEAD_TIME` is the one tunable: it is the safety margin between "now" and the
 * first scheduled buffer. Too small and a slow network underruns into a gap;
 * too large and barge-in feels laggy because there is more already-committed
 * audio to throw away.
 */

const LEAD_TIME = 0.08;   // 80 ms
// Only an *underrun* (cursor in the past) should resync. A previous MAX_DRIFT
// cap of 350 ms treated healthy queued TTS as an error: after ~3×120 ms
// slices the next chunk was scheduled at "now", overlapping the first 2–3
// words and dropping the rest of a 20-word reply. Cap runaway clocks only.

export class AudioPlayback {
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private sources = new Set<AudioBufferSourceNode>();
  private nextStartTime = 0;
  private levels = new Uint8Array(0);

  constructor(private readonly sampleRate: number = 24000) {}

  /**
   * Create the output AudioContext inside a user-gesture call stack.
   * Must run *before* any `await` (getUserMedia, etc.). localhost is exempt
   * from autoplay rules; https://echosync-web.onrender.com is not, so a
   * context created after the click's first await stays suspended forever
   * — transcripts still stream, speakers stay silent.
   */
  prime(): void {
    if (this.ctx) return;
    // Output runs at the TTS rate, independent of the 16 kHz capture context.
    this.ctx = new AudioContext({ sampleRate: this.sampleRate });
    this.gain = this.ctx.createGain();
    this.gain.gain.value = 1;
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.smoothingTimeConstant = 0.75;
    this.levels = new Uint8Array(this.analyser.frequencyBinCount);

    this.gain.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
    this.nextStartTime = this.ctx.currentTime + LEAD_TIME;
    void this.ctx.resume();
  }

  async start(): Promise<void> {
    this.prime();
    if (this.ctx?.state === "suspended") await this.ctx.resume();
  }

  /** Enqueue one PCM16LE chunk. */
  enqueue(pcm: ArrayBuffer): void {
    if (!this.ctx || !this.gain) return;
    if (this.ctx.state === "suspended") void this.ctx.resume();

    if (pcm.byteLength < 2) return;
    // Copy: some browsers reuse the WebSocket ArrayBuffer on the next frame.
    const copy = pcm.slice(0);
    const view = new Int16Array(copy, 0, Math.floor(copy.byteLength / 2));
    if (view.length === 0) return;

    const ctxRate = this.ctx.sampleRate;
    const floats = resampleInt16(view, this.sampleRate, ctxRate);
    const buffer = this.ctx.createBuffer(1, floats.length, ctxRate);
    buffer.getChannelData(0).set(floats);

    const now = this.ctx.currentTime;
    // If the stream stalled (or this is the first chunk of a turn) the cursor
    // is in the past; restart it ahead of now rather than scheduling into a
    // time that has already elapsed, which the browser would skip.
    // Do NOT reset when the cursor is in the future — that is queued speech.
    if (this.nextStartTime < now + 0.005) {
      this.nextStartTime = now + LEAD_TIME;
    }

    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.gain);
    source.start(this.nextStartTime);
    this.nextStartTime += buffer.duration;

    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      try { source.disconnect(); } catch { /* already torn down */ }
    };
  }

  /** Barge-in: drop everything already scheduled, immediately. */
  stop(): void {
    for (const source of this.sources) {
      try { source.stop(); } catch { /* may have finished between calls */ }
      try { source.disconnect(); } catch { /* ignore */ }
    }
    this.sources.clear();
    if (this.ctx) this.nextStartTime = this.ctx.currentTime + LEAD_TIME;
  }

  /** Normalised 0..1 output loudness, for the orb. */
  level(): number {
    if (!this.analyser) return 0;
    this.analyser.getByteFrequencyData(this.levels);
    let sum = 0;
    for (let i = 0; i < this.levels.length; i += 1) sum += this.levels[i];
    return sum / this.levels.length / 255;
  }

  /** Seconds of audio still scheduled — the true "is it talking" signal. */
  pendingSeconds(): number {
    if (!this.ctx) return 0;
    return Math.max(0, this.nextStartTime - this.ctx.currentTime);
  }

  async close(): Promise<void> {
    this.stop();
    if (this.ctx) {
      await this.ctx.close().catch(() => undefined);
      this.ctx = null;
      this.gain = null;
      this.analyser = null;
    }
  }
}

function resampleInt16(view: Int16Array, srcRate: number, dstRate: number): Float32Array {
  if (Math.abs(dstRate - srcRate) < 0.5) {
    const out = new Float32Array(view.length);
    for (let i = 0; i < view.length; i += 1) out[i] = view[i] / 32768;
    return out;
  }
  const outLen = Math.max(1, Math.round(view.length * dstRate / srcRate));
  const out = new Float32Array(outLen);
  const step = srcRate / dstRate;
  const last = view.length - 1;
  for (let i = 0; i < outLen; i += 1) {
    const src = i * step;
    const j = Math.min(last, Math.floor(src));
    const k = Math.min(last, j + 1);
    const frac = src - Math.floor(src);
    out[i] = (view[j] * (1 - frac) + view[k] * frac) / 32768;
  }
  return out;
}
