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
const MAX_DRIFT = 0.35;   // resync if we fall this far behind

export class AudioPlayback {
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private sources = new Set<AudioBufferSourceNode>();
  private nextStartTime = 0;
  private levels = new Uint8Array(0);

  constructor(private readonly sampleRate: number = 24000) {}

  async start(): Promise<void> {
    if (this.ctx) return;
    // Note: the output context runs at the TTS rate, independent of the 16 kHz
    // capture context. Browsers happily run two contexts at different rates.
    this.ctx = new AudioContext({ sampleRate: this.sampleRate });
    this.gain = this.ctx.createGain();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.smoothingTimeConstant = 0.75;
    this.levels = new Uint8Array(this.analyser.frequencyBinCount);

    this.gain.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
    this.nextStartTime = this.ctx.currentTime + LEAD_TIME;

    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  /** Enqueue one PCM16LE chunk. */
  enqueue(pcm: ArrayBuffer): void {
    if (!this.ctx || !this.gain) return;

    const view = new Int16Array(pcm);
    if (view.length === 0) return;

    const buffer = this.ctx.createBuffer(1, view.length, this.sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < view.length; i += 1) channel[i] = view[i] / 32768;

    const now = this.ctx.currentTime;
    // If the stream stalled (or this is the first chunk of a turn) the cursor
    // is in the past; restart it ahead of now rather than scheduling into a
    // time that has already elapsed, which the browser would play immediately
    // and out of order.
    if (this.nextStartTime < now + 0.005 || this.nextStartTime > now + MAX_DRIFT + LEAD_TIME) {
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
