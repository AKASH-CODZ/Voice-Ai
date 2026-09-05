/**
 * Microphone capture worklet.
 *
 * Runs on the audio rendering thread, which is exactly where this work belongs:
 * the old ScriptProcessorNode ran on the main thread, so a React re-render or a
 * garbage collection pause would drop microphone frames mid-word.
 *
 * Responsibilities, in order:
 *   1. Accumulate the browser's 128-sample render quanta into 512-sample frames
 *      (32 ms @ 16 kHz), which is exactly Silero's window on the backend.
 *   2. Resample if the AudioContext did not honour our 16 kHz request. Safari
 *      in particular may hand back the hardware rate instead.
 *   3. Convert to Int16 and transfer the buffer to the main thread — transfer,
 *      not copy, so a 31-messages-per-second stream costs no allocations.
 *   4. Report a smoothed RMS level for the orb, so the visualiser reacts to the
 *      same samples the VAD sees rather than a second analyser node.
 */

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 512;

class RecorderProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || TARGET_RATE;
    this.frameSamples = opts.frameSamples || FRAME_SAMPLES;

    // sampleRate is a global provided by the AudioWorkletGlobalScope.
    this.ratio = sampleRate / this.targetRate;
    this.needsResample = Math.abs(this.ratio - 1) > 1e-6;

    this.buffer = new Float32Array(this.frameSamples);
    this.filled = 0;

    // Fractional read position, carried across quanta so resampling does not
    // drift or click at block boundaries.
    this.readPos = 0;
    this.tail = new Float32Array(0);

    this.level = 0;
    this.muted = false;

    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === "mute") this.muted = !!data.value;
    };
  }

  /** Linear resample with carry-over, appended to whatever we already hold. */
  _resample(input) {
    const merged = new Float32Array(this.tail.length + input.length);
    merged.set(this.tail, 0);
    merged.set(input, this.tail.length);

    const out = [];
    let pos = this.readPos;
    while (pos < merged.length - 1) {
      const i = Math.floor(pos);
      const frac = pos - i;
      out.push(merged[i] * (1 - frac) + merged[i + 1] * frac);
      pos += this.ratio;
    }

    // Keep the last whole sample so the next block interpolates continuously.
    const consumed = Math.floor(pos);
    this.tail = merged.subarray(Math.min(consumed, merged.length - 1));
    this.readPos = pos - consumed;
    return out;
  }

  _emit() {
    const pcm = new Int16Array(this.frameSamples);
    let sumSquares = 0;
    for (let i = 0; i < this.frameSamples; i += 1) {
      const s = Math.max(-1, Math.min(1, this.buffer[i]));
      sumSquares += s * s;
      pcm[i] = s * 32767;
    }

    const rms = Math.sqrt(sumSquares / this.frameSamples);
    // Asymmetric smoothing: rise fast so the orb reacts on the attack, fall
    // slowly so it does not flicker between syllables.
    this.level = rms > this.level ? this.level * 0.4 + rms * 0.6 : this.level * 0.85 + rms * 0.15;

    this.port.postMessage({ type: "frame", pcm: pcm.buffer, level: this.level }, [pcm.buffer]);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    if (this.muted) {
      // Still decay the level so the orb settles instead of freezing lit.
      this.level *= 0.85;
      return true;
    }

    const samples = this.needsResample ? this._resample(channel) : channel;

    for (let i = 0; i < samples.length; i += 1) {
      this.buffer[this.filled] = samples[i];
      this.filled += 1;
      if (this.filled === this.frameSamples) this._emit();
    }
    return true;
  }
}

registerProcessor("recorder-processor", RecorderProcessor);
