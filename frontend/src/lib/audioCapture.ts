/**
 * Microphone capture: getUserMedia → AudioWorklet → Int16 frames.
 *
 * The three constraints below are not optional garnish — they are the fix for
 * the "double-talk echo" failure in the plan. Without `echoCancellation` the
 * microphone re-captures the agent's own voice from the speakers, Whisper
 * transcribes it as user speech, and the agent answers itself in a loop that
 * never terminates. AEC is the browser's job and it does it well; our
 * server-side mic gating is the second layer, not the first.
 */

export interface CaptureOptions {
  sampleRate?: number;
  frameSamples?: number;
  onFrame: (pcm: ArrayBuffer) => void;
  onLevel?: (level: number) => void;
}

export class AudioCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;

  readonly sampleRate: number;
  private frameSamples: number;
  private onFrame: (pcm: ArrayBuffer) => void;
  private onLevel?: (level: number) => void;

  constructor(options: CaptureOptions) {
    this.sampleRate = options.sampleRate ?? 16000;
    this.frameSamples = options.frameSamples ?? 512;
    this.onFrame = options.onFrame;
    this.onLevel = options.onLevel;
  }

  async start(): Promise<void> {
    if (this.ctx) return;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,   // stops the agent hearing itself — see above
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
        sampleRate: this.sampleRate,
      },
      video: false,
    });

    // Ask for 16 kHz directly so the worklet usually has nothing to resample.
    // Safari may ignore this and hand back the hardware rate; the worklet
    // handles that case rather than us failing.
    this.ctx = new AudioContext({ sampleRate: this.sampleRate });
    if (this.ctx.state === "suspended") await this.ctx.resume();

    await this.ctx.audioWorklet.addModule("/worklets/recorder-processor.js");

    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "recorder-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { targetRate: this.sampleRate, frameSamples: this.frameSamples },
    });

    this.node.port.onmessage = (event: MessageEvent) => {
      const data = event.data;
      if (data?.type !== "frame") return;
      this.onFrame(data.pcm as ArrayBuffer);
      this.onLevel?.(data.level as number);
    };

    this.source.connect(this.node);
  }

  /** Stop sending frames without tearing down the mic permission prompt. */
  setMuted(muted: boolean): void {
    this.node?.port.postMessage({ type: "mute", value: muted });
  }

  /** True if the browser gave us a different rate than we asked for. */
  get actualSampleRate(): number {
    return this.ctx?.sampleRate ?? this.sampleRate;
  }

  async close(): Promise<void> {
    try { this.source?.disconnect(); } catch { /* ignore */ }
    try { this.node?.disconnect(); } catch { /* ignore */ }
    this.stream?.getTracks().forEach((t) => t.stop());
    if (this.ctx) await this.ctx.close().catch(() => undefined);
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.source = null;
  }
}
