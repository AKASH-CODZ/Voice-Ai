"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AudioCapture } from "@/lib/audioCapture";
import { AudioPlayback } from "@/lib/audioPlayback";
import type {
  ConnectionStatus, EnginePreference, HardwareReport, LatencySample, Mode,
  ServerEvent, TranscriptTurn,
} from "@/lib/types";

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ??
  (typeof window !== "undefined"
    ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8000/ws/voice`
    : "ws://localhost:8000/ws/voice");

export interface VoiceSessionState {
  status: ConnectionStatus;
  sessionId: string | null;
  mode: Mode;
  engine: string | null;
  engineReason: string;
  hardware: HardwareReport | null;
  transcript: TranscriptTurn[];
  userSpeaking: boolean;
  agentSpeaking: boolean;
  stalled: boolean;
  inputLevel: number;
  outputLevel: number;
  latency: LatencySample | null;
  error: string | null;
}

export interface StartOptions {
  mode: Mode;
  engine: EnginePreference;
  topic?: string;
  resumeSessionId?: string;
}

export function useVoiceSession() {
  const [state, setState] = useState<VoiceSessionState>({
    status: "idle",
    sessionId: null,
    mode: "casual",
    engine: null,
    engineReason: "",
    hardware: null,
    transcript: [],
    userSpeaking: false,
    agentSpeaking: false,
    stalled: false,
    inputLevel: 0,
    outputLevel: 0,
    latency: null,
    error: null,
  });

  const socketRef = useRef<WebSocket | null>(null);
  const captureRef = useRef<AudioCapture | null>(null);
  const playbackRef = useRef<AudioPlayback | null>(null);
  const rafRef = useRef<number | null>(null);
  // The assistant's in-flight reply, accumulated from agent_delta events.
  const streamingRef = useRef<string>("");

  const patch = useCallback((next: Partial<VoiceSessionState>) => {
    setState((prev) => ({ ...prev, ...next }));
  }, []);

  // ── outbound control ───────────────────────────────────────
  const send = useCallback((payload: Record<string, unknown>) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
  }, []);

  // ── inbound events ─────────────────────────────────────────
  const handleEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case "ready":
        patch({
          status: "live",
          sessionId: event.session_id,
          mode: event.mode,
          engine: event.engine,
          engineReason: event.engine_reason,
          hardware: event.hardware,
          error: null,
        });
        break;

      case "vad":
        patch({ userSpeaking: event.speaking, stalled: false });
        break;

      case "speaking":
        if (event.active) streamingRef.current = "";
        patch({ agentSpeaking: event.active });
        if (!event.active) {
          // Promote the streamed draft to a settled turn if the authoritative
          // `transcript` event has not already landed.
          streamingRef.current = "";
          setState((prev) => ({
            ...prev,
            transcript: prev.transcript.filter((t) => !t.streaming || t.text.trim() !== ""),
          }));
        }
        break;

      case "agent_delta": {
        streamingRef.current += event.text;
        const draft = streamingRef.current;
        setState((prev) => {
          const rest = prev.transcript.filter((t) => !t.streaming);
          return {
            ...prev,
            transcript: [...rest, { id: "streaming", role: "assistant", text: draft, streaming: true }],
          };
        });
        break;
      }

      case "transcript":
        setState((prev) => ({
          ...prev,
          // Drop the streaming draft — this event is the authoritative version.
          transcript: [
            ...prev.transcript.filter((t) => !(t.streaming && event.role === "assistant")),
            {
              id: `${event.role}-${event.turn_id}-${prev.transcript.length}`,
              role: event.role,
              text: event.text,
              hesitation: event.hesitation,
              correctionMade: event.correction_made,
            },
          ],
        }));
        break;

      case "stall":
        patch({ stalled: true });
        break;

      case "latency":
        patch({ latency: event });
        break;

      case "engine_switched":
        patch({ engine: event.engine, engineReason: event.reason });
        break;

      case "error":
        patch({ error: event.message, status: event.fatal ? "error" : undefined as never });
        break;

      default:
        break;
    }
  }, [patch]);

  // ── level meter loop ───────────────────────────────────────
  const startLevelLoop = useCallback(() => {
    const tick = () => {
      const playback = playbackRef.current;
      if (playback) {
        setState((prev) => {
          const out = playback.level();
          // Only re-render when the change is visible; a raw 60 fps setState
          // on every frame would thrash React for sub-pixel differences.
          if (Math.abs(out - prev.outputLevel) < 0.012) return prev;
          return { ...prev, outputLevel: out };
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  // ── lifecycle ──────────────────────────────────────────────
  const stop = useCallback(async () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    try { socketRef.current?.send(JSON.stringify({ type: "stop" })); } catch { /* closing */ }
    socketRef.current?.close();
    socketRef.current = null;

    await captureRef.current?.close();
    captureRef.current = null;
    await playbackRef.current?.close();
    playbackRef.current = null;

    setState((prev) => ({
      ...prev,
      status: "closed",
      userSpeaking: false,
      agentSpeaking: false,
      inputLevel: 0,
      outputLevel: 0,
    }));
  }, []);

  const start = useCallback(async (options: StartOptions) => {
    if (socketRef.current) await stop();

    patch({ status: "requesting-mic", error: null, transcript: [], latency: null, mode: options.mode });

    let capture: AudioCapture;
    try {
      capture = new AudioCapture({
        onFrame: (pcm) => {
          const socket = socketRef.current;
          if (socket?.readyState === WebSocket.OPEN) socket.send(pcm);
        },
        onLevel: (level) => {
          setState((prev) =>
            Math.abs(level - prev.inputLevel) < 0.012 ? prev : { ...prev, inputLevel: level });
        },
      });
      await capture.start();
      captureRef.current = capture;
    } catch (err) {
      patch({
        status: "error",
        error:
          err instanceof DOMException && err.name === "NotAllowedError"
            ? "Microphone permission denied. Allow access and try again."
            : `Could not open the microphone: ${(err as Error).message}`,
      });
      return;
    }

    const playback = new AudioPlayback(24000);
    await playback.start();
    playbackRef.current = playback;
    startLevelLoop();

    patch({ status: "connecting" });

    const socket = new WebSocket(WS_URL);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    socket.onopen = () => {
      socket.send(JSON.stringify({
        type: "start",
        mode: options.mode,
        engine: options.engine,
        topic: options.topic ?? null,
        resume_session_id: options.resumeSessionId ?? null,
      }));
    };

    socket.onmessage = (event: MessageEvent) => {
      if (typeof event.data === "string") {
        try { handleEvent(JSON.parse(event.data) as ServerEvent); } catch { /* ignore */ }
        return;
      }
      playbackRef.current?.enqueue(event.data as ArrayBuffer);
    };

    socket.onerror = () => {
      patch({ status: "error", error: "Connection to the voice backend failed. Is it running?" });
    };

    socket.onclose = () => {
      setState((prev) =>
        prev.status === "error" ? prev : { ...prev, status: "closed", agentSpeaking: false });
    };
  }, [handleEvent, patch, startLevelLoop, stop]);

  const setMode = useCallback((mode: Mode) => {
    patch({ mode });
    send({ type: "mode", mode });
  }, [patch, send]);

  const setEngine = useCallback((engine: EnginePreference) => {
    send({ type: "engine", engine });
  }, [send]);

  const interrupt = useCallback(() => {
    // Cut local audio immediately rather than waiting for the server to stop
    // sending — the already-buffered audio is what the user is complaining
    // about by talking over it.
    playbackRef.current?.stop();
    send({ type: "interrupt" });
    patch({ agentSpeaking: false });
  }, [patch, send]);

  useEffect(() => () => { void stop(); }, [stop]);

  return { state, start, stop, setMode, setEngine, interrupt };
}
