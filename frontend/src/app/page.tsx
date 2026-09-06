"use client";

import { useCallback, useEffect, useState } from "react";

import { EngineBadge } from "@/components/EngineBadge";
import { LatencyStrip } from "@/components/LatencyStrip";
import { ModeSwitcher } from "@/components/ModeSwitcher";
import { Transcript } from "@/components/Transcript";
import { VoiceOrb } from "@/components/VoiceOrb";
import { useVoiceSession } from "@/hooks/useVoiceSession";
import type { EnginePreference, HealthResponse, Mode } from "@/lib/types";

export default function Home() {
  const { state, start, stop, setMode, setEngine, interrupt } = useVoiceSession();
  const [topic, setTopic] = useState("");
  const [preflight, setPreflight] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [enginePref, setEnginePref] = useState<EnginePreference>("auto");

  const live = state.status === "live";
  const busy = state.status === "connecting" || state.status === "requesting-mic";

  // Probe hardware before anyone presses anything, so the badge is populated on
  // first paint rather than after the handshake. The diagnostic is the first
  // thing a reviewer looks at.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled) return;
        if (data) {
          setPreflight(data);
          setHealthError(false);
        } else {
          setHealthError(true);
        }
      })
      .catch(() => { if (!cancelled) setHealthError(true); });
    return () => { cancelled = true; };
  }, []);

  const handleStart = useCallback(() => {
    void start({ mode: state.mode, engine: enginePref, topic: topic.trim() || undefined });
  }, [start, state.mode, enginePref, topic]);

  const handleModeChange = useCallback((mode: Mode) => {
    setMode(mode);
  }, [setMode]);

  const handleOverride = useCallback((pref: EnginePreference) => {
    setEnginePref(pref);
    if (live) setEngine(pref);
  }, [live, setEngine]);

  const engine = state.engine ?? preflight?.recommended_engine ?? null;
  const engineReason = state.engineReason || preflight?.reason || "";
  const hardware = state.hardware ?? preflight?.hardware ?? null;

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-4 py-6 lg:px-8 lg:py-10">
      <Header version={preflight?.version} />

      <div className="grid flex-1 gap-6 lg:grid-cols-[300px_1fr_340px]">
        {/* ── left rail: setup ── */}
        <aside className="order-2 space-y-4 lg:order-1">
          <EngineBadge
            engine={engine}
            reason={engineReason}
            hardware={hardware}
            preference={enginePref}
            onOverride={handleOverride}
            healthError={healthError}
          />

          <ModeSwitcher mode={state.mode} onChange={handleModeChange} disabled={busy} />

          <div>
            <label htmlFor="topic" className="mb-1.5 block text-[11px] font-semibold uppercase tracking-[0.14em] text-haze-400">
              Scenario <span className="font-normal normal-case tracking-normal text-haze-400/60">(optional)</span>
            </label>
            <input
              id="topic"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              disabled={live || busy}
              placeholder="e.g. backend engineer interview"
              className="w-full rounded-lg border border-white/[0.06] bg-ink-800/70 px-3 py-2 text-[13px] text-haze-200 placeholder:text-haze-400/40 focus:border-casual/40 focus:outline-none focus:ring-1 focus:ring-casual/30 disabled:opacity-50"
            />
          </div>

          <LatencyStrip latency={state.latency} />
        </aside>

        {/* ── centre: the orb ── */}
        <section className="order-1 flex flex-col items-center justify-center gap-6 lg:order-2">
          <VoiceOrb
            mode={state.mode}
            inputLevel={state.inputLevel}
            outputLevel={state.outputLevel}
            userSpeaking={state.userSpeaking}
            agentSpeaking={state.agentSpeaking}
            connected={live}
            stalled={state.stalled}
          />

          <StatusLine
            state={state}
            ollamaStatus={hardware?.ollama_status}
            healthError={healthError && !hardware}
          />

          <div className="flex items-center gap-2">
            {!live ? (
              <button
                type="button"
                onClick={handleStart}
                disabled={busy}
                className="rounded-full bg-white px-7 py-2.5 text-sm font-semibold text-ink-900 transition-all hover:bg-haze-200 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {busy ? "Connecting…" : "Start talking"}
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={interrupt}
                  disabled={!state.agentSpeaking}
                  title="Cut the agent off and take the floor"
                  className="rounded-full border border-white/10 px-5 py-2.5 text-sm font-medium text-haze-300 transition-colors hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-30"
                >
                  Interrupt
                </button>
                <button
                  type="button"
                  onClick={() => void stop()}
                  className="rounded-full bg-white/10 px-6 py-2.5 text-sm font-semibold text-haze-200 transition-colors hover:bg-white/[0.16]"
                >
                  End session
                </button>
              </>
            )}
          </div>

          {state.error && (
            <p role="alert" className="max-w-sm animate-fade-up rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-center text-[13px] leading-relaxed text-red-300">
              {state.error}
            </p>
          )}
        </section>

        {/* ── right rail: transcript ── */}
        <aside className="order-3 flex min-h-[380px] flex-col rounded-2xl border border-white/[0.06] bg-ink-800/40 lg:h-[calc(100vh-11rem)]">
          <header className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-haze-400">
              Transcript
            </h2>
            {state.sessionId && (
              <a
                href={`/api/sessions/${state.sessionId}/export.md`}
                download
                className="rounded-md border border-white/[0.08] px-2 py-1 text-[11px] text-haze-300 transition-colors hover:bg-white/[0.06]"
              >
                Export .md
              </a>
            )}
          </header>
          <div className="min-h-0 flex-1 scrollbar-thin p-3">
            <Transcript turns={state.transcript} mode={state.mode} agentSpeaking={state.agentSpeaking} />
          </div>
        </aside>
      </div>
    </main>
  );
}

function Header({ version }: { version?: string }) {
  return (
    <header className="flex items-center justify-between">
      <div className="flex items-baseline gap-2.5">
        <h1 className="text-lg font-semibold tracking-tight text-white">EchoSync</h1>
        <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-haze-400/70">
          voice agent
        </span>
      </div>
      {version && (
        <span className="font-mono text-[11px] text-haze-400/50">v{version}</span>
      )}
    </header>
  );
}

function StatusLine({
  state, ollamaStatus, healthError,
}: {
  state: ReturnType<typeof useVoiceSession>["state"];
  ollamaStatus?: string;
  healthError?: boolean;
}) {
  const idle = state.status === "idle" || state.status === "closed";
  const text =
    healthError && idle ? "Backend is down — start it with make backend."
    : state.status === "requesting-mic" ? "Waiting for microphone permission…"
    : state.status === "connecting" ? "Connecting to the voice engine…"
    : state.status === "error" ? "Something went wrong"
    : idle && ollamaStatus === "starting" ? "Starting local model…"
    : idle ? "Ready when you are"
    : state.status !== "live" ? "Ready when you are"
    : state.stalled ? "Take your time"
    : state.agentSpeaking ? "EchoSync is speaking — just talk to interrupt"
    : state.userSpeaking ? "Listening…"
    : "Listening";

  return (
    <p className="h-5 text-center text-[13px] text-haze-400 transition-opacity">
      {text}
    </p>
  );
}
