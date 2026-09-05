"use client";

import type { LatencySample } from "@/lib/types";

/**
 * Per-turn latency breakdown.
 *
 * This is portfolio surface as much as debugging surface: the plan's whole
 * premise is a sub-second loop, and a claim like that is worth more when the
 * page shows its own measured numbers per turn instead of asserting them in a
 * README. `e2e` is the one that matters — the gap between the user finishing a
 * sentence and hearing the first syllable back.
 */

const TARGET_E2E_MS = 800;

interface LatencyStripProps {
  latency: LatencySample | null;
}

export function LatencyStrip({ latency }: LatencyStripProps) {
  if (!latency) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-white/[0.05] bg-ink-800/40 px-3 py-2">
        <span className="font-mono text-[11px] text-haze-400/50">awaiting first turn</span>
      </div>
    );
  }

  const good = latency.e2e_ms <= TARGET_E2E_MS;
  const stages = [
    { label: "stt", value: latency.stt_ms },
    { label: "ttft", value: latency.llm_ttft_ms },
    { label: "tts", value: latency.tts_ms },
  ];
  const total = Math.max(latency.e2e_ms, 1);

  return (
    <div className="rounded-lg border border-white/[0.05] bg-ink-800/40 px-3 py-2">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-haze-400">
          turn latency
        </span>
        <span
          className={`font-mono text-[13px] font-semibold ${good ? "text-emerald-400" : "text-observation"}`}
          title={`End to end: you stop speaking → first audio back. Target ${TARGET_E2E_MS} ms.`}
        >
          {Math.round(latency.e2e_ms)} ms
        </span>
      </div>

      {/* Proportional stage bar — shows where the budget actually went. */}
      <div className="mb-1.5 flex h-1 gap-px overflow-hidden rounded-full bg-ink-900">
        {stages.map((stage, i) => (
          <div
            key={stage.label}
            className={["bg-casual/70", "bg-teaching/70", "bg-observation/70"][i]}
            style={{ width: `${Math.min(100, (stage.value / total) * 100)}%` }}
          />
        ))}
      </div>

      <dl className="flex justify-between font-mono text-[10px] text-haze-400">
        {stages.map((stage) => (
          <div key={stage.label} className="flex gap-1">
            <dt className="text-haze-400/60">{stage.label}</dt>
            <dd>{Math.round(stage.value)}</dd>
          </div>
        ))}
        <div className="flex gap-1">
          <dt className="text-haze-400/60">via</dt>
          <dd className="capitalize">{latency.engine}</dd>
        </div>
      </dl>
    </div>
  );
}
