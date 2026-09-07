"use client";

import { useState } from "react";

import { backendDownMessage } from "@/lib/backend";
import type { EnginePreference, HardwareReport } from "@/lib/types";

/**
 * The engine badge, and the honest version of "hardware detection".
 *
 * A reviewer opening this on a laptop with no GPU should be able to see *why*
 * they were routed to the cloud, not just that they were. The collapsed badge
 * is a status light; expanding it shows the actual diagnostic the router ran,
 * verbatim — VRAM, temperature, which instruct model was picked, whether
 * Ollama answered.
 */

interface EngineBadgeProps {
  engine: string | null;
  reason: string;
  hardware: HardwareReport | null;
  preference?: EnginePreference;
  onOverride?: (preference: EnginePreference) => void;
  disabled?: boolean;
  healthError?: boolean;
}

export function EngineBadge({
  engine, reason, hardware, preference, onOverride, disabled, healthError,
}: EngineBadgeProps) {
  const [open, setOpen] = useState(false);

  const isLocal = engine === "local";
  const gpu = hardware?.gpus?.[0];
  const starting = hardware?.ollama_status === "starting";

  const label = healthError && !hardware
    ? "Backend is down"
    : engine === null && starting
      ? "Starting local model…"
      : engine === null
        ? "Detecting hardware…"
        : collapsedLabel(engine, hardware, isLocal, gpu);

  const dot =
    healthError && !hardware ? "bg-red-400"
    : engine === null || starting ? "bg-haze-400"
    : isLocal ? "bg-emerald-400"
    : "bg-sky-400";

  return (
    <div className="w-full">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2.5 rounded-lg border border-white/[0.06] bg-ink-800/70 px-3 py-2 text-left transition-colors hover:bg-ink-700/70"
      >
        <span className={["h-2 w-2 shrink-0 rounded-full", dot].join(" ")} />
        <span className="flex-1 truncate text-[13px] font-medium text-haze-200">{label}</span>
        <span className={`text-haze-400 transition-transform ${open ? "rotate-180" : ""}`}>▾</span>
      </button>

      {open && (
        <div className="mt-1.5 animate-fade-up space-y-3 rounded-lg border border-white/[0.06] bg-ink-800/50 p-3 text-[12px]">
          {healthError && !hardware && (
            <p className="leading-relaxed text-red-300">{backendDownMessage()}</p>
          )}
          {reason && <p className="leading-relaxed text-haze-300">{reason}</p>}

          {hardware && (
            <dl className="space-y-1 font-mono text-[11px] text-haze-400">
              <Row label="CPU" value={`${hardware.cpu_model} · ${hardware.cpu_logical_cores} cores`} />
              <Row label="RAM" value={`${hardware.available_ram_gb.toFixed(1)} / ${hardware.total_ram_gb.toFixed(1)} GB free`} />
              {gpu ? (
                <>
                  <Row label="GPU" value={gpu.name} />
                  <Row label="VRAM" value={`${gpu.free_vram_gb.toFixed(1)} / ${gpu.total_vram_gb.toFixed(1)} GB free`} />
                  {gpu.temperature_c !== null && <Row label="Temp" value={`${gpu.temperature_c}°C`} />}
                </>
              ) : (
                <Row label="GPU" value={hardware.apple_silicon ? "Apple Silicon (Metal)" : "none detected"} />
              )}
              <Row label="Whisper" value={hardware.whisper_device ?? "—"} />
              <Row label="Model" value={hardware.ollama_model ? shortModel(hardware.ollama_model) : "—"} />
              <Row label="Ollama" value={ollamaLabel(hardware)} />
              <Row label="Groq key" value={hardware.cloud_credentials ? "present" : "not set"} />
            </dl>
          )}

          {onOverride && (
            <div className="flex gap-1 border-t border-white/[0.06] pt-2.5">
              {(["auto", "local", "cloud"] as EnginePreference[]).map((pref) => {
                const selected = preference === pref;
                return (
                  <button
                    key={pref}
                    type="button"
                    disabled={disabled}
                    onClick={() => onOverride(pref)}
                    className={[
                      "flex-1 rounded-md border px-2 py-1 text-[11px] capitalize transition-colors",
                      selected
                        ? "border-white/20 bg-white/[0.08] text-haze-100"
                        : "border-white/[0.06] text-haze-300 hover:bg-white/[0.06]",
                      "disabled:cursor-not-allowed disabled:opacity-40",
                    ].join(" ")}
                  >
                    {pref}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function collapsedLabel(
  engine: string,
  hardware: HardwareReport | null,
  isLocal: boolean,
  gpu: HardwareReport["gpus"][number] | undefined,
): string {
  const model = shortModel(hardware?.ollama_model);
  if (isLocal) {
    const device = gpu
      ? shorten(gpu.name)
      : hardware?.apple_silicon
        ? "Apple Silicon"
        : "CPU";
    const free = gpu
      ? `${gpu.free_vram_gb.toFixed(1)} GB free`
      : hardware
        ? `${hardware.available_ram_gb.toFixed(1)} GB RAM free`
        : "";
    return ["Local", model, device, free].filter(Boolean).join(" · ");
  }
  const why =
    hardware?.ollama_status === "missing" ? "Ollama not installed"
    : hardware?.ollama_status === "down" ? "Ollama not running"
    : hardware?.ollama_models?.length && !hardware.ollama_model ? "no instruct model"
    : null;
  return ["Cloud", "Groq", why].filter(Boolean).join(" · ");
}

function ollamaLabel(hardware: HardwareReport): string {
  const status = hardware.ollama_status
    ?? (hardware.ollama_reachable ? "up" : "down");
  if (status === "up" && hardware.ollama_model) {
    return `up · ${shortModel(hardware.ollama_model)}`;
  }
  return status;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="shrink-0 text-haze-400/70">{label}</dt>
      <dd className="truncate text-right text-haze-300">{value}</dd>
    </div>
  );
}

function shorten(name: string): string {
  return name.replace(/^NVIDIA\s+/i, "").replace(/\s+Laptop GPU$/i, "");
}

function shortModel(name?: string | null): string {
  if (!name) return "";
  const [fam, tag] = name.split(":");
  if (!tag) return fam;
  return `${fam}:${tag.split("-")[0]}`;
}
