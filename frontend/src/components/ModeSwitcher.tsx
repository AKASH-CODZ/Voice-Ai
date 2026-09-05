"use client";

import type { Mode } from "@/lib/types";

/**
 * Mode switching is live: the backend rebuilds the system prompt on the next
 * turn and keeps the conversation history, so switching mid-session changes
 * behaviour without losing what was already said. The copy here states what
 * each mode *does to you*, since that is the only difference a user feels.
 */

const MODES: { id: Mode; label: string; blurb: string; accent: string; ring: string }[] = [
  {
    id: "casual",
    label: "Casual",
    blurb: "Natural chat. Mistakes are repaired through cross-questions, never called out.",
    accent: "text-casual",
    ring: "ring-casual/60 bg-casual/10",
  },
  {
    id: "teaching",
    label: "Teaching",
    blurb: "Active tutor. Corrects gently, and steps in when you go quiet.",
    accent: "text-teaching",
    ring: "ring-teaching/60 bg-teaching/10",
  },
  {
    id: "observation",
    label: "Observation",
    blurb: "Mock interview. Zero feedback until the end — you won't know how you did.",
    accent: "text-observation",
    ring: "ring-observation/60 bg-observation/10",
  },
];

interface ModeSwitcherProps {
  mode: Mode;
  onChange: (mode: Mode) => void;
  disabled?: boolean;
}

export function ModeSwitcher({ mode, onChange, disabled }: ModeSwitcherProps) {
  const active = MODES.find((m) => m.id === mode) ?? MODES[0];

  return (
    <div className="w-full">
      <div
        role="radiogroup"
        aria-label="Interaction mode"
        className="grid grid-cols-3 gap-1 rounded-xl border border-white/[0.06] bg-ink-800/70 p-1"
      >
        {MODES.map((m) => {
          const selected = m.id === mode;
          return (
            <button
              key={m.id}
              type="button"
              role="radio"
              aria-checked={selected}
              disabled={disabled}
              onClick={() => onChange(m.id)}
              className={[
                "rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40",
                "disabled:cursor-not-allowed disabled:opacity-40",
                selected
                  ? `ring-1 ${m.ring} ${m.accent}`
                  : "text-haze-400 hover:bg-white/[0.04] hover:text-haze-200",
              ].join(" ")}
            >
              {m.label}
            </button>
          );
        })}
      </div>
      <p className="mt-2 min-h-[2.5rem] px-1 text-[13px] leading-relaxed text-haze-400">
        {active.blurb}
      </p>
    </div>
  );
}
