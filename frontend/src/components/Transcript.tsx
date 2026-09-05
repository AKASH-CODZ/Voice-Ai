"use client";

import { useEffect, useRef } from "react";

import type { Mode, TranscriptTurn } from "@/lib/types";

/**
 * Live transcript.
 *
 * Two deliberate restraints:
 *
 * 1. **Observation mode shows no flags.** The hesitation and correction markers
 *    are suppressed entirely in that mode — surfacing them would defeat the
 *    whole point, which is that the user learns nothing until the session ends.
 * 2. **Auto-scroll yields to the user.** If they have scrolled up to re-read
 *    something, new turns must not yank them back down. We only follow the tail
 *    when they were already at the tail.
 */

interface TranscriptProps {
  turns: TranscriptTurn[];
  mode: Mode;
  agentSpeaking: boolean;
}

export function Transcript({ turns, mode, agentSpeaking }: TranscriptProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !pinnedRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [turns]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  const showFlags = mode !== "observation";

  return (
    <div
      ref={scrollRef}
      onScroll={onScroll}
      className="h-full overflow-y-auto scroll-smooth px-1"
      role="log"
      aria-live="polite"
      aria-label="Conversation transcript"
    >
      {turns.length === 0 ? (
        <div className="flex h-full items-center justify-center px-6 text-center">
          <p className="max-w-xs text-sm leading-relaxed text-haze-400/70">
            {agentSpeaking
              ? "…"
              : "Your conversation will appear here as it happens. Just start talking."}
          </p>
        </div>
      ) : (
        <ul className="space-y-3 py-2">
          {turns.map((turn) => (
            <li key={turn.id} className="animate-fade-up">
              <div
                className={[
                  "rounded-xl border px-3.5 py-2.5",
                  turn.role === "user"
                    ? "border-white/[0.05] bg-ink-700/50"
                    : "border-white/[0.07] bg-ink-600/40",
                ].join(" ")}
              >
                <div className="mb-1 flex items-center gap-2">
                  <span
                    className={[
                      "text-[10px] font-semibold uppercase tracking-[0.14em]",
                      turn.role === "user" ? "text-haze-400" : "text-haze-200",
                    ].join(" ")}
                  >
                    {turn.role === "user" ? "You" : "EchoSync"}
                  </span>

                  {showFlags && turn.hesitation && (
                    <span
                      title="A pause long enough to look like hesitation was recorded here."
                      className="rounded-full bg-observation/15 px-1.5 py-px text-[9px] font-medium uppercase tracking-wide text-observation"
                    >
                      paused
                    </span>
                  )}
                  {showFlags && turn.correctionMade && (
                    <span
                      title="EchoSync corrected something in this turn."
                      className="rounded-full bg-teaching/15 px-1.5 py-px text-[9px] font-medium uppercase tracking-wide text-teaching"
                    >
                      corrected
                    </span>
                  )}
                  {turn.streaming && (
                    <span className="ml-auto flex gap-0.5" aria-label="speaking">
                      {[0, 1, 2].map((i) => (
                        <span
                          key={i}
                          className="h-1 w-1 animate-pulse rounded-full bg-haze-400"
                          style={{ animationDelay: `${i * 140}ms` }}
                        />
                      ))}
                    </span>
                  )}
                </div>
                <p className="text-[14px] leading-relaxed text-haze-200">{turn.text}</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
