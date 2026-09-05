"use client";

import { useEffect, useRef } from "react";

import type { Mode } from "@/lib/types";

/**
 * The reactive orb.
 *
 * Drawn on a canvas rather than with CSS transforms because it animates every
 * frame off live audio levels: driving CSS custom properties at 60 fps forces
 * style recalculation on the main thread, which is the same thread the
 * WebSocket and React reconciliation share. A canvas keeps all of that off the
 * critical path — and this element is animating during the exact moments
 * latency matters most.
 *
 * The visual grammar carries state without any text:
 *   • hue      → active mode
 *   • radius   → live loudness (user's voice when listening, agent's when speaking)
 *   • rings    → who holds the floor: inward when listening, outward when speaking
 *   • unrest   → the surface roughens as it gets louder
 */

const MODE_COLORS: Record<Mode, [number, number, number]> = {
  casual: [76, 201, 240],
  teaching: [139, 92, 246],
  observation: [245, 158, 11],
};

interface Ring { radius: number; alpha: number; width: number }

interface VoiceOrbProps {
  mode: Mode;
  inputLevel: number;
  outputLevel: number;
  userSpeaking: boolean;
  agentSpeaking: boolean;
  connected: boolean;
  stalled: boolean;
}

export function VoiceOrb({
  mode, inputLevel, outputLevel, userSpeaking, agentSpeaking, connected, stalled,
}: VoiceOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Live values are read inside the animation loop, so they go in a ref: putting
  // them in the effect's dependency list would tear down and rebuild the whole
  // rAF loop on every audio frame.
  const propsRef = useRef({ mode, inputLevel, outputLevel, userSpeaking, agentSpeaking, connected, stalled });
  propsRef.current = { mode, inputLevel, outputLevel, userSpeaking, agentSpeaking, connected, stalled };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frame = 0;
    let raf = 0;
    let smoothed = 0;
    let hue: [number, number, number] = MODE_COLORS[mode];
    const rings: Ring[] = [];

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const size = canvas.clientWidth;
      canvas.width = size * dpr;
      canvas.height = size * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    const draw = () => {
      const p = propsRef.current;
      const size = canvas.clientWidth;
      const cx = size / 2;
      const cy = size / 2;
      const base = size * 0.19;

      // Ease the colour so a mode switch glides instead of snapping.
      const target = MODE_COLORS[p.mode];
      hue = hue.map((c, i) => c + (target[i] - c) * 0.08) as [number, number, number];
      const [r, g, b] = hue.map(Math.round);

      // Whoever holds the floor drives the amplitude.
      const raw = p.agentSpeaking ? p.outputLevel * 1.5 : p.inputLevel * 2.2;
      smoothed += (Math.min(raw, 1) - smoothed) * 0.16;

      const idle = 0.5 + 0.5 * Math.sin(frame * 0.018);
      const energy = p.connected ? smoothed : 0;
      const radius = base * (1 + energy * 0.5 + (p.connected ? idle * 0.03 : 0));

      ctx.clearRect(0, 0, size, size);

      // ── outer glow ──
      const glow = ctx.createRadialGradient(cx, cy, radius * 0.4, cx, cy, radius * 3.1);
      glow.addColorStop(0, `rgba(${r},${g},${b},${0.20 + energy * 0.30})`);
      glow.addColorStop(0.45, `rgba(${r},${g},${b},${0.05 + energy * 0.10})`);
      glow.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, size, size);

      // ── travelling rings ──
      if (frame % 26 === 0 && p.connected && (p.userSpeaking || p.agentSpeaking)) {
        rings.push({ radius: p.agentSpeaking ? radius : radius * 2.4, alpha: 0.5, width: 1.6 });
      }
      for (let i = rings.length - 1; i >= 0; i -= 1) {
        const ring = rings[i];
        // Direction encodes who is talking: the agent radiates outward, the
        // user's voice is drawn inward toward the orb.
        ring.radius += p.agentSpeaking ? 1.5 : -1.4;
        ring.alpha *= 0.975;
        if (ring.alpha < 0.02 || ring.radius < radius * 0.9 || ring.radius > size * 0.55) {
          rings.splice(i, 1);
          continue;
        }
        ctx.beginPath();
        ctx.arc(cx, cy, ring.radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${r},${g},${b},${ring.alpha})`;
        ctx.lineWidth = ring.width;
        ctx.stroke();
      }

      // ── the orb body: a wobbling blob, rougher when louder ──
      ctx.beginPath();
      const points = 96;
      const unrest = energy * 0.16;
      for (let i = 0; i <= points; i += 1) {
        const angle = (i / points) * Math.PI * 2;
        const wobble =
          Math.sin(angle * 3 + frame * 0.045) * unrest +
          Math.sin(angle * 5 - frame * 0.032) * unrest * 0.55 +
          Math.sin(angle * 2 + frame * 0.021) * 0.012;
        const rr = radius * (1 + wobble);
        const x = cx + Math.cos(angle) * rr;
        const y = cy + Math.sin(angle) * rr;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();

      const body = ctx.createRadialGradient(
        cx - radius * 0.3, cy - radius * 0.35, radius * 0.1, cx, cy, radius * 1.15,
      );
      body.addColorStop(0, `rgba(255,255,255,${0.85 + energy * 0.15})`);
      body.addColorStop(0.35, `rgba(${r},${g},${b},0.95)`);
      body.addColorStop(1, `rgba(${Math.round(r * 0.35)},${Math.round(g * 0.35)},${Math.round(b * 0.45)},0.92)`);
      ctx.fillStyle = body;
      ctx.fill();

      // ── stall halo: dashed, slowly rotating, unmistakably different ──
      if (p.stalled) {
        ctx.beginPath();
        ctx.setLineDash([5, 9]);
        ctx.lineDashOffset = -frame * 0.35;
        ctx.arc(cx, cy, radius * 1.55, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(245,158,11,0.65)";
        ctx.lineWidth = 1.6;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // ── offline: a thin dormant outline ──
      if (!p.connected) {
        ctx.beginPath();
        ctx.arc(cx, cy, base * 1.5, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(139,147,167,0.18)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      frame += 1;
      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
    // `mode` seeds the initial colour only; live updates come through the ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="relative aspect-square w-full max-w-[340px]">
      <canvas ref={canvasRef} className="h-full w-full" aria-hidden="true" />
      <span className="sr-only">
        {!connected ? "Not connected"
          : agentSpeaking ? "EchoSync is speaking"
          : userSpeaking ? "Listening to you"
          : stalled ? "Waiting — you seem stuck"
          : "Listening"}
      </span>
    </div>
  );
}
