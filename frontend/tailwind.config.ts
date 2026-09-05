import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // A near-black ground with a faint blue cast reads as "instrument"
        // rather than "empty page", which is the Hume EVI trick.
        ink: { 900: "#07080c", 800: "#0c0e14", 700: "#12151d", 600: "#1a1e28" },
        haze: { 400: "#8b93a7", 300: "#a8b0c2", 200: "#c9cfdc" },
        casual: "#4cc9f0",
        teaching: "#8b5cf6",
        observation: "#f59e0b",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      animation: {
        "fade-up": "fadeUp 320ms cubic-bezier(0.16,1,0.3,1) both",
        "pulse-ring": "pulseRing 2.4s cubic-bezier(0.4,0,0.6,1) infinite",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        pulseRing: {
          "0%,100%": { opacity: "0.35", transform: "scale(1)" },
          "50%": { opacity: "0.05", transform: "scale(1.18)" },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
