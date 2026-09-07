/** Browser-facing API origin. Empty string → same-origin (Next rewrites). */

/**
 * Render `fromService.property: host` is the *internal* DNS name
 * (`echosync-api`), which browsers cannot resolve. A public web service
 * is `<name>.onrender.com`. Localhost and already-qualified hosts pass
 * through. Keep in sync with the copy in `next.config.mjs`.
 */
export function normalizeBackendOrigin(raw: string | undefined, fallback = ""): string {
  let value = (raw ?? "").trim().replace(/\/$/, "");
  if (!value) value = fallback;
  if (!value) return "";

  if (!value.includes("://")) {
    const hostname = value.split(":")[0] ?? value;
    if (
      !hostname.includes(".")
      && hostname !== "localhost"
      && !hostname.startsWith("127.")
    ) {
      value = `${hostname}.onrender.com`;
    }
  }

  if (value.startsWith("http://") || value.startsWith("https://")) {
    return value.replace(/\/$/, "");
  }
  const host = value.split("/")[0] ?? value;
  const scheme =
    host.startsWith("localhost") || host.startsWith("127.") ? "http" : "https";
  return `${scheme}://${value}`;
}

export function backendOrigin(): string {
  return normalizeBackendOrigin(process.env.NEXT_PUBLIC_BACKEND_URL);
}

export function apiUrl(path: string): string {
  const origin = backendOrigin();
  const p = path.startsWith("/") ? path : `/${path}`;
  return origin ? `${origin}${p}` : p;
}

export function voiceWsUrl(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  const origin = backendOrigin() || "http://localhost:8000";
  try {
    const u = new URL(origin);
    const scheme = u.protocol === "https:" ? "wss:" : "ws:";
    return `${scheme}//${u.host}/ws/voice`;
  } catch {
    return "ws://localhost:8000/ws/voice";
  }
}

export function backendDownMessage(): string {
  if (typeof window !== "undefined" && /\.onrender\.com$/.test(window.location.hostname)) {
    return "The voice API is waking from sleep (Render free tier, ~30s). This page retries automatically.";
  }
  return "Could not reach the voice backend. Start it with make backend.";
}
