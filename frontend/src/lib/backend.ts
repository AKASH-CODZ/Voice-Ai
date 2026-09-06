/** Browser-facing API origin. Empty string → same-origin (Next rewrites). */
export function backendOrigin(): string {
  const raw = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (!raw) return "";
  if (raw.startsWith("http://") || raw.startsWith("https://")) {
    return raw.replace(/\/$/, "");
  }
  return `https://${raw.replace(/\/$/, "")}`;
}

export function apiUrl(path: string): string {
  const origin = backendOrigin();
  const p = path.startsWith("/") ? path : `/${path}`;
  return origin ? `${origin}${p}` : p;
}
