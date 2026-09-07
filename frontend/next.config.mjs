/** @type {import('next').NextConfig} */
// Keep in sync with src/lib/backend.ts `normalizeBackendOrigin`.
function backendOrigin() {
  let value = (process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000")
    .trim()
    .replace(/\/$/, "");
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
  if (value.startsWith("http://") || value.startsWith("https://")) return value;
  const host = value.split("/")[0] ?? value;
  const scheme =
    host.startsWith("localhost") || host.startsWith("127.") ? "http" : "https";
  return `${scheme}://${value}`;
}
const backend = backendOrigin();

const nextConfig = {
  reactStrictMode: true,
  // Emit a traced standalone server so the Docker runner stage needs no
  // node_modules — ~150 MB instead of ~1.2 GB.
  output: "standalone",
  // Proxy the API through Next in development so the browser sees one origin.
  // This sidesteps CORS entirely and, more importantly, means `getUserMedia`
  // runs on the same secure context as the socket.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
