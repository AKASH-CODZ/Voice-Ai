/** @type {import('next').NextConfig} */
const backend = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

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
