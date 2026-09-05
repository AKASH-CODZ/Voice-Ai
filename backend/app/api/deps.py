"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request, WebSocket

# Header order matters: the first hop that is actually ours wins. Behind the
# Docker Compose network the app sees the proxy's address, so we honour
# X-Forwarded-For before falling back to the socket peer.
_FORWARD_HEADERS = ("x-forwarded-for", "x-real-ip")


def client_ip(request: Request) -> str:
    for header in _FORWARD_HEADERS:
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def ws_client_ip(websocket: WebSocket) -> str:
    for header in _FORWARD_HEADERS:
        value = websocket.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return websocket.client.host if websocket.client else "127.0.0.1"
