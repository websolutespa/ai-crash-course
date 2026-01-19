from __future__ import annotations
import datetime as dt
from fastapi import FastAPI
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn

def utcnow() -> dt.datetime:
	return dt.datetime.now(dt.timezone.utc)

app = FastAPI(title="api probe", version="1.0.0")

@app.get("/health")
def health() -> dict[str, str]:
	"""Lightweight readiness probe, returns status and current timestamp."""
	return {"status": "ok", "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}

mcp = FastMCP(
	name="chat-mcp-api-probe",
	instructions=(
		"This MCP server provides a health check endpoint for monitoring purposes. "
		"Use it to verify the server's operational status and responsiveness."
	),
).from_fastapi(app)

# add cors for js clients, like MCP inspector
middlewares = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["mcp-session-id"],
    )
]
mcp_app = mcp.http_app(path="/mcp")

combined_app = FastAPI(
    title="Health API with MCP",
    routes=[
        *mcp_app.routes,  # MCP routes
        *app.routes,      # Original API routes
    ],
    lifespan=mcp_app.lifespan,
	middleware=middlewares,
)


if __name__ == "__main__":	
	import uvicorn
	uvicorn.run(
        combined_app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
