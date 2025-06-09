import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport
from mcpserver import mcp  # Your FastMCP instance with tools/resources/prompts
from router import route   # Optional: your own FastAPI routes

# === Init FastAPI App ===
app = FastAPI(title="🧠 DataFlyWheel MCP Server")

# === Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔐 Allow all for dev — restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Set up the SSE transport manually
sse = SseServerTransport("/sse")

# ✅ Route 1: Documentation placeholder for /messages
@app.get("/messages", tags=["MCP"], include_in_schema=True)
def messages_docs(session_id: str):
    """
    Endpoint for posting messages to SSE clients.
    This is for documentation only — actual logic handled via SseServerTransport.
    """
    pass

# ✅ Route 2: Custom SSE handler — compatible with MCP client
@app.get("/sse", tags=["MCP"])
async def handle_sse(request: Request):
    """
    Establishes a streaming SSE connection to the MCP server.
    Compatible with sse_client in MCP Python SDK.
    """
    async with sse.connect_sse(request.scope, request.receive, request._send) as (
        read_stream,
        write_stream,
    ):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )

# ✅ Include your additional FastAPI tool routes
app.include_router(route)

# === Run server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
