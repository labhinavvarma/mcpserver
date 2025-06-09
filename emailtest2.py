import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession
from httpx import ConnectError

# === Update to use /sse because /messages does not serve full SSE
MCP_SSE_URL = "http://<EC2-IP>:8000/sse"  # Replace with your EC2 IP

# === Email Payload
email_payload = {
    "subject": "📨 Test Email via MCP Tool",
    "body": "<p>This is a test email from client using /sse route.</p>",
    "receivers": "you@example.com"
}

async def run_test():
    try:
        async with sse_client(MCP_SSE_URL) as conn:
            print(f"✅ Connected to server. Session ID: {conn.session_id}")
            async with ClientSession(*conn) as session:
                result = await session.invoke_tool("send_email", **email_payload)
                print("📧 Tool Invocation Result:")
                print(json.dumps(result, indent=2))
    except ConnectError as ce:
        print(f"❌ Connection failed: {ce}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
