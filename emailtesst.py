import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession
from httpx import ConnectError, ReadTimeout

# === MCP EC2 Server Config ===
MCP_SSE_URL = "http://<EC2-IP>:8000/messages"  # Replace with your actual server URL

# === Email Parameters ===
EMAIL_SUBJECT = "📨 MCP Email Tool Test"
EMAIL_BODY = "<p>This is a test email sent from the MCP client directly.</p>"
EMAIL_RECEIVERS = "your-email@example.com"  # Replace with your test email

# === MCP Tool Name
TOOL_NAME = "send_email"

# === Utility: Check MCP Server Connection
async def check_mcp_connection():
    try:
        async with sse_client(MCP_SSE_URL) as conn:
            print("✅ Successfully connected to MCP server")
            return True
    except (ConnectError, ReadTimeout, Exception) as e:
        print(f"❌ Failed to connect to MCP server: {e}")
        return False

# === Run Email Tool
async def test_send_email():
    if not await check_mcp_connection():
        return

    async with sse_client(MCP_SSE_URL) as conn:
        async with ClientSession(*conn) as session:
            tool_input = {
                "subject": EMAIL_SUBJECT,
                "body": EMAIL_BODY,
                "receivers": EMAIL_RECEIVERS
            }

            try:
                result = await session.invoke_tool(TOOL_NAME, **tool_input)
                print("📧 Tool Invocation Result:")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"❌ Error invoking tool: {e}")

# === Entry Point
if __name__ == "__main__":
    asyncio.run(test_send_email())
