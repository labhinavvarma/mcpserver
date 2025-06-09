import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession
import json

# ✅ Replace this with your actual EC2 public IP or DNS
MCP_SSE_URL = "http://<EC2-IP>:8000/sse"  # e.g., http://54.123.45.67:8000/sse

# ✅ Define your email input (MUST match server's `send_email` args)
email_input = {
    "subject": "📨 Test Email from MCP Client",
    "body": "<p>Hello, this is a test email sent via MCP server.</p>",
    "receivers": "your-email@example.com"  # Replace with a real test address
}

# ✅ Main Async Function
async def send_test_email():
    async with sse_client(MCP_SSE_URL) as conn:
        print(f"✅ Connected to MCP SSE server: {conn.session_id}")
        async with ClientSession(*conn) as session:
            result = await session.invoke_tool("send_email", **email_input)
            print("📧 MCP Tool Response:")
            print(json.dumps(result, indent=2))

# ✅ Run the script
if __name__ == "__main__":
    asyncio.run(send_test_email())
