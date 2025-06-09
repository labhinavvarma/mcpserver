import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession

# ✅ Replace with your actual MCP server's public IP or hostname
MCP_SSE_URL = "http://<EC2-IP>:8000/sse"

# ✅ Input to the MCP tool
email_input = {
    "subject": "Test Email from MCP Client",
    "body": "<p>This is a test email triggered using MCP streaming.</p>",
    "receivers": "your-email@example.com"
}

# ✅ Main client logic
async def send_test_email():
    async with sse_client(MCP_SSE_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            result = await session.invoke({
                "tool": "send_email",
                "input": email_input
            })
            print("📧 Tool Response:")
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(send_test_email())
