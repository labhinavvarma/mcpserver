import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession

# Replace with your server's actual public IP or DNS
MCP_SSE_URL = "http://<EC2-IP>:8000/sse"

email_input = {
    "subject": "Test Email from MCP Client",
    "body": "<p>This is a test email sent from Python using MCP.</p>",
    "receivers": "your-email@example.com"
}

async def send_test_email():
    async with sse_client(MCP_SSE_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            result = await session.invoke_tool("send_email", **email_input)
            print("📧 Tool Response:")
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(send_test_email())
