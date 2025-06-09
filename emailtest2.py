import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession

MCP_SSE_URL = "http://<EC2-IP>:8000/sse"

email_input = {
    "subject": "📨 Test Email via MCP",
    "body": "<p>This email was triggered from a Python client.</p>",
    "receivers": "your-email@example.com"
}

async def send_test_email():
    async with sse_client(MCP_SSE_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            result = await session.invoke({
                "tool": "send_email",
                "input": email_input
            })
            print("✅ Email Sent Tool Response:")
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(send_test_email())
