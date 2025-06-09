import requests

# Replace with your MCP server's actual IP or DNS
SERVER_URL = "http://<EC2-IP>:8000/invoke"

# Define the payload to trigger the send_email tool
payload = {
    "tool": "send_email",
    "input": {
        "subject": "📨 MCP Email Test",
        "body": "<p>This is a test email sent via MCP /invoke endpoint.</p>",
        "receivers": "your-email@example.com"  # <-- replace with a valid email
    }
}

def send_test_email():
    try:
        print(f"📡 Sending request to MCP server at {SERVER_URL}...")
        response = requests.post(SERVER_URL, json=payload)
        response.raise_for_status()
        print("✅ Email tool response:")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print("❌ Request failed:", e)

if __name__ == "__main__":
    send_test_email()
