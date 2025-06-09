# router.py

from fastapi import APIRouter, Request, HTTPException
from mcpserver import mcp
from mcp.types import ToolOutput

route = APIRouter()

@route.post("/invoke", tags=["MCP"])
async def invoke_tool(request: Request):
    """
    Generic MCP tool tester endpoint.

    Input JSON:
    {
        "tool": "tool_name",
        "input": {
            ...tool_args...
        }
    }
    """
    try:
        payload = await request.json()
        tool_name = payload.get("tool")
        tool_input = payload.get("input", {})

        if not tool_name:
            raise HTTPException(status_code=400, detail="Missing 'tool' field")

        # Look up the tool in the registered tools list
        tool = next((t for t in mcp.tools if t.name == tool_name), None)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

        # Invoke the tool with the input
        output: ToolOutput = await tool.invoke(tool_input)

        return {
            "status": "success",
            "tool": tool_name,
            "output_type": output.type,
            "output_content": output.text
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
