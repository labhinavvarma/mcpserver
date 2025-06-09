import traceback

@route.post("/invoke", tags=["MCP"])
async def invoke_tool(request: Request):
    try:
        payload = await request.json()
        tool_name = payload.get("tool")
        tool_input = payload.get("input", {})

        if not tool_name:
            raise HTTPException(status_code=400, detail="Missing 'tool' field")

        tool = next((t for t in mcp.tools if t.name == tool_name), None)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

        output = await tool.invoke(tool_input)

        return {
            "status": "success",
            "tool": tool_name,
            "output_type": getattr(output, "type", "text"),
            "output_content": getattr(output, "text", str(output))
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()  # 👈 Print full error trace to console
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
