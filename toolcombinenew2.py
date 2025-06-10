import streamlit as st
import asyncio
import json
import yaml

from mcp.client.sse import sse_client
from mcp import ClientSession
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dependencies import SnowFlakeConnector
from llmobject_wrapper import ChatSnowflakeCortex
from snowflake.snowpark import Session

# Page configuration
st.set_page_config(page_title="Healthcare AI Chat", page_icon="🏥")
st.title("Healthcare AI Chat")

# MCP Server URL
server_url = st.sidebar.text_input("MCP Server URL", "http://10.126.192.183:8001/sse")
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)

# === MOCK LLM (for testing if needed) ===
def mock_llm_response(prompt_text: str) -> str:
    return f"🤖 Mock LLM Response to: '{prompt_text}'"

# --- MCP Server Info Section ---
if show_server_info:
    async def fetch_mcp_info():
        result = {"resources": [], "tools": [], "prompts": [], "yaml": [], "search": []}
        try:
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()

                    # --- Resources ---
                    resources = await session.list_resources()
                    if hasattr(resources, 'resources'):
                        for r in resources.resources:
                            result["resources"].append({"name": r.name})
                   
                    # --- Tools (filtered) ---
                    tools = await session.list_tools()
                    hidden_tools = {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}
                    if hasattr(tools, 'tools'):
                        for t in tools.tools:
                            if t.name not in hidden_tools:
                                result["tools"].append({"name": t.name})

                    # --- Prompts ---
                    prompts = await session.list_prompts()
                    if hasattr(prompts, 'prompts'):
                        for p in prompts.prompts:
                            args = []
                            if hasattr(p, 'arguments'):
                                for arg in p.arguments:
                                    args.append(f"{arg.name} ({'Required' if arg.required else 'Optional'}): {arg.description}")
                            result["prompts"].append({
                                "name": p.name,
                                "description": getattr(p, 'description', ''),
                                "args": args
                            })

                    # --- YAML Resources ---
                    try:
                        yaml_content = await session.read_resource("schematiclayer://cortex_analyst/schematic_models/hedis_stage_full/list")
                        if hasattr(yaml_content, 'contents'):
                            for item in yaml_content.contents:
                                if hasattr(item, 'text'):
                                    parsed = yaml.safe_load(item.text)
                                    result["yaml"].append(yaml.dump(parsed, sort_keys=False))
                    except Exception as e:
                        result["yaml"].append(f"YAML error: {e}")

                    # --- Search Objects ---
                    try:
                        content = await session.read_resource("search://cortex_search/search_obj/list")
                        if hasattr(content, 'contents'):
                            for item in content.contents:
                                if hasattr(item, 'text'):
                                    objs = json.loads(item.text)
                                    result["search"].extend(objs)
                    except Exception as e:
                        result["search"].append(f"Search error: {e}")

        except Exception as e:
            st.sidebar.error(f"❌ MCP Connection Error: {e}")
        return result

    mcp_data = asyncio.run(fetch_mcp_info())

    # Display Resources
    with st.sidebar.expander("📦 Resources", expanded=False):
        for r in mcp_data["resources"]:
            if "cortex_search/search_obj/list" in r["name"]:
                display_name = "Cortex Search"
            else:
                display_name = r["name"]
            st.markdown(f"**{display_name}**")
    # --- YAML Section ---
    with st.sidebar.expander("Schematic Layer", expanded=False):
        for y in mcp_data["yaml"]:
            st.code(y, language="yaml")
    # --- Tools Section ---
    with st.sidebar.expander("🛠 Tools", expanded=False):
        for t in mcp_data["tools"]:
            st.markdown(f"**{t['name']}**")
    # --- Prompts Section ---
    with st.sidebar.expander("🧐 Prompts", expanded=False):
        for p in mcp_data["prompts"]:
            st.markdown(f"**{p['name']}**")
else:
    # Re-enable Snowflake and LLM chatbot features
    @st.cache_resource
    def get_snowflake_connection():
        return SnowFlakeConnector.get_conn('aedl', '')

    @st.cache_resource
    def get_model():
        sf_conn = get_snowflake_connection()
        return ChatSnowflakeCortex(
            model="llama3.1-70b-elevance",
            cortex_function="complete",
            session=Session.builder.configs({"connection": sf_conn}).getOrCreate()
        )

    # Update prompt type options: adding Email and Data Analysis
    prompt_type = st.sidebar.radio("Select Prompt Type", 
                                   ["Calculator", "HEDIS Expert", "Weather", "Email", "Data Analysis", "No Context"])

    prompt_map = {
        "Calculator": "calculator-prompt",
        "HEDIS Expert": "hedis-prompt",
        "Weather": "weather-prompt",
        "Email": "mcp-prompt-send-email",
        "Data Analysis": "mcp-prompt-analyze-data",
        "No Context": None
    }

    examples = {
        "Calculator": ["(4+5)/2.0", "3*7-2", "sqrt(16) + 7"],
        "HEDIS Expert": ["What are the HEDIS codes for diabetes management?"],
        "Weather": [
            "What is the weather in Richmond?",
            "Forecast for Atlanta?",
            "Is it raining in NYC?"
        ],
        "Email": [
            "Send an email with Subject: 'Meeting Reminder', Body: 'Please attend the meeting at 3 PM', and Recipients: john@example.com, jane@example.com."
        ],
        "Data Analysis": [
            "Analyze the JSON data for total sales in the 'amount' column.",
            "Compute the average value for the 'price' field."
        ],
        "No Context": ["Who won the world cup in 2022?", "Summarize recent climate change news."]
    }

    if prompt_type == "HEDIS Expert":
        try:
            async def fetch_hedis_examples():
                async with sse_client(url=server_url) as sse_connection:
                    async with ClientSession(*sse_connection) as session:
                        await session.initialize()
                        content = await session.read_resource("genaiplatform://hedis/frequent_questions/Initialization")
                        if hasattr(content, "contents"):
                            for item in content.contents:
                                if hasattr(item, "text"):
                                    examples["HEDIS Expert"].extend(json.loads(item.text))
            asyncio.run(fetch_hedis_examples())
        except Exception as e:
            examples["HEDIS Expert"] = [f"⚠️ Failed to load examples: {e}"]

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    with st.sidebar.expander("Example Queries", expanded=True):
        for example in examples[prompt_type]:
            if st.button(example, key=example):
                st.session_state.query_input = example

    query = st.chat_input("Type your query here...")
    if "query_input" in st.session_state:
        query = st.session_state.query_input
        del st.session_state.query_input

    async def process_query(query_text):
        st.session_state.messages.append({"role": "user", "content": query_text})
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("Processing...")
            try:
                async with MultiServerMCPClient(
                    {"DataFlyWheelServer": {"url": server_url, "transport": "sse"}}
                ) as client:
                    # Get the model and create the agent with server tools
                    model = get_model()
                    agent = create_react_agent(model=model, tools=client.get_tools())
                    
                    # Retrieve the prompt from the MCP server based on the chosen type
                    prompt_name = prompt_map[prompt_type]
                    prompt_from_server = await client.get_prompt(
                        server_name="DataFlyWheelServer",
                        prompt_name=prompt_name,
                        arguments={}
                    )
                    if prompt_from_server and "{query}" in prompt_from_server[0].content:
                        formatted_prompt = prompt_from_server[0].content.format(query=query_text)
                    elif prompt_from_server:
                        formatted_prompt = prompt_from_server[0].content + query_text
                    else:
                        formatted_prompt = query_text
                    
                    # Invoke the agent with the formatted prompt
                    response = await agent.ainvoke({"messages": formatted_prompt})
                    # Extract result (adjusting based on your agent's return format)
                    result = list(response.values())[0][1].content
                    message_placeholder.text(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.text(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

    if query:
        asyncio.run(process_query(query))

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("Healthcare AI Chat v1.0")
