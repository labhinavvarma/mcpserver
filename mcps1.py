from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import httpx
from urllib.parse import urlparse
from pathlib import Path
import json
import snowflake.connector
import requests  
import os
import uvicorn
from fastapi.responses import JSONResponse
from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import DatabaseError
from snowflake.core import Root
from typing import Optional, List
from ReduceReuseRecycleGENAI import get_ser_conn
from loguru import logger
from typing import Any, Dict, List
import numpy as np
from fastapi import (
 HTTPException,
 status,
)
 
from mcp.server.fastmcp.prompts.base import Message
from mcp.server.fastmcp import Context, FastMCP
import mcp.types as types
 
 
# Create a named server
 
mcp = FastMCP("DataFlyWheel App")
 
@dataclass
class AppContext:
    conn : SnowflakeConnection
    db: str
    schema: str
    host: str  
 
 
#@asynccontextmanager
#async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
#    """Manage application lifecycle with type-safe context"""
#    # Initialize on startup
#    HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
#    conn = snowflake.connector.connect(
#        user="AN666554AD",
#        password="Hyder@1234567890",
#        account="JIB90126.privatelink",
#        host=HOST,
#        port=443,
#        warehouse="DOC_AI_WH",
#        role="DOC_AI_BUSINESS_USER"
#    )
#    try:
#        yield AppContext(conn=conn,db="DOC_AI_DB",schema="HEDIS_SCHEMA",host=HOST)
#    finally:
#        # Cleanup on shutdown
#        conn.close()
 
 
# Pass lifespan to server
#mcp = FastMCP("DataFlyWheel App", lifespan=app_lifespan)
 
#Stag name may need to be determined; requires code change
#Resources; Have access to resources required for the server; Cortex Search; Cortex stage schematic config; stage area should be fully qualified name
@mcp.resource(uri="schematiclayer://cortex_analyst/schematic_models/{stagename}/list",name="hedis_schematic_models",description="Hedis Schematic models")
async def get_schematic_model(stagename: str):
    """Cortex analyst scematic layer model, model is in yaml format"""
    #ctx = mcp.get_context()
 
    HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
    conn = snowflake.connector.connect(
       user="AN611824AD",
       password="",
       account="JIB90126.privatelink",
       host=HOST,
       port=443,
       warehouse="DOC_AI_WH",
       role="DOC_AI_BUSINESS_USER",
       authenticator='externalbrowser',
    )
 
    #conn = ctx.request_context.lifespan_context.conn
    db = 'DOC_AI_DB'
    schema = 'HEDIS_SCHEMA'
    cursor = conn.cursor()
    snfw_model_list = cursor.execute("LIST @{db}.{schema}.{stagename}".format(db=db,schema=schema,stagename=stagename))
 
    return [stg_nm[0].split("/")[-1] for stg_nm in  snfw_model_list if stg_nm[0].endswith('yaml')]
   
@mcp.resource("search://cortex_search/search_obj/list")
async def get_search_service():
    """Cortex search service"""
 
    HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
    conn = snowflake.connector.connect(
       user="AN611824AD",
       password="",
       account="JIB90126.privatelink",
       host=HOST,
       port=443,
       warehouse="DOC_AI_WH",
       role="DOC_AI_BUSINESS_USER",
       authenticator='externalbrowser',
    )
 
    #conn = ctx.request_context.lifespan_context.conn
    db = 'DOC_AI_DB'
    schema = 'HEDIS_SCHEMA'
    cursor = conn.cursor()
    snfw_search_objs = cursor.execute("SHOW CORTEX SEARCH SERVICES IN SCHEMA {db}.{schema}".format(db=db,schema=schema))
    result = [search_obj[1] for search_obj in  snfw_search_objs.fetchall()]
    print(result)
    return result
     
@mcp.resource("genaiplatform://{aplctn_cd}/frequent_questions/{user_context}")
async def frequent_questions(aplctn_cd: str, user_context: str) -> List[str]:
    resource_name = aplctn_cd + "_freq_questions.json"
    freq_questions = json.load(open(resource_name))
    aplcn_question = freq_questions.get(aplctn_cd)
    return [rec["prompt"] for rec in aplcn_question if rec["user_context"] == user_context]
@mcp.resource("genaiplatform://{aplctn_cd}/prompts/{user_context}")
async def prompt_library(aplctn_cd: str, user_context: str) -> List[dict]:
    resource_name = aplctn_cd + "_prompts.json"
    prompt_data = json.load(open(resource_name))
    aplcn_prompts = prompt_data.get(aplctn_cd)
    return [rec for rec in aplcn_prompts if rec["user_context"] == user_context]
@mcp.tool(
        name="add-prompts",
        description="""
        Tool to add prompts to MCP server.
 
        Example inputs:
        {
           "uri": "genaiplatform://hedis/prompts/Initialization",
           "prompts": [
             {
               "name": "hedis-prompt",
               "description": "Prompt to explain HEDIS measures",
               "messages": [
                 {
                   "role": "user",
                   "content": "You are expert in HEDIS..."
                 }
               ]
             }
           ]
        }
 
        Args:
            uri (str): Prompt library URI with context
            prompts (list): List of prompt definitions
        """
)
async def add_prompts(ctx: Context, uri: str, prompts: list) -> list:
    # Parse and extract aplctn_cd and user_context (urllib)
    url_path = urlparse(uri)
    aplctn_cd = url_path.netloc
    user_context = Path(url_path.path).name
    file_data = []
    file_name = aplctn_cd + "_prompts.json"
 
    if Path(file_name).exists():
        file_data = json.load(open(file_name, 'r'))
        for prompt in prompts:
            prompt["user_context"] = user_context
            if prompt not in file_data[aplctn_cd]:
                file_data[aplctn_cd].append(prompt)
    else:
        file_data = {}
        file_data[aplctn_cd] = []
        for prompt in prompts:
            prompt["user_context"] = user_context
            file_data[aplctn_cd].append(prompt)
 
    file = open(file_name, 'w')
    file.write(json.dumps(file_data, indent=2))
    return file_data[aplctn_cd]
 
 
 
@mcp.tool(
        name="add-frequent-questions"
       ,description="""
        Tool to add frequent questions to MCP server
       
        Example inputs:
        {
           "uri"
        }  
 
        Args:
               uri (str):  text to be passed
               questions (list):
               [
                 {
                   "user_context" (str): "User context for the prompt"
                   "prompt" (str): "prompt"  
 
                 }
               ]
        """    
)
async def add_frequent_questions(ctx: Context,uri: str,questions: list) -> list:
    #Parse and extract aplctn_cd and user_context (urllib)
    url_path = urlparse(uri)
    aplctn_cd = url_path.netloc
    user_context = Path(url_path.path).name
    file_data = []
    file_name = aplctn_cd + "_freq_questions.json"
    if Path(file_name).exists():
        file_data  = json.load(open(file_name,'r'))
        file_data[aplctn_cd].extend(questions)
    else:
        file_data[aplctn_cd] =  questions
 
    file = open(file_name,'w')
    file.write(json.dumps(file_data))
    return file_data[aplctn_cd]
   
#Tools; corex Analyst; Cortex Search; Cortex Complete
 
@mcp.tool(
        name="DFWAnalyst"
       ,description="""
        Coneverts text to valid SQL which can be executed on HEDIS value sets and code sets.
       
        Example inputs:
           What are the codes in <some value> Value Set?
 
        Returns valid sql to retive data from underlying value sets and code sets.  
 
        Args:
               prompt (str):  text to be passed
 
        """
)
async def dfw_text2sql(prompt:str,ctx: Context) -> dict:
    """Tool to convert natural language text to snowflake sql for hedis system, text should be passed as 'prompt' input perameter"""
 
    HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
    conn = snowflake.connector.connect(
       user="AN611824AD",
       password="",
       account="JIB90126.privatelink",
       host=HOST,
       port=443,
       warehouse="DOC_AI_WH",
       role="DOC_AI_BUSINESS_USER",
       authenticator='externalbrowser',
    )
 
    #conn = ctx.request_context.lifespan_context.conn
    db = 'DOC_AI_DB'
    schema = 'HEDIS_SCHEMA'    
    host = HOST
    stage_name= "hedis_stage_full"
    file_name= "hedis_semantic_model_complete.yaml"
    request_body = {
        "messages":[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "semantic_model_file": f"@{db}.{schema}.{stage_name}/{file_name}",
    }
 
    token = conn.rest.token
    resp = requests.post(
        url=f"https://{host}/api/v2/cortex/analyst/message",
        json=request_body,
        headers={
            "Authorization": f'Snowflake Token="{token}"',
            "Content-Type": "application/json",
        },
    )
    return resp.json()
 
#Need to change the type of serch, implimented in the below code; Revisit
@mcp.tool(
        name="DFWSearch"
       ,description= """
        Searches HEDIS measure specification documents.
 
        Example inputs:
        What is the age criteria for  BCS Measure ?
        What is EED Measure in HEDIS?
        Describe COA Measure?
        What LOB is COA measure scoped under?
 
        Returns information utilizing HEDIS measure speficification documents.
 
        Args:
              query (str): text to be passed
       """
)
async def dfw_search(ctx: Context,query: str):
    """Tool to provide search againest HEDIS business documents for the year 2024, search string should be provided as 'query' perameter"""
 
    HOST = "carelon-eda-preprod.privatelink.snowflakecomputing.com"
    conn = snowflake.connector.connect(
       user="AN611824AD",
       password="",
       account="JIB90126.privatelink",
       host=HOST,
       port=443,
       warehouse="DOC_AI_WH",
       role="DOC_AI_BUSINESS_USER",
       authenticator='externalbrowser',
    )
 
    #conn = ctx.request_context.lifespan_context.conn
    db = 'DOC_AI_DB'
    schema = 'HEDIS_SCHEMA'    
    search_service = 'CS_HEDIS_FULL_2024'
    columns = ['chunk']
    limit = 2
 
    root = Root(conn)
    search_service = root.databases[db].schemas[schema].cortex_search_services[search_service]
    response = search_service.search(
        query=query,
        columns=columns,
        limit = limit
    )
    return response.to_json()

@mcp.tool()
def send_email(subject: str, body: str, receivers: str) -> dict:
    """
    Sends an email to the specified recipients.

    Args:
        subject (str): Subject of the email.
        body (str): HTML content of the email.
        receivers (str): Comma-separated list of recipient email addresses.

    Returns:
        dict: Status message indicating success or failure.
    """
    try:
        sender = 'noreply-vkhvkm'
        recipients = [email.strip() for email in receivers.split(",")]

        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        msg.attach(MIMEText(body, 'html'))

        # Establish SMTP connection using your custom function
        smtpObj = get_ser_conn(
            logger,
            env="preprod",
            region_name="us-east-1",
            aplctn_cd="aedl",
            port=None,
            tls=True,
            debug=False
        )
        smtpObj.sendmail(sender, recipients, msg.as_string())
        smtpObj.quit()
        logger.info("Email sent successfully.")
        return {"status": "success", "message": "Email sent successfully."}
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return {"status": "error", "message": f"Error sending email: {e}"}

 
@mcp.tool(
        name="calculator",
        description="""
        Evaluates a basic arithmetic expression.
        Supports: +, -, *, /, parentheses, decimals.
 
        Example inputs:
        3+4/5
        3.0/6*8
 
        Returns decimal result
 
        Args:
             expression (str): Arthamatic expression input
         
        """
)
def calculate(expression: str) -> str:
    """
    Evaluates a basic arithmetic expression.
    Supports: +, -, *, /, parentheses, decimals.
    """
    print(f" calculate() called with expression: {expression}", flush=True)
    try:
        allowed_chars = "0123456789+-*/(). "
        if any(char not in allowed_chars for char in expression):
            return " Invalid characters in expression."
 
        result = eval(expression)
        return f" Result: {result}"
    except Exception as e:
        print(" Error in calculate:", str(e), flush=True)
        return f" Error: {str(e)}"
 
@mcp.tool(
        name="suggested_top_prompts",
        description="""
        Suggests requested number of prompts with given context.
 
        Example Input:
        {
          top_n_suggestions: 3,
          context: "Initialization" | "The age criteria for the BCS (Breast Cancer Screening) measure is 50-74 years of age."
          aplctn_cd: "hedis"
        }
 
        Returns List of string values.
 
        Args:
            top_n_suggestions (int): how many suggestions to be generated.
            context (str): context that need to be used for the promt suggestions.
            aplctn_cd (str): application code.
        """
)
async def question_suggestions(ctx: Context,aplctn_cd: str, app_lvl_prefix: str, session_id: str, top_n: int = 3,context: str="Initialization",llm_flg: bool = False):
    """Tool to suggest aditional prompts with in the provided context, context should be passed as 'context' input perameter"""
   
    if  not llm_flg:
        return ctx.read_resource(f"genaiplatform://{aplctn_cd}/frequent_questions/{context}")
   
    try:
        sf_conn = SnowFlakeConnector.get_conn(
            aplctn_cd,
            app_lvl_prefix,
            session_id,
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not authorized to resources"
        )
    clnt = httpx.AsyncClient(verify=False)
           
    request_body = {
        "model": "llama3.1-70b-elevance",
        "messages": [
            {
                "role": "user",
                "content": f"""
                You are an expert in suggesting hypothetical questions.
                Suggest a list of {top_n} hypothetical questions that the below context could be used to answer:
               
                {context}
                Return List with hypothetical questions
                """
            }
        ]
    }
 
    headers = {
        "Authorization": f'Snowflake Token="{sf_conn.rest.token}"',
        "Content-Type": "application/json",
        "Accept": "application/json",
        "method":"cortex",
    }
 
    url = "https://jib90126.us-east-1.privatelink.snowflakecomputing.com/api/v2/cortex/inference:complete"
 
    response_text = []      
   
    async with clnt.stream('POST', url, headers=headers, json=request_body) as response:
        if response.is_client_error:
            error_message = await response.aread()
            raise HTTPException(
                status_code=response.status_code,
                detail=error_message.decode("utf-8")
            )
        if response.is_server_error:
            error_message = await response.aread()
            raise HTTPException(
                status_code=response.status_code,
                detail=error_message.decode("utf-8")
            )
        # Stream the response content
        async for result_chunk in response.aiter_bytes():
            for elem in result_chunk.split(b'\n\n'):
                if b'content' in elem:  # Check for data presence
                   
                    chunk_dict = json.loads(elem.replace(b'data: ', b''))
                    print(chunk_dict)
                    full_response = chunk_dict['choices'][0]['delta']['text']
                    full_response = full_response
                    response_text.append(full_response)
                       
    return json.loads(response_text)
   
def extract_numeric_values(data: Any) -> Dict[str, float]:
    result = {}

    def process_value(value: Any, path: str = ""):
        if isinstance(value, dict):
            for k, v in value.items():
                new_path = f"{path}.{k}" if path else k
                process_value(v, new_path)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                new_path = f"{path}[{i}]"
                process_value(v, new_path)
        else:
            try:
                if isinstance(value, str):
                    value = value.replace(',', '')
                num_value = float(value)
                result[path] = num_value
            except (ValueError, TypeError):
                pass

    process_value(data)
    return result


@mcp.tool(
    name="analyze-data",
    description="Analyzes a JSON dataset and performs statistical operations on a specified column."
)
def analyze_data(data: List[Dict], column: str, operation: str) -> Dict[str, Any]:
    try:
        numeric_data = []
        for item in data:
            numeric_values = extract_numeric_values(item)
            if column in numeric_values:
                numeric_data.append(numeric_values[column])

        if not numeric_data:
            raise ValueError(f"No numeric values found for column '{column}'")

        arr = np.array(numeric_data)

        result_fn = {
            "sum": np.sum,
            "mean": np.mean,
            "average": np.mean,
            "median": np.median,
            "min": np.min,
            "max": np.max,
            "count": lambda x: len(x)
        }.get(operation.lower())

        if result_fn is None:
            raise ValueError(f"Unsupported operation: {operation}")

        return {"status": "success", "value": float(result_fn(arr))}
    except Exception as e:
        return {"status": "error", "error": str(e)}
@mcp.prompt(
        name="hedis-prompt",
        description="Hedis prompt to query HEDIS measures for measurement year, or identify the changes between two different measurement years"
)
async def hedis_prompt(query: str)-> List[Message]:
    return [
        {
            "role": "user",
            "content": f""" You are expert in HEDIS system, HEDIS is a set of standardized measures that aim to improve healthcare quality by promoting accountability and transparency.You are provided with below tools:1) DFWAnalyst - Generates SQL to retrive information for hedis codes and value sets.2) DFWSearch -  Provides search capability againest HEDIS measures for measurement year.You will respond with the results returned from right tool.{query}"""
        }
    ]
 
@mcp.prompt(
        name="caleculator-promt",
        description="Prompt to perform calculations and verify the results"
)
async def caleculator_prompt()-> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are expert in performing arthametic operations.You are provided with the tool calculator to verify the results.You will respond with the results after verifying with the tool result."""
        }
    ]
@mcp.prompt(
    name="mcp-prompt-analyze-data",
    description="Prompt to analyze numeric columns in JSON and compute summary statistics."
)
async def mcp_prompt_analyze_data(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are a data analysis assistant.You are provided with the tool `analyze_data` which performs statistical analysis on a list of JSON objects. Your job is to:1. Identify the column (even nested keys in dot-path format) that contains numeric values.2. Choose one of the operations: sum, mean, average, median, min, max, count.3. Call the tool with three inputs:- data: the JSON list- column: the target column to analyze- operation: the operation to apply.Return only the tool output result with status.{query}"""
        }
    ]

@mcp.prompt(
    name="mcp-prompt-send-email",
    description="Prompt to guide the LLM to send an email using the 'mcp-send-email' tool."
)
async def mcp_prompt_send_email(query: str) -> List[Message]:
    return [
        {
            "role": "user",
            "content": f"""You are an email assistant.You are provided with the tool `mcp-send-email` to send emails to users. Your job is to:1. Extract the recipient's email address.2. Extract the subject of the email.3. Extract the message body.4. Use the tool with these three arguments:- to: recipient's email address- subject: subject line of the email- body: the message content.You must return only the tool result or any error it returns.{query}"""
        }
    ]

 
if __name__ == "__main__":
    mcp.run(transport="sse")
 
