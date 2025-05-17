from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_ollama import ChatOllama
import json


mcp = FastMCP("MongoDB_MCP")
client = MongoClient("mongodb://localhost:27017/")
db = client["mcppracdb"]
collection = db["newcol"]


@mcp.tool()
def query_by_field(field_name: str, field_value: float) -> dict:
    result = collection.find_one({field_name: field_value})
    return result or {"message": "No matching document found."}

@mcp.tool()
def update_field(match_field: str, match_value: float, update_field: str, update_value: float) -> dict:
    result = collection.update_one(
        {match_field: match_value},
        {"$set": {update_field: update_value}}
    )
    return {"modified_count": result.modified_count}

@mcp.tool()
def delete_by_field(field_name: str, field_value: float) -> dict:
    result = collection.delete_one({field_name: field_value})
    return {"deleted_count": result.deleted_count}

@mcp.tool()
def create_index_on_field(field_name: str) -> dict:
    index_name = collection.create_index([(field_name, 1)])
    return {"index_created": index_name}

langchain_tools = [
    Tool(
        name="query_by_field",
        func=query_by_field,
        description="Query MongoDB collection by a field name and float value. Returns the matching document or a message if none found. Parameters: field_name (str), field_value (float)."
    ),
    Tool(
        name="update_field",
        func=update_field,
        description="Update a field in a MongoDB document. Matches by field name and float value, updates another field with a new float value. Returns modified count. Parameters: match_field (str), match_value (float), update_field (str), update_value (float)."
    ),
    Tool(
        name="delete_by_field",
        func=delete_by_field,
        description="Delete a MongoDB document by a field name and float value. Returns deleted count. Parameters: field_name (str), field_value (float)."
    ),
    Tool(
        name="create_index_on_field",
        func=create_index_on_field,
        description="Create an index on a specified field in the MongoDB collection. Returns the created index name. Parameters: field_name (str)."
    )
]


llm = ChatOllama(model="llama3.1", temperature=0.4)

react_prompt = PromptTemplate.from_template("""
You are a helpful assistant that can interact with a MongoDB database using the following tools: {tool_names}.
Answer the user's query by reasoning through the steps and calling the appropriate tool.
Return the final answer in a clear and concise manner.

Tools available: {tools}

User query: {input}

{agent_scratchpad}
""")


agent = create_react_agent(llm, langchain_tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=langchain_tools, verbose=True)


if __name__ == "__main__":



    query = "Find a revenue where the Temperature is 24.6 from my database monogdb , named mcppracdb"
    result = agent_executor.invoke({"input": query})
    print("Agent result:", json.dumps(result["output"], indent=2))
