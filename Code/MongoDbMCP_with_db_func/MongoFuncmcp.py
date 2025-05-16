import asyncio
import os
from pymongo import MongoClient
from mcp.server.fastmcp import FastMCP, Context
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.tools import Tool


load_dotenv()


mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["mcp_demo"]
collection = db["users"]


mcp = FastMCP("MongoDBService", port=8000)


@mcp.tool()
async def create_user(ctx: Context, name: str, email: str) -> str:
    """Add a new user."""
    result = collection.insert_one({"name": name, "email": email})
    return f"Added user, ID: {result.inserted_id}"

@mcp.tool()
async def read_user(ctx: Context, email: str) -> str:
    """Find a user by email."""
    user = collection.find_one({"email": email})
    return f"User: {user['name']}, Email: {user['email']}" if user else "User not found"

@mcp.tool()
async def update_user(ctx: Context, email: str, name: str) -> str:
    """Update user name by email."""
    result = collection.update_one({"email": email}, {"$set": {"name": name}})
    return "Updated user" if result.modified_count else "User not found"

@mcp.tool()
async def delete_user(ctx: Context, email: str) -> str:
    """Remove a user by email."""
    result = collection.delete_one({"email": email})
    return "Deleted user" if result.deleted_count else "User not found"


async def run_agent():
    llm = ChatOllama(model="llama3.1")
    tools = [
    Tool.from_function(
        func=create_user,
        name="create_user",
        description="Add a new user to the database using name and email."
    ),
    Tool.from_function(
        func=read_user,
        name="read_user",
        description="Read a user by email."
    ),
    Tool.from_function(
        func=update_user,
        name="update_user",
        description="Update a user's name using their email."
    ),
    Tool.from_function(
        func=delete_user,
        name="delete_user",
        description="Delete a user from the database using email."
    )
]


    llm_with_tools = llm.bind_tools(tools)
    prompt = PromptTemplate.from_template(
    """You are an assistant that can use tools to interact with a MongoDB database.
    Answer the user's query by reasoning through the steps and using the available tools.

    Tools available: {tool_names}

    {tools}

    User query: {input}

    {agent_scratchpad}"""
)

    

    agent = create_react_agent(
        llm=llm_with_tools,
        tools=tools,
        prompt=prompt
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


    print("🤖 MCP Agent ready! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("👋 Exiting.")
            break
        response = await agent_executor.ainvoke({"input": user_input})
        if not response['output']:
            print("\nAgent: I couldn't understand that. Could you please rephrase?")
            
        else:
            print(f"\nAgent: {response['output']}")



async def main():
    server_task = asyncio.create_task(mcp.run_sse_async())
    await asyncio.sleep(1) 
    await run_agent()
    server_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
