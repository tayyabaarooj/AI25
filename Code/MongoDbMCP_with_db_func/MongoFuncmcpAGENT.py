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
db = mongo_client["mcppracdb"]
collection = db["newcol"]


mcp = FastMCP("MongoDBService", port=8000)


@mcp.tool()
async def create_document(ctx: Context, collection_name: str, document: dict) -> str:
    """Add a document to a specified collection."""
    collection = db[collection_name]
    result = collection.insert_one(document)
    return f"Document added with ID: {result.inserted_id}"

@mcp.tool()
async def read_documents(ctx: Context, collection_name: str, filters: dict = {}) -> str:
    """Read documents from a specified collection with filters."""
    collection = db[collection_name]
    documents = list(collection.find(filters))
    return str(documents) if documents else "No documents found."

@mcp.tool()
async def update_documents(ctx: Context, collection_name: str, filters: dict, updates: dict) -> str:
    """Update documents in a specified collection based on filters."""
    collection = db[collection_name]
    result = collection.update_many(filters, {"$set": updates})
    return f"Updated {result.modified_count} document(s)" if result.modified_count else "No documents matched the criteria."

@mcp.tool()
async def delete_documents(ctx: Context, collection_name: str, filters: dict) -> str:
    """Delete documents from a specified collection based on filters."""
    collection = db[collection_name]
    result = collection.delete_many(filters)
    return f"Deleted {result.deleted_count} document(s)" if result.deleted_count else "No documents matched the criteria."

@mcp.tool()
async def list_all_collections(ctx: Context) -> str:
    """List all collections in the database."""
    collections = db.list_collection_names()
    return f"Collections: {', '.join(collections)}" if collections else "No collections found."

async def run_agent():
    llm = ChatOllama(model="llama3.1")
    tools = [
    Tool.from_function(
        func=create_document,
        name="create_doc",
        description="Add a new document to the database "
    ),
    Tool.from_function(
        func=read_documents,
        name="read_doc",
        description="Read a document in database"
    ),
    Tool.from_function(
        func=delete_documents,
        name="update_doc",
        description="Update a document iin database"
    ),
    Tool.from_function(
        func=delete_documents,
        name="delete_doc",
        description="Delete a document from the database."
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
