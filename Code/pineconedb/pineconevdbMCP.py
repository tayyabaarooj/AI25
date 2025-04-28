import os
from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool as LangChainTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



PINECONE_INDEX_NAME = "langchainvecdb2"
os.environ["OPENAI_API_KEY"] = "your_key_here"

OPENAI_API_KEY= os.getenv("your_Key_here")
import getpass
import os

from pinecone import Pinecone, ServerlessSpec

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(PINECONE_INDEX_NAME)


# Set OpenAI key
openai.api_key = OPENAI_API_KEY
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Pinecone(
    index=index,
    embedding=embedding_model,
    text_key="text",  # adjust if your metadata key is different
)
mcp = FastMCP("pcprac")
@mcp.tool()
@mcp.tool("semantic_search")
def semantic_search(query: str) -> str:
    query_embedding = embedding_model.embed_query(query)
    response = index.query(
        vector=query_embedding,
        top_k=6,
        include_metadata=True  # assuming you stored your documents in metadata
    )
    matches = response.get('matches', [])
    if not matches:
        return "No matching documents found."
    
    results = []
    for match in matches:
        metadata = match.get('metadata', {})
        page_content = metadata.get('text', '')  # assuming your text is stored in 'text'
        results.append(page_content)
    
    return "\n\n".join(results)


# --- SETUP LANGCHAIN AGENT ---
# 1. Convert MCP Tool into LangChain Tool
mcp_tool = LangChainTool.from_function(
    func=semantic_search,
    name="semantic_search",
    description="Search documents semantically using Pinecone."
)

# 2. LLM for the agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Create the agent with the tool
agent = create_tool_calling_agent(llm=llm, tools=[mcp_tool], prompt=prompt)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=[mcp_tool], verbose=True)

# --- TEST QUERY ---
if __name__ == "__main__":
    while True:
        query = input("\nEnter your query: ")
        response = agent_executor.invoke({"input": query})
        print("\nAnswer:", response["output"])
