from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.agent import AgentExecutor
from langchain.agents import create_react_agent
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import tool


MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://tayyabaarooj:123abc@cluster0.4ogldto.mongodb.net/"  
DB_NAME = "ragdb"
COLLECTION_NAME = "documents"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"


client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
collection = client[DB_NAME][COLLECTION_NAME]

existing_docs_count = collection.count_documents({})


if existing_docs_count == 0:
    print("No embeddings found. Loading PDF and generating embeddings...")
    pdf_path="roadmap.py"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="all-minilm")
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )

else:
    print("Embeddings already exist in MongoDB. Skipping embedding step.")

    embeddings = OllamaEmbeddings(model="all-minilm")
    vector_store = MongoDBAtlasVectorSearch(
        embedding=embeddings,
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )



retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatOllama(model="mistral", temperature=0.4)



@tool
def retrieve_context(question: str) -> str:
    """Retrieve relevant context for a question."""
    docs = retriever.invoke(question)
    return "\n".join(f"- {doc.page_content}" for doc in docs)
tools = [retrieve_context]

react_prompt = PromptTemplate.from_template(
    """Answer questions using the following tools when needed: {tool_names}.

Tools:
{tools}

Use this format:

Question: {question}
Thought: [What you think you should do]
Action: [the action to take, like retrieve_context or none]
Action Input: [input to the action]
Observation: [result of the action]
... (repeat Thought/Action/Action Input/Observation as needed)
Final Answer: [your final answer ending with "Thanks for asking!"]

Begin!

{agent_scratchpad}
"""
)



agent = create_react_agent(llm=llm, tools=tools,prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


rag_template = """Answer based on the context. If the context lacks info, say so. Keep it short and end with "Thanks for asking!"

Context: {context}

Question: {question}

Answer:"""
rag_prompt = PromptTemplate.from_template(rag_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


question = "What is second task in week 1 in our database?"
response = rag_chain.invoke(question)
print(f"Question: {question}")
print(f"RAG Answer: {response}")

agent_response = agent_executor.invoke({"question": question})
print(f"\nReAct Agent Response: {agent_response['output']}")


client.close()
