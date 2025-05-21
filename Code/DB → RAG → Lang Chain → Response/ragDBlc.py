from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import AgentExecutor, create_react_agent
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
    return "\n\n".join(doc.page_content for doc in docs)

tools = [retrieve_context]

react_prompt = PromptTemplate.from_template(
    """You are an assistant that answers questions using the following tools when needed: {tool_names}.

Tools available:
{tools}

Question: {question}
Thought: [Your reasoning]
Action: [Call a tool like retrieve_context or none]
Observation: [Result of action, if any]
{agent_scratchpad}
Answer: [Your final answer]

If you don't need a tool, answer directly. End with "Thanks for asking!"
"""
)

agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)



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