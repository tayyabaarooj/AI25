from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
@tool
def classify_as_car(question: str) -> str:
    """classifies the question as 'car' if it is about car mechnaics or automobiles technology.
    
    Args:
    question: The input question to classify
    
    """
    print("Function 'classify_as_car' is used")


@tool
def classify_as_restaurant(question: str) -> str:
    """classifies the question as 'restaurant' if it is about food, restaurant services, cuisine, dining experince.
    
    Args:
    question: The input question to classify
    
    """
    print("Function 'classify_as_restaurant' is used")

@tool
def classify_as_tech(question: str) -> str:
    """classifies the question as 'tech' if it is about technology trends, gadgets, softwares etc. 
    
    Args:
    question: The input question to classify
    
    """
    print("Function 'classify_as_tech' is used")

llm= ChatOllama(model="mistral")
query="how do i identify the most delicious biryani?"
tools=[classify_as_car,classify_as_restaurant, classify_as_tech]
llm_with_tools =llm.bind_tools(tools)

messages= [HumanMessage(query)]
ai_msg=llm_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool={
        "classify_as_car":classify_as_car,
        "classify_as_restaurant":classify_as_restaurant,
        "classify_as_tech":classify_as_tech,
    }[tool_call["name"].lower()]
    tool_output=selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output,tool_call_id=tool_call["id"]))
    

messages

llm_with_tools.invoke(messages)
    