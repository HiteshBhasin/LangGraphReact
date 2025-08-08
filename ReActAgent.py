from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage# foundational class for all message types in Langgraph
from langchain_core.messages import ToolMessage# Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage# Message for providing instructions to LLM
from langchain_core.tools import Tool, tool
from langgraph.graph.message import add_messages # reduceer function
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages] # this is saying preserve the state by overriding it

@tool
def add(a:int, b:int):
    """This is an addition function, which add two numbers"""
    return a+b

@tool
def subtract(a:int, b:int):
    """This is an subtraction function, which subtracts two numbers"""
    return a-b

@tool
def multiply(a:int, b:int):
    """This is an multiplication function, which multiplies two numbers"""
    return a*b

tools=[add,subtract,multiply]

model = ChatOpenAI(model="gpt-4.1").bind_tools(tools)

def model_call(state:AgentState)->AgentState:
    system_promt = SystemMessage(content= "you are my AI assistant please answer my queries to the best of your abilities.")
    response = model.invoke([system_promt]+state["messages"])
    return {"message": [response]}

def should_continue(state:AgentState)->AgentState:
    message = state["messages"]
    last_message = message[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end":END
    }
)

graph.add_edge("tools", "our_agent")
app= graph.compile()

def print_stream(stream):
    for s in stream:
        message =s["message"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()
            
inputs = {"message":[("user","Add 40+12 and then multiply the result by 6.")]}
print_stream(app.stream(inputs, stream_mode="values"))