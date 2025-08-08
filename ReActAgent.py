from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage# foundational class for all message types in Langgraph
from langchain_core.messages import ToolMessage# Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage# Message for providing instructions to LLM
from langchain_core.tools import Tool
from langgraph.graph.message import add_messages # reduceer function
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages] # this is saying preserve the state by overriding it

@tool
def add