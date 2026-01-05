from langgraph.graph import START, StateGraph
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import sqlite3
import requests
import random
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()


llm = ChatOpenAI()

search_tool = DuckDuckGoSearchRun(region= 'us-en')


# 2. tool -> Calculator custom tool.

@tool
def calculators(first_num: float, second_num: float, operation: str)->dict:
    """ 
    Performe basic Arthmatic operation of two numbers.
    Supported operations:  add subtract mul and division.
    """ 

    try:

        if operation == "add":
            result = first_num + second_num
        
        elif operation == "subract":
            result = first_num - second_num

        elif operation == "multiply":
            result = first_num * second_num
        
        else:
            result = first_num / second_num

        return {'first_num': first_num, 'second_num': second_num, 'operation': operation, 'result': result}

    except Exception as e:
        return {'error': str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


tools = [search_tool, calculators, get_stock_price]


llm_with_tools = llm.bind_tools(tools=tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer or request to tool"""

    message = state['messages']
    response = llm_with_tools.invoke(message)
    return {'messages': [response]}


tool_node = ToolNode(tools=tools)


# define the graph

graph = StateGraph(ChatState)

# add node
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

#add edge:
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)

# Some modification.
graph.add_edge('tools', 'chat_node')


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)