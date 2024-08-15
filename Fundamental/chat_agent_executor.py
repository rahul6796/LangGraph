from langchain import hub 
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults


from typing import TypedDict, Annotated, Union, List, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langgraph.prebuilt.tool_executor import ToolExecutor,ToolInvocation
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import FunctionMessage
import json

from langchain_experimental.llms.ollama_functions import OllamaFunctions

import os
load_dotenv()

# Load the Ollama model and tools:

model = OllamaFunctions(model="mistral", format="json")
tools = [TavilySearchResults(max_results=1)]

model = model.bind_tools(
    tools=[
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, " "e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        }
    ],
    function_call={"name": "get_current_weather"},
)

tool_executor = ToolExecutor(tools)

## define the agent state:

class AgentState(TypedDict):
    message: Annotated[Sequence[BaseMessage], operator.add]



## deine the Node:

def should_continue(state):
    message = state['message']
    last_message = message[-1]

    if "function_call" not in last_message.additional_kwargs:
        return "end"
    else:
        return "continue"
    
def call_model(state):
    message = state['message']
    response = model.invoke(message)
    return {'message': response}


def call_tool(state):
    message = state['message']
    last_message = message[-1]
    action = ToolInvocation(
        tool=last_message.additional_kwargs['function_call']['name'],
        tool_input=json.loads(last_message.additional_kwargs['function_call']['arguments'])
    )

    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {'message': [function_message]}


# define the new Graph
workflow = StateGraph(AgentState)

## define the two node cycle between :
workflow.add_node('agent', call_model)
workflow.add_node('action', call_tool)

# set entry point:
workflow.set_entry_point('agent')


workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue":"action",
        "end":END
    }

)

workflow.add_edge('action', 'agent')

app = workflow.compile()



## Use It:s
from langchain_core.messages import HumanMessage

inputs = {'message': [HumanMessage(content='what is the weather of sf')]}

app.invoke(inputs)






